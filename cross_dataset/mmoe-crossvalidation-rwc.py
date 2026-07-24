"""
MMoE Music Structure Analysis — Option A: Partial Unfreeze
============================================================
Extends the SIMI framework (my-training-vocals.py) with Multi-gate Mixture-of-Experts.

Key change vs attention-towers-mmoe.py:
  The LAST SpecTNT layer of each expert is unfrozen and fine-tuned jointly with
  the gates and towers. This allows the expert representations to adapt to the
  multi-task signal instead of being rigidly frozen, breaking the representation
  saturation ceiling seen in all fully-frozen variants.

Architecture:
  - 4 frozen SIMI experts: Vocals+Mix, Drums+Mix, Bass+Mix, Others+Mix
  - Each expert's enc_FCT [B, T, 80] is extracted as its "opinion"
  - Two lightweight TaskGate networks (one for boundary, one for labeling)
    each produce a softmax weight over the 4 experts
  - The weighted-sum fused representation goes to two task-specific towers
  - All loss functions, metrics, decoding, checkpointing, CSV logging, and
    prediction saving are identical to my-training-vocals.py

Data requirements per song (same folder structure as SIMI preprocessing):
  mix:    *_spec.npy, *_chroma.npy
  vocal:  *_vocalspec.npy,  *_vocalchroma.npy
  drum:   *_drumspec.npy,   *_drumchroma.npy
  bass:   *_bassspec.npy,   *_basschroma.npy
  other:  *_otherspec.npy,  *_otherchroma.npy
  labels: *_boundary.npy, *_function.npy, *_section.npy

Usage:
  python mmoe_training.py

PATHS TO UPDATE (search for <- UPDATE):
  DATA_BASE_PATH, CONFIG_PATH, and 4 SIMI checkpoint paths
"""

import os
import json
import glob
import types
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d, filters
from tensorflow.keras import backend
import mir_eval
import librosa
import math
import pandas as pd
import csv
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# GPU setup  (same as your SIMI file)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Add this explicitly after os.environ:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

global_frame_size = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Import your existing SIMI classes
# ─────────────────────────────────────────────────────────────────────────────
# We need FunctionalSegmentModel and all its supporting classes.
# The cleanest way: just import them from your vocals training file.
# Make sure my_training_vocals.py is in the same directory (or on PYTHONPATH).
# NOTE: rename the file below to match your actual filename exactly.

from model import (          # <- UPDATE filename if different
    FunctionalSegmentModel,
    # utilities
    shape_list,
    peak_picking_MSAF,
    peak_picking_boeck,
    segmentFrame2interval,
    frame2interval,
    get_spectral_mask,
    get_temporal_mask,
    function_dict,
    class_conversion,
    format_cluster_sequence,
    print_temp,
    print_confusion_matrix,
    color_text,
    ImprovedEarlyStopping,
    BestTestPredictionSaver,
    load_dataset_config,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH = "/Scratch/repository/msa/MSATSUNGPING/"
CONFIG_PATH    = "/Scratch/repository/msa/MSATSUNGPING/my_dataset_selection_beatles_salami_rwc.json"

# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset validation setting  <- CHANGE THIS BEFORE EACH RUN
# ─────────────────────────────────────────────────────────────────────────────
#   'rwc'     -> Set 1: train Beatles + SALAMI,  test RWC
#   'beatles' -> Set 2: train SALAMI  + RWC,     test Beatles
#   'salami'  -> Set 3: train Beatles + RWC,     test SALAMI
HELD_OUT = 'rwc'   # <- CHANGE THIS per run

# ─────────────────────────────────────────────────────────────────────────────
# RWC-trained expert checkpoints
# ─────────────────────────────────────────────────────────────────────────────
CKPT_VOCALS = '/Scratch/repository/msa/MSATSUNGPING/vocals_F_rwc/best_models/best-epoch-58-82'
CKPT_DRUMS  = '/Scratch/repository/msa/MSATSUNGPING/drums_F_rwc/best_models/best-epoch-69-94'
CKPT_BASS   = '/Scratch/repository/msa/MSATSUNGPING/bass_F_rwc/best_models/best-epoch-69-91'
CKPT_OTHER  = '/Scratch/repository/msa/MSATSUNGPING/others_F_rwc/best_models/best-epoch-44-67'



# ─────────────────────────────────────────────────────────────────────────────
# Helper: patch a loaded FunctionalSegmentModel with an _encode() method
# that returns enc_FCT  (the rich [B, T, 80] representation used by MMoE)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_frozen_method(self, spec, chromagram, stem_spec, stem_chromagram, valid_len):
    """
    Stage 1 of Option A encoding — everything BEFORE the last SpecTNT layer.
    All operations here are frozen; call this OUTSIDE GradientTape to avoid
    storing activations for backprop (saves ~4x GPU memory).

    Returns (enc_S, enc_FCT, stem_features) as intermediate tensors that will
    be passed into _encode_last_layer_method inside the tape.
    """
    # Log compression
    spec        = tf.math.log(1 + 100 * tf.nn.relu(spec        + 80))
    spec        = tf.expand_dims(spec,        axis=-1)          # [B, T, 80, 1]
    chromagram  = tf.expand_dims(chromagram,  axis=-1)          # [B, T, 12, 1]
    stem_spec   = tf.math.log(1 + 100 * tf.nn.relu(stem_spec   + 80))

    # All frozen — training=False throughout
    stem_features = self.stem_encoder(stem_spec, stem_chromagram, training=False)
    spec          = self.spec_prenorm(spec, valid_len, training=False)

    enc_spec   = self.specCNNBase(spec,         valid_len, training=False)
    enc_spec   = self.specCNN(enc_spec,          valid_len, training=False)
    enc_chroma = self.chromaCNNBase(chromagram,  valid_len, training=False)
    enc_chroma = self.chromaCNN(enc_chroma,       valid_len, training=False)
    enc_spec   = self.spec_transition(enc_spec)
    enc_spec   = self.spec_transition_norm(enc_spec)
    enc_chroma = self.chroma_transition(enc_chroma)
    enc_chroma = self.chroma_transition_norm(enc_chroma)

    enc_S   = tf.concat([enc_spec, enc_chroma], axis=2)                      # [B,T,92,d/2]
    enc_FCT = self.fct_dense(tf.reduce_mean(enc_S, axis=2, keepdims=True))   # [B,T,1,d]
    enc_FCT = self.fct_dense_norm(enc_FCT)
    enc_S  += self.fpe_S
    enc_FCT += self.fpe_FCT

    # Run all SpecTNT layers EXCEPT the last one (all frozen)
    for specTNT in self.specTNT_layers[:-1]:
        if self.return_maps:
            enc_S, enc_FCT, _, _ = specTNT(enc_S, enc_FCT, stem_features,
                                            valid_len=valid_len, training=False)
        else:
            enc_S, enc_FCT, _ = specTNT(enc_S, enc_FCT, stem_features,
                                         valid_len=valid_len, training=False)

    # stop_gradient so the tape doesn't track back through these frozen tensors
    return (tf.stop_gradient(enc_S),
            tf.stop_gradient(enc_FCT),
            tf.stop_gradient(stem_features))


def _encode_last_layer_method(self, enc_S, enc_FCT, stem_features, valid_len, training=False):
    """
    Stage 2 of Option A encoding — ONLY the last (unfrozen) SpecTNT layer.
    Call this INSIDE GradientTape so its weights receive gradients.
    Takes the stop_gradient outputs from _encode_frozen_method as inputs.

    Returns enc_FCT [B, T, 80].
    """
    last_specTNT = self.specTNT_layers[-1]
    if self.return_maps:
        enc_S, enc_FCT, _, _ = last_specTNT(enc_S, enc_FCT, stem_features,
                                             valid_len=valid_len, training=training)
    else:
        enc_S, enc_FCT, _ = last_specTNT(enc_S, enc_FCT, stem_features,
                                          valid_len=valid_len, training=training)

    return tf.squeeze(enc_FCT, axis=2)   # [B, T, 80]


def patch_simi_model_with_encode(model):
    """Attach _encode_frozen() and _encode_last_layer() to a loaded FunctionalSegmentModel."""
    model._encode_frozen     = types.MethodType(_encode_frozen_method,     model)
    model._encode_last_layer = types.MethodType(_encode_last_layer_method, model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load and freeze one SIMI expert
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_expert(checkpoint_path, stem_name):
    """
    Load a pretrained FunctionalSegmentModel.
    
    Freezing strategy (Option A):
      - ALL layers frozen EXCEPT the last SpecTNT layer (specTNT_layers[-1])
      - The last SpecTNT layer fine-tunes jointly with the MMoE gates+towers
      - This lets each expert's representation adapt to the multi-task signal
        without losing the knowledge encoded in the earlier layers
    """
    print(f"🔄 Loading {stem_name} expert from {checkpoint_path} ...")
    model = FunctionalSegmentModel(
        max_len=935, n_units=80, n_heads=8, n_layers=2,
        cnn_dropout_rate=0.0, attn_dropout_rate=0.0,
        use_boundary_fusion=False,
    )
    # Build the model with dummy data so weights exist before restoring
    dummy_spec   = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_len    = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_spec, dummy_chroma, dummy_len, training=False)

    # Restore checkpoint
    ckpt   = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path).expect_partial()
    try:
        status.assert_existing_objects_matched()
        print(f"  ✅ {stem_name}: checkpoint restored")
    except Exception as e:
        print(f"  ⚠️  {stem_name}: {e}")

    # ── Selective freezing: freeze everything first ──
    for layer in model.layers:
        layer.trainable = False
    model.trainable = True   # keep model itself trainable so sub-layers can be

    # ── Then unfreeze only the LAST SpecTNT layer ──
    last_spectnt = model.specTNT_layers[-1]
    last_spectnt.trainable = True
    n_unfrozen = sum(tf.size(v).numpy() for v in last_spectnt.trainable_variables)
    print(f"  🔓 {stem_name}: last SpecTNT layer unfrozen ({n_unfrozen:,} params)")
    print(f"  🔒 {stem_name}: all other layers remain frozen")

    # Attach the _encode method
    patch_simi_model_with_encode(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific gating network
# ─────────────────────────────────────────────────────────────────────────────

class TaskGate(tf.keras.layers.Layer):
    """
    Implements  g^k(x) = softmax(W_gk * x)  from the MMoE paper.
    One instance per task (boundary gate and label gate).

    Input : mix_features  [B, T, d_model=80]
    Output: gate_weights  [B, T, n_experts=4]  (softmax, sums to 1)

    We add a small hidden layer with ReLU before the softmax because the
    relationship between audio context and expert trustworthiness is non-linear.
    Dropout (rate=0.1) prevents overfitting on the small dataset.
    """
    def __init__(self, n_experts=4, d_model=80, hidden_units=64,
                 dropout_rate=0.1, name='task_gate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_experts = n_experts
        self.hidden    = tf.keras.layers.Dense(hidden_units, activation='relu',
                                               name=f'{name}_hidden')
        self.dropout   = tf.keras.layers.Dropout(dropout_rate,
                                                  name=f'{name}_dropout')
        self.out       = tf.keras.layers.Dense(n_experts,
                                               name=f'{name}_out')

    def call(self, x, training=False):
        """x: [B, T, d] -> [B, T, n_experts]"""
        h = self.hidden(x)                          # [B, T, hidden_units]
        h = self.dropout(h, training=training)      # [B, T, hidden_units]
        logits = self.out(h)                        # [B, T, n_experts]
        return tf.nn.softmax(logits, axis=-1)       # [B, T, n_experts]


class LightSelfAttention(tf.keras.layers.Layer):
    """
    Single-head self-attention over a sequence.
    Input/output shape: [B, T, d_model]
    Uses causal=False (bidirectional) — we want to see both sides of a boundary.
    """
    def __init__(self, d_model=80, num_heads=4, dropout_rate=0.1, name='light_attn', **kwargs):
        super().__init__(name=name, **kwargs)
        self.attn    = tf.keras.layers.MultiHeadAttention(
                           num_heads=num_heads,
                           key_dim=d_model // num_heads,   # 20 per head
                           dropout=dropout_rate,
                           name=f'{name}_mha'
                       )
        self.norm    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len=None, training=False):
        # Build padding mask so attention ignores padded frames
        if valid_len is not None:
            maxlen   = tf.shape(x)[1]
            pad_mask = tf.sequence_mask(valid_len, maxlen=maxlen)  # [B, T] bool
            # MultiHeadAttention expects mask shape [B, 1, 1, T]
            pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
        else:
            pad_mask = None

        attn_out = self.attn(x, x, attention_mask=pad_mask, training=training)
        attn_out = self.dropout(attn_out, training=training)
        return self.norm(x + attn_out)   # residual connection + LayerNorm


# ─────────────────────────────────────────────────────────────────────────────
# MMoE Music Structure Model
# ─────────────────────────────────────────────────────────────────────────────

class MMoEMusicModel(tf.keras.Model):
    """
    Multi-gate Mixture-of-Experts model for Music Structure Analysis.

    Frozen experts (4 SIMI models) + 2 trainable gates + 2 trainable towers.

    Forward pass:
      1. Run all 4 frozen experts -> enc_FCT_e  [B, T, 80]  each
      2. Run vocals enc_FCT through boundary gate -> g_b  [B, T, 4]
         Run vocals enc_FCT through label    gate -> g_l  [B, T, 4]
      3. fused_b = sum_e  g_b_e * enc_FCT_e         [B, T, 80]
         fused_l = sum_e  g_l_e * enc_FCT_e         [B, T, 80]
      4. logits_boun = boundary_tower(fused_b)       [B, T]
         logits_func = label_tower(fused_l)          [B, T, 7]

    Trainable components:
      - boundary_gate, label_gate
      - boundary_tower (boun1, boun2, boun_out)
      - label_tower    (func1,  func2,  func_out)

    Everything else (metric tracking, loss functions, decoding, result storage)
    is copied verbatim from FunctionalSegmentModel so outputs are comparable.
    """

    def __init__(
        self,
        expert_vocals,
        expert_drums,
        expert_bass,
        expert_other,
        n_units=80,
        n_classes=7,
        gate_hidden=64,
        gate_dropout=0.1,
        tower_dropout=0.3,
        steps_per_epoch=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ── frozen experts ────────────────────────────────────────────────
        self.experts      = [expert_vocals, expert_drums, expert_bass, expert_other]
        self.expert_names = ['vocals', 'drums', 'bass', 'other']
        self.n_experts    = len(self.experts)

        # ── gates ─────────────────────────────────────────────────────────
        self.boundary_gate = TaskGate(n_experts=self.n_experts, d_model=n_units,
                                      hidden_units=gate_hidden,
                                      dropout_rate=gate_dropout,
                                      name='boundary_gate')
        self.label_gate    = TaskGate(n_experts=self.n_experts, d_model=n_units,
                                      hidden_units=gate_hidden,
                                      dropout_rate=gate_dropout,
                                      name='label_gate')

        # ── boundary tower  (mirrors SIMI: boun1->boun2->boun_out Conv1D) ──
        
        self.boun_attn  = LightSelfAttention(d_model=n_units, num_heads=4,
                                     dropout_rate=tower_dropout, name='boun_attn')
        self.boun_proj  = tf.keras.layers.Dense(n_units, activation='relu', name='boun_proj')
        self.boun_proj2 = tf.keras.layers.Dense(n_units, activation='relu', name='boun_proj2')  # ← added
        self.boun_out   = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same', name='boun_out')
        self.boun_drop  = tf.keras.layers.Dropout(tower_dropout)
        # ── label tower  (mirrors SIMI: func1->func2->func_out Conv1D) ───

        self.func_attn  = LightSelfAttention(d_model=n_units, num_heads=4,
                                     dropout_rate=tower_dropout, name='func_attn')
        self.func_proj  = tf.keras.layers.Dense(n_units, activation='relu', name='func_proj')
        self.func_proj2 = tf.keras.layers.Dense(n_units, activation='relu', name='func_proj2')  # ← added
        self.func_out   = tf.keras.layers.Conv1D(n_classes, kernel_size=7, padding='same', name='func_out')
        self.func_drop  = tf.keras.layers.Dropout(tower_dropout)

        # ── bookkeeping (identical to FunctionalSegmentModel) ─────────────
        self.n_units         = n_units
        self.n_classes       = n_classes
        self.steps_per_epoch = steps_per_epoch
        self.optimizer       = None
        self.flag            = True

        self.w_b = 18
        self.w_f = 2

        self.confusion_matrix_train_max  = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_max   = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_boun  = tf.zeros([n_classes, n_classes], tf.int32)

        self.result = {k: [] for k in [
            'Acc_max', 'Acc_smooth',
            'P_seg',  'R_seg',  'F1_seg',
            'P_seg3', 'R_seg3', 'F1_seg3',
            'P_pair', 'R_pair', 'F1_pair',
            'loss', 'loss_b', 'loss_f', 'gate_entropy', 'ssm_loss', 'gate_sim',
        ]}
        self.temp = {k: [] for k in [
            'b_ref', 'b_est', 'matched',
            'n_b_ref', 'n_b_est', 'n_matched',
            'b_ref_in_second', 'b_est_in_second',
            'f_ref', 'f_est',
        ]}
        self.gate_log = []

        print("\n" + "="*80)
        print("MMoE MODEL CONFIGURATION")
        print("="*80)
        print(f"  Experts     : {self.expert_names}")
        print(f"  Gate hidden : {gate_hidden}   dropout: {gate_dropout}")
        print(f"  Tower drop  : {tower_dropout}")
        print(f"  n_units     : {n_units}   n_classes: {n_classes}")
        print("="*80 + "\n")

    # ── forward pass ──────────────────────────────────────────────────────────

    def call(self, spec, chromagram,
             vocal_spec,  vocal_chromagram,
             drum_spec,   drum_chromagram,
             bass_spec,   bass_chromagram,
             other_spec,  other_chromagram,
             valid_len, training=False):
        """
        Returns logits_boun [B,T], logits_func [B,T,7],
                gate_b [B,T,4], gate_l [B,T,4]
        """

        # 1. Collect expert representations (two-stage, inference only)
        stem_inputs = [
            (vocal_spec,  vocal_chromagram),
            (drum_spec,   drum_chromagram),
            (bass_spec,   bass_chromagram),
            (other_spec,  other_chromagram),
        ]
        expert_feats = []
        for expert, (s, c) in zip(self.experts, stem_inputs):
            enc_S, enc_FCT, stem_feats = expert._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            feat = expert._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                             valid_len, training=training)
            expert_feats.append(feat)
        # Stack -> [B, T, n_experts, 80]
        stacked = tf.stack(expert_feats, axis=2)

        # 2. Use the vocals expert features as gate context
        #    (vocals is the strongest single expert in SIMI)
        # gate_context = expert_feats[0]                                          # [B, T, 80]

        # Replace line 384 with:
        gate_context_feats = tf.reduce_mean(tf.stack(expert_feats, axis=2), axis=2)  # [B, T, 80]
        T = tf.shape(stacked)[1]
        positions = tf.cast(tf.range(T), tf.float32) / tf.cast(T, tf.float32)
        positions = tf.tile(positions[tf.newaxis, :, tf.newaxis],
                            [tf.shape(stacked)[0], 1, 1])
        gate_context = tf.concat([gate_context_feats, positions], axis=-1)           # [B, T, 81]
        gate_b = self.boundary_gate(gate_context, training=training)            # [B, T, 4]
        gate_l = self.label_gate(gate_context,    training=training)            # [B, T, 4]

        # 3. Weighted fusion
        fused_b = tf.reduce_sum(
            tf.expand_dims(gate_b, axis=-1) * stacked, axis=2
        )   # [B, T, 80]
        fused_l = tf.reduce_sum(
            tf.expand_dims(gate_l, axis=-1) * stacked, axis=2
        )   # [B, T, 80]

        # 4. Boundary tower
        x = self.boun_attn(fused_b, valid_len=valid_len, training=training)    # [B, T, 80]
        x = self.boun_drop(x, training=training)
        x = self.boun_proj(x)                                                   # [B, T, 80]
        logits_boun = tf.squeeze(self.boun_out(x), axis=2)                     # [B, T]

        # 5. Label tower
        y = self.func_attn(fused_l, valid_len=valid_len, training=training)    # [B, T, 80]
        y = self.func_drop(y, training=training)
        y = self.func_proj(y)                                                   # [B, T, 80]
        logits_func = self.func_out(y)                                          # [B, T, 7]

        return logits_boun, logits_func, gate_b, gate_l
        

    # ── train step ────────────────────────────────────────────────────────────

    def call_from_encoded(self, stacked, valid_len, training=False):
        # gate_context = stacked[:, :, 0, :]  # vocals expert [B, T, 80]
        # Replace line 414 with:
        gate_context_feats = tf.reduce_mean(stacked, axis=2)                          # [B, T, 80]
        T = tf.shape(stacked)[1]
        positions = tf.cast(tf.range(T), tf.float32) / tf.cast(T, tf.float32)        # [T]
        positions = tf.tile(positions[tf.newaxis, :, tf.newaxis],
                            [tf.shape(stacked)[0], 1, 1])                             # [B, T, 1]
        gate_context = tf.concat([gate_context_feats, positions], axis=-1)           # [B, T, 81]

        gate_b = self.boundary_gate(gate_context, training=training)  # [B, T, 4]
        gate_l = self.label_gate(gate_context,    training=training)  # [B, T, 4]

        mixed_b = tf.reduce_sum(stacked * gate_b[:, :, :, tf.newaxis], axis=2)  # [B, T, 80]
        mixed_l = tf.reduce_sum(stacked * gate_l[:, :, :, tf.newaxis], axis=2)  # [B, T, 80]

        # ── boundary tower ──────────────────────────────────────────────

        x_b = self.boun_attn(mixed_b, valid_len=valid_len, training=training)  # [B, T, 80]
        x_b = self.boun_drop(x_b, training=training)
        x_b = self.boun_proj(x_b)                                               # [B, T, 80]
        x_b = self.boun_proj2(x_b)                                              # [B, T, 80] ← added
        logits_boun = tf.squeeze(self.boun_out(x_b), axis=-1)                  # [B, T]

        # ── label tower ─────────────────────────────────────────────────

        x_f = self.func_attn(mixed_l, valid_len=valid_len, training=training)  # [B, T, 80]
        x_f = self.func_drop(x_f, training=training)
        x_f = self.func_proj(x_f)                                               # [B, T, 80]
        x_f = self.func_proj2(x_f)                                              # [B, T, 80] ← added
        logits_func = self.func_out(x_f)                                        # [B, T, 7]

        # return logits_boun, logits_func, gate_b, gate_l
        return logits_boun, logits_func, gate_b, gate_l, mixed_b
    

    def train_step(self, data):
        (spec, chromagram,
        vocal_spec,  vocal_chromagram,
        drum_spec,   drum_chromagram,
        bass_spec,   bass_chromagram,
        other_spec,  other_chromagram,
        valid_len, boun_ref, func_ref, sec_ref) = data

        stems = [
            (vocal_spec, vocal_chromagram),
            (drum_spec,  drum_chromagram),
            (bass_spec,  bass_chromagram),
            (other_spec, other_chromagram),
        ]

        # ── Stage 1: frozen layers OUTSIDE tape ───────────────────────────────
        # Runs CNN + stem encoder + all SpecTNT layers except the last one.
        # stop_gradient applied inside _encode_frozen — no activations stored.
        intermediates = []
        for exp, (s, c) in zip(self.experts, stems):
            enc_S, enc_FCT, stem_feats = exp._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            intermediates.append((enc_S, enc_FCT, stem_feats))
        # ──────────────────────────────────────────────────────────────────────

        with tf.GradientTape() as tape:
            # ── Stage 2: last SpecTNT layer INSIDE tape — receives gradients ──
            enc_list = []
            for exp, (enc_S, enc_FCT, stem_feats) in zip(self.experts, intermediates):
                feat = exp._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                              valid_len, training=True)
                enc_list.append(feat)
            stacked = tf.stack(enc_list, axis=2)  # [B, T, 4, 80]
            # logits_boun, logits_func, gate_b, gate_l = self.call_from_encoded(
            #     stacked, valid_len, training=True
            # )

            logits_boun, logits_func, gate_b, gate_l, mixed_b = self.call_from_encoded(
                stacked, valid_len, training=True
            )

            prob_boun       = tf.nn.sigmoid(logits_boun)
            boun_est        = self.decode_boundary(prob_boun, valid_len)
            func_est_max    = tf.argmax(logits_func, axis=-1, output_type=tf.int32)
            func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len)

            self.confusion_matrix_train_max  += self.compute_confusion_matrix(func_ref, func_est_max)
            self.confusion_matrix_train_boun += self.compute_confusion_matrix(func_ref, func_est_smooth)

            ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
            ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)

            if self.flag:
                print('ce_b', ce_b.numpy())
                print('ce_f', ce_f.numpy())
                self.flag = False

            # loss = ce_b + ce_f

            # REPLACE with:
            gate_entropy_b = -tf.reduce_mean(
                tf.reduce_sum(gate_b * tf.math.log(gate_b + 1e-8), axis=-1)
            )
            gate_entropy_l = -tf.reduce_mean(
                tf.reduce_sum(gate_l * tf.math.log(gate_l + 1e-8), axis=-1)
            )
            # gate_entropy = gate_entropy_b + gate_entropy_l  # encourage spreading weights

            gate_entropy   = gate_entropy_b + gate_entropy_l

            gate_cosine_sim = tf.reduce_mean(
                tf.reduce_sum(gate_b * gate_l, axis=-1) /
                (tf.norm(gate_b, axis=-1) * tf.norm(gate_l, axis=-1) + 1e-8)
            )

            # SSM loss — teach fused_b that same-section frames should be similar
            gt_ssm   = self.compute_gt_ssm(func_ref, valid_len)    # [B, T, T]
            pred_ssm = self.compute_pred_ssm(mixed_b, valid_len)   # [B, T, T]

            # Normalize pred_ssm from [-1,1] to [0,1] for MSE with gt_ssm
            pred_ssm_01 = (pred_ssm + 1.0) / 2.0
            ssm_loss = tf.reduce_mean(tf.square(pred_ssm_01 - gt_ssm))

            # loss = ce_b + ce_f + 0.01 * gate_entropy + 0.1 * ssm_loss
            loss = ce_b + ce_f + 0.005 * gate_entropy + 0.02 * ssm_loss   # SSM weight reduced 0.1→0.02
            
            # loss = ce_b + ce_f + 0.01 * gate_entropy
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)

        if not hasattr(self, 'optimizer') or self.optimizer is None:
            # Two separate optimizers:
            #   optimizer       — gates + towers          (higher LR)
            #   optimizer_expert — unfrozen SpecTNT layers (lower LR, 10x smaller)
            # Using a lower LR for the expert fine-tuning protects pretrained
            # weights from being overwritten by the MMoE training signal.
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-4, clipnorm=1.0, epsilon=1e-7
            )
            self.optimizer_expert = tf.keras.optimizers.Adam(
                learning_rate=1e-5, clipnorm=0.5, epsilon=1e-7
            )

        current_epoch = tf.cast(
            self.optimizer.iterations // self.steps_per_epoch, tf.float32
        )

        def lr_schedule():
            # Gates + towers LR schedule:
            #   Epochs  0-30:  warm-up  1e-4 → 5e-4   (slow ramp, expert layers need time)
            #   Epochs 30-150: hold     5e-4            (main training phase)
            #   Epochs 150-220: linear decay 5e-4 → 5e-5
            #   Epochs 220+:   fine-tune 5e-5
            return tf.cond(
                current_epoch < 30,
                lambda: 1e-4 + (4e-4) * (current_epoch / 30.0),
                lambda: tf.cond(
                    current_epoch < 150,
                    lambda: 5e-4,
                    lambda: tf.cond(
                        current_epoch < 220,
                        lambda: 5e-4 - (4.5e-4) * ((current_epoch - 150.0) / 70.0),
                        lambda: 5e-5
                    )
                )
            )

        def lr_schedule_expert():
            # Expert last-layer: always 10x lower than gate/tower LR
            #   Epochs  0-30:  warm-up  1e-5 → 5e-5
            #   Epochs 30-150: hold     5e-5
            #   Epochs 150-220: linear decay 5e-5 → 5e-6
            #   Epochs 220+:   fine-tune 5e-6
            return tf.cond(
                current_epoch < 30,
                lambda: 1e-5 + (4e-5) * (current_epoch / 30.0),
                lambda: tf.cond(
                    current_epoch < 150,
                    lambda: 5e-5,
                    lambda: tf.cond(
                        current_epoch < 220,
                        lambda: 5e-5 - (4.5e-5) * ((current_epoch - 150.0) / 70.0),
                        lambda: 5e-6
                    )
                )
            )

        self.optimizer.learning_rate.assign(lr_schedule())
        self.optimizer_expert.learning_rate.assign(lr_schedule_expert())

        # Split gradients: expert last-layer vars vs gate/tower vars
        # Note: fpe_S and fpe_FCT are frozen positional embeddings that appear
        # in trainable_variables but receive no gradient — filter them out here
        # to silence the optimizer warning.
        expert_vars = []
        for exp in self.experts:
            expert_vars.extend(exp.specTNT_layers[-1].trainable_variables)
        expert_var_ids = {id(v) for v in expert_vars}

        gate_tower_pairs = [(g, v) for g, v in zip(grads, trainable_vars)
                            if id(v) not in expert_var_ids and g is not None]
        expert_pairs     = [(g, v) for g, v in zip(grads, trainable_vars)
                            if id(v) in expert_var_ids and g is not None]

        if gate_tower_pairs:
            self.optimizer.apply_gradients(gate_tower_pairs)
        if expert_pairs:
            self.optimizer_expert.apply_gradients(expert_pairs)

        score_dict = self.compute_classification_score(func_ref, func_est_max,     valid_len, key='Acc_max')
        score_dict.update(self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth'))
        score_dict.update(self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size))
        score_dict.update(self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size))

        score_dict.update({'loss': loss, 'loss_b': ce_b, 'loss_f': ce_f,
            'gate_entropy': gate_entropy, 'ssm_loss': ssm_loss,
            'gate_sim': gate_cosine_sim})
        
        [self.result[k].append(v) for k, v in score_dict.items()]

        return boun_est, func_est_smooth
    

    
    # ── test step ─────────────────────────────────────────────────────────────
    

    def test_step(self, data, log_gates=False):
        (spec, chromagram,
        vocal_spec,  vocal_chromagram,
        drum_spec,   drum_chromagram,
        bass_spec,   bass_chromagram,
        other_spec,  other_chromagram,
        valid_len, boun_ref, func_ref, sec_ref) = data

        # ── Encode all 4 experts (two-stage, no tape needed in test) ──────────
        stems = [
            (vocal_spec, vocal_chromagram),
            (drum_spec,  drum_chromagram),
            (bass_spec,  bass_chromagram),
            (other_spec, other_chromagram),
        ]

        enc_list = []
        for exp, (s, c) in zip(self.experts, stems):
            enc_S, enc_FCT, stem_feats = exp._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            feat = exp._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                          valid_len, training=False)
            enc_list.append(feat)
        stacked = tf.stop_gradient(tf.stack(enc_list, axis=2))  # [B, T, 4, 80]
        # ─────────────────────────────────────────────────────────────────────

        logits_boun, logits_func, gate_b, gate_l, mixed_b = self.call_from_encoded(
            stacked, valid_len, training=False
        )



        prob_boun       = tf.nn.sigmoid(logits_boun)
        boun_est        = self.decode_boundary(prob_boun, valid_len)
        func_est_max    = tf.argmax(logits_func, axis=-1, output_type=tf.int32)
        func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len)

        self.confusion_matrix_test_max  += self.compute_confusion_matrix(func_ref, func_est_max)
        self.confusion_matrix_test_boun += self.compute_confusion_matrix(func_ref, func_est_smooth)

        ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
        ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)

        gate_entropy_b = -tf.reduce_mean(
        tf.reduce_sum(gate_b * tf.math.log(gate_b + 1e-8), axis=-1)
        )
        gate_entropy_l = -tf.reduce_mean(
            tf.reduce_sum(gate_l * tf.math.log(gate_l + 1e-8), axis=-1)
        )
        gate_entropy = gate_entropy_b + gate_entropy_l

        # Monitor gate collapse — if gate_sim approaches 1.0, gates have collapsed
        gate_cosine_sim = tf.reduce_mean(
            tf.reduce_sum(gate_b * gate_l, axis=-1) /
            (tf.norm(gate_b, axis=-1) * tf.norm(gate_l, axis=-1) + 1e-8)
        )
        
        gt_ssm      = self.compute_gt_ssm(func_ref, valid_len)
        pred_ssm    = self.compute_pred_ssm(mixed_b, valid_len)
        pred_ssm_01 = (pred_ssm + 1.0) / 2.0
        ssm_loss    = tf.reduce_mean(tf.square(pred_ssm_01 - gt_ssm))
        loss = ce_b + ce_f

        score_dict = self.compute_classification_score(func_ref, func_est_max,     valid_len, key='Acc_max')
        score_dict.update(self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth'))
        score_dict.update(self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size))
        score_dict.update(self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size))
        score_dict.update({'loss': loss, 'loss_b': ce_b, 'loss_f': ce_f,
           'gate_entropy': gate_entropy, 'ssm_loss': ssm_loss,
           'gate_sim': gate_cosine_sim})
        [self.result[k].append(v) for k, v in score_dict.items()]

        # ── Optional: log per-song gate weights ──────────────────────────────
        if log_gates:
            sec_np    = sec_ref.numpy()
            gate_b_np = gate_b.numpy()
            gate_l_np = gate_l.numpy()
            vlen_np   = valid_len.numpy()
            for i in range(gate_b_np.shape[0]):
                l = int(vlen_np[i])
                avg_b   = gate_b_np[i, :l, :].mean(axis=0)
                avg_l   = gate_l_np[i, :l, :].mean(axis=0)
                song_id = sec_np[i].decode('utf-8') if isinstance(sec_np[i], bytes) else str(sec_np[i])
                self.gate_log.append({
                    'song':              song_id,
                    'boun_gate_vocals':  avg_b[0], 'boun_gate_drums': avg_b[1],
                    'boun_gate_bass':    avg_b[2], 'boun_gate_other': avg_b[3],
                    'label_gate_vocals': avg_l[0], 'label_gate_drums': avg_l[1],
                    'label_gate_bass':   avg_l[2], 'label_gate_other': avg_l[3],
                })
        # ─────────────────────────────────────────────────────────────────────

        return boun_est, func_est_smooth

    # ── helper methods (verbatim copies from FunctionalSegmentModel) ──────────

    def clear_result(self):
        self.result = {k: [] for k in self.result.keys()}
        self.temp   = {k: [] for k in self.temp.keys()}
        self.confusion_matrix_train_max  = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_max   = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_boun  = tf.zeros([self.n_classes, self.n_classes], tf.int32)

    def average_result(self):
        result_dict = {}
        for k, v in self.result.items():
            if len(v) > 0:
                try:
                    concatenated = tf.concat(v, axis=0)
                    result_dict[k] = tf.reduce_mean(concatenated)
                except tf.errors.InvalidArgumentError:
                    stacked = tf.stack(v)
                    result_dict[k] = tf.reduce_mean(stacked)
        return result_dict

    def bce_from_logits(self, gt, logits, valid_len, pos_weight=0.3):
        gt_expanded = self.expand_boundary(gt, valid_len, value=0.5)
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(gt)[1], dtype=tf.float32)
        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_expanded, logits, pos_weight=pos_weight)
        loss = tf.reduce_sum(wbce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    def cce_from_logits(self, gt, logits, valid_len):
        # Same class weights as your SIMI file
        weights = tf.constant([
            1.03,   # intro
            0.27,   # verse
            0.47,   # chorus
            0.94,   # bridge
            0.88,   # inst
            3.00,   # outro
            1.68,   # silence
        ], tf.float32)
        seq_mask    = tf.sequence_mask(valid_len, maxlen=tf.shape(gt)[1], dtype=tf.float32)
        gt_onehot   = tf.one_hot(gt, depth=self.n_classes)
        ce          = tf.nn.softmax_cross_entropy_with_logits(gt_onehot, logits)
        cw          = tf.gather(weights, gt)
        weighted_ce = ce * cw
        loss = tf.reduce_sum(weighted_ce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    def decode_boundary(self, prob_boun, valid_len, method='librosa'):
        assert method in ['msaf', 'librosa', 'boeck']
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(prob_boun)[1], dtype=tf.float32)
        prob_boun *= seq_mask
        prob_boun_numpy = prob_boun.numpy()
        peaks = np.zeros_like(prob_boun_numpy, dtype=np.int32)
        if method == 'msaf':
            peak_indices = [peak_picking_MSAF(seq, median_len=7, offset_rel=0.05, sigma=4)
                            for seq in prob_boun_numpy]
        elif method == 'librosa':
            peak_indices = [
                librosa.util.peak_pick(seq, pre_max=10, post_max=10, pre_avg=20,
                                       post_avg=10, delta=0.03, wait=10)
                for seq in prob_boun_numpy
            ]
            peak_indices = [ids.astype(int) for ids in peak_indices]
        elif method == 'boeck':
            peak_indices = [
                peak_picking_boeck(seq, threshold=0.01, fps=2, combine=10,
                                   pre_max=10, post_max=10, pre_avg=20, post_avg=10)
                for seq in prob_boun_numpy
            ]
        for i in range(prob_boun_numpy.shape[0]):
            peaks[i, peak_indices[i]] = 1
        peaks[:, 0] = 1
        return tf.constant(peaks, tf.int32) * tf.cast(seq_mask, tf.int32)

    def decode_labeling(self, boun_est, logits_func, valid_len):
        boun_est  = boun_est.numpy()
        prob_func = tf.nn.sigmoid(logits_func).numpy()
        valid_len = valid_len.numpy()
        max_len   = valid_len.max()
        func_est  = []
        for i in range(valid_len.shape[0]):
            l  = valid_len[i]
            b_i = np.where(np.equal(boun_est[i, :l], 1))[0]
            segments  = [s for s in np.split(prob_func[i, :l], indices_or_sections=b_i) if len(s)]
            centroids = np.stack([np.sum(seg, axis=0) for seg in segments])
            clusters  = np.argmax(centroids, axis=-1)
            label_frame = np.array([c for (seg, c) in zip(segments, clusters) for _ in range(len(seg))])
            if l < max_len:
                label_frame = np.pad(label_frame, (0, max_len - l), 'constant',
                                     constant_values=self.n_classes - 1)
            func_est.append(label_frame)
        return tf.constant(func_est)

    def expand_boundary(self, boundary, valid_len, value=0.5, size=3):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boundary)[1], dtype=tf.float32)
        boundary = tf.cast(boundary, tf.float32)
        filter_  = tf.ones([size, 1, 1])
        boundary_expanded = tf.squeeze(
            tf.nn.conv1d(boundary[:, :, tf.newaxis], filters=filter_, stride=1, padding='SAME'),
            axis=-1
        )
        cond = tf.logical_and((boundary_expanded != boundary),
                              tf.logical_not(tf.cast(boundary, tf.bool)))
        boundary_expanded = tf.where(cond, value, boundary)
        return boundary_expanded * seq_mask

    def compute_segment_score(self, boun_ref, boun_est, valid_len, resolution):
        seq_mask          = tf.sequence_mask(valid_len, maxlen=shape_list(boun_ref)[1], dtype=tf.int32)
        boun_ref_expanded = tf.cast(self.expand_boundary(boun_ref, valid_len, value=1), tf.int32)
        matched           = boun_est * boun_ref_expanded * seq_mask
        n_boun_ref = tf.reduce_sum(boun_ref, axis=1)
        n_boun_est = tf.reduce_sum(boun_est, axis=1)
        n_matched  = tf.reduce_sum(matched,  axis=1)
        precision, recall, fscore   = [], [], []
        precision3, recall3, fscore3 = [], [], []
        for i in range(shape_list(boun_ref)[0]):
            l = valid_len[i].numpy()
            b_ref = boun_ref[i, :l].numpy()
            b_est = boun_est[i, :l].numpy()
            b_ref_in_second   = np.where(b_ref == 1)[0] * resolution
            b_est_in_second   = np.where(b_est == 1)[0] * resolution
            b_ref_in_interval = segmentFrame2interval(b_ref, frame_size=resolution)
            b_est_in_interval = segmentFrame2interval(b_est, frame_size=resolution)
            self.temp['b_ref'].append(b_ref)
            self.temp['b_est'].append(b_est)
            self.temp['matched'].append(matched[i, :l].numpy())
            self.temp['n_b_ref'].append(n_boun_ref[i].numpy())
            self.temp['n_b_est'].append(n_boun_est[i].numpy())
            self.temp['n_matched'].append(n_matched[i].numpy())
            self.temp['b_ref_in_second'].append(b_ref_in_second)
            self.temp['b_est_in_second'].append(b_est_in_second)
            P,  R,  F1 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=0.5, beta=1.0)
            P3, R3, F3 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=3,   beta=1.0)
            precision.append(P);  recall.append(R);  fscore.append(F1)
            precision3.append(P3); recall3.append(R3); fscore3.append(F3)
        return {
            'P_seg':  tf.constant(precision),   'R_seg':  tf.constant(recall),   'F1_seg':  tf.constant(fscore),
            'P_seg3': tf.constant(precision3),  'R_seg3': tf.constant(recall3),  'F1_seg3': tf.constant(fscore3),
        }

    def compute_pairwise_score(self, boun_ref, func_ref, boun_est, func_est, valid_len, resolution):
        precision_pair, recall_pair, fscore_pair = [], [], []
        for i in range(shape_list(func_ref)[0]):
            l = valid_len[i].numpy()
            f_ref = func_ref[i, :l].numpy()
            b_ref = boun_ref[i, :l].numpy()
            f_est = func_est[i, :l].numpy()
            b_est = boun_est[i, :l].numpy()
            interval_ref, label_ref = frame2interval(b_ref, f_ref, frame_size=resolution)
            interval_est, label_est = frame2interval(b_est, f_est, frame_size=resolution)
            self.temp['f_ref'].append(f_ref)
            self.temp['f_est'].append(f_est)
            P, R, F1 = mir_eval.segment.pairwise(interval_ref, label_ref,
                                                  interval_est, label_est, frame_size=0.1)
            precision_pair.append(P); recall_pair.append(R); fscore_pair.append(F1)
        return {'P_pair': tf.constant(precision_pair),
                'R_pair': tf.constant(recall_pair),
                'F1_pair': tf.constant(fscore_pair)}

    def compute_classification_score(self, func_ref, func_est, valid_len, key='Accuracy'):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(func_ref)[1], dtype=tf.float32)
        matched  = tf.cast(func_ref == func_est, tf.float32) * seq_mask
        accuracy = tf.reduce_sum(matched, axis=1) / tf.cast(valid_len, tf.float32)
        return {key: accuracy}

    def compute_confusion_matrix(self, func_ref, func_est):
        return tf.math.confusion_matrix(
            labels=tf.reshape(func_ref, [-1]),
            predictions=tf.reshape(func_est, [-1]),
            num_classes=self.n_classes,
            dtype=tf.dtypes.int32,
        )

    def compute_gt_ssm(self, func_ref, valid_len):
        """
        Build ground-truth self-similarity matrix from section labels.
        Two frames are similar (1.0) if they have the same functional label,
        dissimilar (0.0) otherwise.
        
        func_ref: [B, T]  int32 labels
        Returns:  [B, T, T] float32  (only valid region, rest is 0)
        """
        # [B, T, 1] vs [B, 1, T] → [B, T, T] broadcast comparison
        labels_row = tf.expand_dims(func_ref, axis=2)   # [B, T, 1]
        labels_col = tf.expand_dims(func_ref, axis=1)   # [B, 1, T]
        gt_ssm = tf.cast(tf.equal(labels_row, labels_col), tf.float32)  # [B, T, T]

        # Mask out padding regions
        mask_1d = tf.sequence_mask(valid_len, maxlen=tf.shape(func_ref)[1],
                                    dtype=tf.float32)          # [B, T]
        mask_2d = tf.einsum('bi,bj->bij', mask_1d, mask_1d)   # [B, T, T]
        gt_ssm  = gt_ssm * mask_2d

        return gt_ssm

    def compute_pred_ssm(self, features, valid_len):
        """
        Build predicted self-similarity matrix via cosine similarity.
        
        features: [B, T, D]  fused representation (mixed_b from boundary gate)
        Returns:  [B, T, T]  float32, values in [-1, 1]
        """
        # L2-normalize along feature dimension
        features_norm = tf.nn.l2_normalize(features, axis=-1)  # [B, T, D]

        # Cosine similarity: dot product of normalized vectors
        pred_ssm = tf.matmul(features_norm,
                            tf.transpose(features_norm, perm=[0, 2, 1]))  # [B, T, T]

        # Mask padding
        mask_1d = tf.sequence_mask(valid_len, maxlen=tf.shape(features)[1],
                                    dtype=tf.float32)
        mask_2d = tf.einsum('bi,bj->bij', mask_1d, mask_1d)
        pred_ssm = pred_ssm * mask_2d

        return pred_ssm
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stem data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_files_multistem(dataset_name, song_ids, data_path, include_augmented=False):
    """
    Like your SIMI load_dataset_files() but also loads drum/bass/other stems.
    Expected filename suffixes (matching your preprocessing pipeline):
      mix:   *_spec.npy,       *_chroma.npy
      vocal: *_vocalspec.npy,  *_vocalchroma.npy
      drum:  *_drumspec.npy,   *_drumchroma.npy
      bass:  *_bassspec.npy,   *_basschroma.npy
      other: *_otherspec.npy,  *_otherchroma.npy
    """
    data = {k: [] for k in [
        'spec', 'chromagram',
        'vocal_spec',  'vocal_chromagram',
        'drum_spec',   'drum_chromagram',
        'bass_spec',   'bass_chromagram',
        'other_spec',  'other_chromagram',
        'boundary', 'function', 'len', 'section',
    ]}
    successful = 0

    for song_id in song_ids:
        try:
            if include_augmented:
                pattern    = f"{song_id}*_a1_spec.npy"
                spec_files = glob.glob(os.path.join(data_path, pattern))
                spec_files = [f for f in spec_files if '_original_' not in f]
            else:
                spec_files = [os.path.join(data_path, f"{song_id}_original_a1_spec.npy")]

            for spec_file in spec_files:
                if not os.path.exists(spec_file):
                    continue
                base = spec_file.replace('_spec.npy', '')
                try:
                    spec         = np.load(spec_file)
                    chroma       = np.load(base + '_chroma.npy')
                    vocal_spec   = np.load(base + '_vocalspec.npy')
                    vocal_chroma = np.load(base + '_vocalchroma.npy')
                    drum_spec    = np.load(base + '_drumspec.npy')
                    drum_chroma  = np.load(base + '_drumchroma.npy')
                    bass_spec    = np.load(base + '_bassspec.npy')
                    bass_chroma  = np.load(base + '_basschroma.npy')
                    other_spec   = np.load(base + '_othersspec.npy')
                    other_chroma = np.load(base + '_otherschroma.npy')
                    boundary     = np.load(base + '_boundary.npy')
                    function     = np.load(base + '_function.npy')
                    section      = np.load(base + '_section.npy')
                    vlen         = spec.shape[0]

                    shapes_ok = all(
                        arr.shape[0] == vlen for arr in [
                            chroma, vocal_spec, vocal_chroma,
                            drum_spec, drum_chroma, bass_spec, bass_chroma,
                            other_spec, other_chroma, boundary, function
                        ]
                    )
                    if not shapes_ok:
                        print(f"⚠️ Shape mismatch in {os.path.basename(spec_file)}, skipping")
                        continue

                    data['spec'].append(spec);              data['chromagram'].append(chroma)
                    data['vocal_spec'].append(vocal_spec);  data['vocal_chromagram'].append(vocal_chroma)
                    data['drum_spec'].append(drum_spec);    data['drum_chromagram'].append(drum_chroma)
                    data['bass_spec'].append(bass_spec);    data['bass_chromagram'].append(bass_chroma)
                    data['other_spec'].append(other_spec);  data['other_chromagram'].append(other_chroma)
                    data['boundary'].append(boundary);      data['function'].append(function)
                    data['len'].append(vlen)
                    data['section'].append(f"{dataset_name}_{song_id}")
                    successful += 1

                except FileNotFoundError as e:
                    print(f"  ⚠️ Missing stem file: {e}")
                    continue
                except Exception as e:
                    print(f"  ❌ Error loading {os.path.basename(spec_file)}: {e}")
                    continue
        except Exception as e:
            print(f"❌ Error processing {song_id}: {e}")
            continue

    print(f"   📊 Successfully loaded {successful} files from {dataset_name}")
    for key in data:
        data[key] = np.array(data[key], dtype=object)
    return data


def create_multistem_datasets(config_path, data_base_path, held_out='rwc'):
    """
    Cross-dataset validation loader.

    Uses ALL songs from the JSON (training_set + test_set combined per dataset),
    then assigns them to train or test purely based on which dataset is held out.

    held_out: which dataset to use as the test set
        'rwc'     -> train: Beatles + SALAMI,  test: RWC      (Set 1)
        'beatles' -> train: SALAMI  + RWC,     test: Beatles  (Set 2)
        'salami'  -> train: Beatles + RWC,     test: SALAMI   (Set 3)

    Augmented data is used ONLY for training datasets, never for the test set.
    """
    assert held_out in ('rwc', 'beatles', 'salami'), \
        f"held_out must be 'rwc', 'beatles', or 'salami', got '{held_out}'"

    train_datasets = [d for d in ['beatles', 'salami', 'rwc'] if d != held_out]
    print("\n" + "="*60)
    print(f"🎯 CROSS-DATASET VALIDATION")
    print(f"   Train : {train_datasets}")
    print(f"   Test  : [{held_out}]")
    print("="*60 + "\n")

    config = load_dataset_config(config_path)

    dataset_paths = {
        'beatles': {
            'original': os.path.join(data_base_path, 'beatles-original-preprocessed-data'),
            'aug':      os.path.join(data_base_path, 'beatles-aug-preprocessed-data'),
        },
        'salami': {
            'original': os.path.join(data_base_path, 'salami-original-preprocessed-data'),
            'aug':      os.path.join(data_base_path, 'salami-aug-preprocessed-data'),
        },
        'rwc': {
            'original': os.path.join(data_base_path, 'rwc-original-preprocessed-data'),
            'aug':      os.path.join(data_base_path, 'rwc-aug-preprocessed-data'),
        },
    }

    # Collect ALL song IDs per dataset — merging training_set + test_set from JSON
    # so every available song is used (not the old 70/30 random split)
    all_song_ids = {}
    for dataset_name in ['beatles', 'salami', 'rwc']:
        ids = []
        if dataset_name in config.get('training_set', {}):
            ids.extend(config['training_set'][dataset_name]['song_ids'])
        if dataset_name in config.get('test_set', {}):
            ids.extend(config['test_set'][dataset_name]['song_ids'])
        # Remove duplicates while preserving order
        seen = set()
        ids  = [x for x in ids if not (x in seen or seen.add(x))]
        all_song_ids[dataset_name] = ids
        print(f"   {dataset_name:10s}: {len(ids)} total songs available")

    all_keys = [
        'spec', 'chromagram',
        'vocal_spec',  'vocal_chromagram',
        'drum_spec',   'drum_chromagram',
        'bass_spec',   'bass_chromagram',
        'other_spec',  'other_chromagram',
        'boundary', 'function', 'len', 'section',
    ]

    train_data = {k: [] for k in all_keys}
    test_data  = {k: [] for k in all_keys}

    for dataset_name in ['beatles', 'salami', 'rwc']:
        song_ids = all_song_ids[dataset_name]
        is_test  = (dataset_name == held_out)

        # Original (non-augmented) songs — always loaded for both train and test
        orig = load_dataset_files_multistem(
            dataset_name, song_ids,
            dataset_paths[dataset_name]['original'],
            include_augmented=False
        )

        target = test_data if is_test else train_data
        for k in all_keys:
            target[k].extend(list(orig[k]))

        # Augmented songs — ONLY for training datasets, never for test
        if not is_test:
            aug = load_dataset_files_multistem(
                dataset_name, song_ids,
                dataset_paths[dataset_name]['aug'],
                include_augmented=True
            )
            for k in all_keys:
                train_data[k].extend(list(aug[k]))

    for k in all_keys:
        train_data[k] = np.array(train_data[k], dtype=object)
        test_data[k]  = np.array(test_data[k],  dtype=object)

    print(f"\n✅ Train: {len(train_data['spec'])} samples   Test: {len(test_data['spec'])} samples\n")
    return train_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# TF Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def make_tf_dataset(data):
    """Build a tf.data.Dataset from a multi-stem data dict."""

    def generator():
        for i in range(len(data['spec'])):
            vlen = data['len'][i]
            if hasattr(vlen, '__len__'):
                vlen = int(vlen[0])
            else:
                vlen = int(vlen)
            sec = data['section'][i]
            if hasattr(sec, '__len__') and not isinstance(sec, str):
                sec = str(sec[0])
            else:
                sec = str(sec)

            yield (
                data['spec'][i].astype(np.float32),
                data['chromagram'][i].astype(np.float32),
                data['vocal_spec'][i].astype(np.float32),
                data['vocal_chromagram'][i].astype(np.float32),
                data['drum_spec'][i].astype(np.float32),
                data['drum_chromagram'][i].astype(np.float32),
                data['bass_spec'][i].astype(np.float32),
                data['bass_chromagram'][i].astype(np.float32),
                data['other_spec'][i].astype(np.float32),
                data['other_chromagram'][i].astype(np.float32),
                vlen,
                data['boundary'][i].astype(np.int32),
                data['function'][i].astype(np.int32),
                sec,
            )

    output_signature = (
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),   # spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),   # chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),   # vocal_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),   # vocal_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),   # drum_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),   # drum_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),   # bass_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),   # bass_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),   # other_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),   # other_chromagram
        tf.TensorSpec(shape=[],         dtype=tf.int32),     # valid_len
        tf.TensorSpec(shape=[None],     dtype=tf.int32),     # boundary
        tf.TensorSpec(shape=[None],     dtype=tf.int32),     # function
        tf.TensorSpec(shape=[],         dtype=tf.string),    # section
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction saver for MMoE (same CSV format as SIMI + gate weights CSV)
# ─────────────────────────────────────────────────────────────────────────────

class MMoEPredictionSaver:
    """Saves per-song comparison CSVs and a gate_weights.csv at best epoch."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.pred_dir = os.path.join(save_dir, 'test_predictions')
        os.makedirs(self.pred_dir, exist_ok=True)

    def save_predictions(self, model, test_dataset, test_data, epoch):
        print(f"\n🎯 Saving MMoE predictions (epoch {epoch})...")
        model.gate_log.clear()

        sample_idx  = 0
        saved_count = 0
        epoch_dir   = os.path.join(self.pred_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)

        for batch in test_dataset:
            (spec, chroma,
             vs, vc, ds, dc, bs, bc, os_, oc,
             valid_len, boun_ref, func_ref, sec_ref) = batch

            logits_boun, logits_func, gate_b, gate_l = model.call(
                spec, chroma, vs, vc, ds, dc, bs, bc, os_, oc,
                valid_len, training=False
            )
            prob_boun       = tf.nn.sigmoid(logits_boun)
            boun_est        = model.decode_boundary(prob_boun, valid_len)
            func_est_smooth = model.decode_labeling(boun_est, logits_func, valid_len)

            boun_ref_np = boun_ref.numpy()
            func_ref_np = func_ref.numpy()
            boun_est_np = boun_est.numpy()
            func_est_np = func_est_smooth.numpy()
            vlen_np     = valid_len.numpy()
            gate_b_np   = gate_b.numpy()
            gate_l_np   = gate_l.numpy()
            sec_np      = sec_ref.numpy()

            for i in range(boun_ref_np.shape[0]):
                if sample_idx >= len(test_data['spec']):
                    break
                l       = int(vlen_np[i])
                sec_str = sec_np[i].decode('utf-8') if isinstance(sec_np[i], bytes) else str(sec_np[i])
                safe_id = sec_str.replace('/', '_').replace('\\', '_').replace(' ', '_')

                # comparison CSV (identical format to SIMI)
                b_ref_l = np.append(boun_ref_np[i, :l].copy(), [1]);  b_ref_l[0] = 1
                b_est_l = np.append(boun_est_np[i, :l].copy(), [1]);  b_est_l[0] = 1
                f_ref_l = func_ref_np[i, :l]
                f_est_l = func_est_np[i, :l]
                ref_bnd = np.where(b_ref_l == 1)[0]
                est_bnd = np.where(b_est_l == 1)[0]

                def bnd_to_segs(bnd_idx, labels):
                    segs, lab = [], []
                    for j in range(len(bnd_idx) - 1):
                        s, e = int(bnd_idx[j]), int(bnd_idx[j+1])
                        segs.append((s * 0.5, e * 0.5))
                        lab.append(int(labels[min(s, len(labels)-1)]))
                    return segs, lab

                ref_segs, ref_labs = bnd_to_segs(ref_bnd, f_ref_l)
                est_segs, est_labs = bnd_to_segs(est_bnd, f_est_l)

                comp_file = os.path.join(epoch_dir, f'{safe_id}_compare.csv')
                with open(comp_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Seg','GT_Start','GT_End','GT_Dur','GT_Label',
                                     'Pred_Start','Pred_End','Pred_Dur','Pred_Label','Match'])
                    for k in range(max(len(ref_labs), len(est_labs))):
                        row = [k + 1]
                        if k < len(ref_labs):
                            s, e = ref_segs[k]
                            lbl  = class_conversion(ref_labs[k], reduced=False)
                            row.extend([f"{s:.2f}", f"{e:.2f}", f"{e-s:.2f}", f"{lbl}({ref_labs[k]})"])
                        else:
                            row.extend(['', '', '', ''])
                        if k < len(est_labs):
                            s, e = est_segs[k]
                            lbl  = class_conversion(est_labs[k], reduced=False)
                            row.extend([f"{s:.2f}", f"{e:.2f}", f"{e-s:.2f}", f"{lbl}({est_labs[k]})"])
                        else:
                            row.extend(['', '', '', ''])
                        if k < len(ref_labs) and k < len(est_labs):
                            row.append('✓' if ref_labs[k] == est_labs[k] else '✗')
                        else:
                            row.append('')
                        writer.writerow(row)

                # gate weights for this song
                avg_b = gate_b_np[i, :l, :].mean(axis=0)
                avg_l = gate_l_np[i, :l, :].mean(axis=0)
                model.gate_log.append({
                    'song': sec_str,
                    'boun_gate_vocals': avg_b[0], 'boun_gate_drums': avg_b[1],
                    'boun_gate_bass':   avg_b[2], 'boun_gate_other': avg_b[3],
                    'label_gate_vocals': avg_l[0], 'label_gate_drums': avg_l[1],
                    'label_gate_bass':   avg_l[2], 'label_gate_other': avg_l[3],
                })
                saved_count += 1
                sample_idx  += 1

        # Gate weights summary CSV
        if model.gate_log:
            gate_df = pd.DataFrame(model.gate_log)
            gate_df.to_csv(os.path.join(epoch_dir, 'gate_weights.csv'), index=False)
            print(f"  📊 Gate weights -> {epoch_dir}/gate_weights.csv")
            print("\n  🔬 Mean gate weights across test set:")
            print(f"  Boundary gate: vocals={gate_df['boun_gate_vocals'].mean():.3f}  "
                  f"drums={gate_df['boun_gate_drums'].mean():.3f}  "
                  f"bass={gate_df['boun_gate_bass'].mean():.3f}  "
                  f"other={gate_df['boun_gate_other'].mean():.3f}")
            print(f"  Label gate:    vocals={gate_df['label_gate_vocals'].mean():.3f}  "
                  f"drums={gate_df['label_gate_drums'].mean():.3f}  "
                  f"bass={gate_df['label_gate_bass'].mean():.3f}  "
                  f"other={gate_df['label_gate_other'].mean():.3f}\n")

        print(f"✅ Saved {saved_count} comparison CSVs")
        return saved_count


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_mmoe():
    """Train the MMoE model on all 4 stems."""

    # 1. Load data
    train_data, test_data = create_multistem_datasets(CONFIG_PATH, DATA_BASE_PATH, held_out=HELD_OUT)

    # 2. Build TF datasets
    TRAIN_BATCH_SIZE = 4   # Reduced from 6: last SpecTNT layer inside tape needs more memory
    TEST_BATCH_SIZE  = 4
    TRAIN_SHUFFLE    = len(train_data['spec'])
    N_EPOCHS         = 300

    tf_train_data = make_tf_dataset(train_data)
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE)

    def create_test_dataset():
        return make_tf_dataset(test_data).padded_batch(TEST_BATCH_SIZE)

    # 3. Load frozen experts
    expert_vocals = load_frozen_expert(CKPT_VOCALS, 'Vocals+Mix')
    expert_drums  = load_frozen_expert(CKPT_DRUMS,  'Drums+Mix')
    expert_bass   = load_frozen_expert(CKPT_BASS,   'Bass+Mix')
    expert_other  = load_frozen_expert(CKPT_OTHER,  'Other+Mix')

    # 4. Create MMoE model
    steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE / TRAIN_BATCH_SIZE))

    model = MMoEMusicModel(
        expert_vocals=expert_vocals,
        expert_drums=expert_drums,
        expert_bass=expert_bass,
        expert_other=expert_other,
        n_units=80,
        n_classes=7,
        gate_hidden=128,    # increased from 64 — gate input is 81-dim (80 + positional)
        gate_dropout=0.1,
        tower_dropout=0.3,
        steps_per_epoch=steps_per_epoch,
    )

    # Build by running one dummy forward pass
    print("🔄 Building MMoE model ...")
    dummy    = tf.zeros((1, 100, 80))
    dummy12  = tf.zeros((1, 100, 12))
    dummy_len = tf.constant([100])
    _ = model(dummy, dummy12, dummy, dummy12, dummy, dummy12, dummy, dummy12, dummy, dummy12,
              dummy_len, training=False)
    print("✅ MMoE model built!")
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    gate_tower_params = sum([tf.size(v).numpy() for v in model.trainable_variables
                             if not any(v is ev
                                        for exp in [expert_vocals, expert_drums, expert_bass, expert_other]
                                        for ev in exp.specTNT_layers[-1].trainable_variables)])
    expert_finetune_params = trainable_params - gate_tower_params
    print(f"  Trainable parameters total      : {trainable_params:,}")
    print(f"  ├─ Gates + towers               : {gate_tower_params:,}")
    print(f"  └─ Expert last-SpecTNT (×4)     : {expert_finetune_params:,}  (LR = 10× lower)")

    # 5. Checkpoint and saver setup
    model_path = f'./mmoe_cross_{HELD_OUT}_heldout'
    os.makedirs(f'{model_path}/all_epochs',  exist_ok=True)
    os.makedirs(f'{model_path}/best_models', exist_ok=True)
    checkpoint     = tf.train.Checkpoint(model=model)
    pred_saver     = MMoEPredictionSaver(save_dir=model_path)
    # patience=80: unfrozen expert layers learn more slowly — need more epochs
    # min_delta=0.0005: accept smaller improvements to avoid stopping too early
    early_stopping = ImprovedEarlyStopping(patience=80, min_delta=0.0005, restore_best=True)

    all_epoch_results  = []
    best_test_F1       = -1.0
    best_test_epoch    = 0
    supervised_metrics = ['F1_seg']

    print("🚀 Starting MMoE training loop ...\n")

    # 6. Training loop
    for epoch in range(1, N_EPOCHS + 1):
        print(f"🔄 Epoch {epoch}/{N_EPOCHS}")
        print(color_text("RED") + "--training phase--" + color_text("END"))

        # ── timing + progress tracking ──────────────────────────────────
        import time
        epoch_start  = time.time()
        step         = 0
        total_steps  = steps_per_epoch

        for batch in tf_train_data:
            model.train_step(batch)

            # ── print progress every 50 steps ───────────────────────────
            step += 1
            if step % 50 == 0 or step == 1:
                elapsed      = time.time() - epoch_start
                sec_per_step = elapsed / step
                eta_epoch    = sec_per_step * (total_steps - step)
                
                # GPU memory
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                gpu_used_gb  = gpu_info['current'] / 1024**3
                gpu_peak_gb  = gpu_info['peak']    / 1024**3

                print(f"  step {step:4d}/{total_steps} | "
                    f"{sec_per_step:.2f}s/step | "
                    f"ETA epoch: {eta_epoch/60:.1f}min | "
                    f"GPU: {gpu_used_gb:.1f}/{gpu_peak_gb:.1f}GB (cur/peak)")

        print_temp(model.temp)
        result = model.average_result()
        for k in ['F1_seg', 'P_seg', 'R_seg']:
            if k not in result:
                result[k] = tf.constant(0.0)

        print_confusion_matrix(model.confusion_matrix_train_max.numpy(),
                                model.confusion_matrix_train_boun.numpy())
        model.clear_result()

        try:
            train_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError:
            train_F1 = 0.0

        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['loss','loss_b','loss_f'] if k in result]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['P_seg','R_seg','F1_seg'] if k in result]))

        # Testing
        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        tf_test_data = create_test_dataset()
        for batch in tf_test_data:
            model.test_step(batch)

        safe_sample = np.random.randint(min(TEST_BATCH_SIZE, len(test_data['spec'])))
        print_temp(model.temp, sample=safe_sample)
        result = model.average_result()
        for k in ['F1_seg', 'P_seg', 'R_seg']:
            if k not in result:
                result[k] = tf.constant(0.0)

        print_confusion_matrix(model.confusion_matrix_test_max.numpy(),
                                model.confusion_matrix_test_boun.numpy())
        model.clear_result()

        try:
            test_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError:
            test_F1 = 0.0

        # Save every-epoch checkpoint
        try:
            checkpoint.save(f'{model_path}/all_epochs/epoch-{epoch}')
        except Exception as e:
            print(f'⚠️  Epoch checkpoint failed: {e}')

        # CSV logging (identical to SIMI)
        epoch_result = {'epoch': epoch, 'phase': 'test'}
        for k, v in result.items():
            try:
                epoch_result[k] = float(v.numpy())
            except:
                epoch_result[k] = float(v)
        all_epoch_results.append(epoch_result)
        try:
            pd.DataFrame(all_epoch_results).to_csv(f'{model_path}/training_results.csv', index=False)
        except Exception as e:
            print(f'⚠️  CSV save failed: {e}')

        # Print test results
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['loss','loss_b','loss_f']  if k in result]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['P_seg','R_seg','F1_seg']  if k in result]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['P_seg3','R_seg3','F1_seg3'] if k in result]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['P_pair','R_pair','F1_pair'] if k in result]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(result[k].numpy()))) for k in ['Acc_max','Acc_smooth']    if k in result]))

        print(color_text("CYAN") +
              f'### Best test F1_seg at epoch {best_test_epoch}: {best_test_F1:.3f}' +
              color_text("END"))

        # Best model saving
        if test_F1 > best_test_F1:
            best_test_F1    = test_F1
            best_test_epoch = epoch
            print(color_text("YELLOW") +
                  f"🏆 NEW BEST MODEL at epoch {epoch} -- F1_seg={test_F1:.3f}" +
                  color_text("END"))
            try:
                checkpoint.save(f'{model_path}/best_models/best-epoch-{epoch}')
                with open(f'{model_path}/best_models/best_info.json', 'w') as f:
                    json.dump({
                        'epoch': epoch,
                        'F1_seg': test_F1,
                        'all_metrics': {k: float(v.numpy()) for k, v in result.items()},
                    }, f, indent=2)
                tf_test_pred = create_test_dataset()
                pred_saver.save_predictions(model, tf_test_pred, test_data, epoch)
            except Exception as e:
                print(f'❌ Best model save failed: {e}')
                import traceback; traceback.print_exc()

        # Early stopping
        if early_stopping(test_F1, model, epoch):
            print(color_text("YELLOW") +
                  f"🛑 Early stopping at epoch {epoch}. Best F1: {early_stopping.best_score:.3f}" +
                  color_text("END"))
            break

        print()

    # 7. Training complete
    print("\n" + "="*80)
    print("🎉 MMoE TRAINING COMPLETE!")
    print("="*80)
    if early_stopping.stopped_epoch > 0:
        print(f"🛑 Stopped early at epoch {early_stopping.stopped_epoch}")
    print(f"🏆 Best model: epoch {best_test_epoch}   F1_seg={best_test_F1:.3f}")
    print(f"📂 Checkpoints : {model_path}/")
    print(f"📊 Results CSV : {model_path}/training_results.csv")
    print(f"📝 Best info   : {model_path}/best_models/best_info.json")
    print(f"📂 Predictions : {model_path}/test_predictions/")
    print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_mmoe()