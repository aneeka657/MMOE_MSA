"""
MMoE Music Structure Analysis — 5 Experts (SIMI + SpecTNT)
==============================================================
Professor's experiment: 4 SIMI stem experts + 1 SpecTNT-Attention expert.

Experts:
  Expert 0: Vocals+Mix      (SIMI pretrained)
  Expert 1: Drums+Mix       (SIMI pretrained)
  Expert 2: Bass+Mix        (SIMI pretrained)
  Expert 3: Others+Mix      (SIMI pretrained)
  Expert 4: SpecTNT-Attention  (Wang et al. 2022 pretrained — mixture only)

Key changes vs mmoe_5experts.py (SIMI+AllInOne):
  - Expert 4 is SpecTNT-Attention instead of All-In-One
  - SpecTNT takes only mixture (spec, chromagram) — no stem stack needed
  - No instruments_mel input required — data pipeline same as attention-towers-mmoe.py
  - Gradient splitting: all 5 experts → specTNT_layers[-1]

PATHS TO UPDATE (search for  <- UPDATE):
  DATA_BASE_PATH, CONFIG_PATH, CKPT_VOCALS, CKPT_DRUMS, CKPT_BASS,
  CKPT_OTHER, CKPT_SPECTNT, and the module import path for spectnt.
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
# GPU setup
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

global_frame_size = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Import SIMI model class + utilities
# ─────────────────────────────────────────────────────────────────────────────
from model import (          # <- UPDATE filename if different
    FunctionalSegmentModel,
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
# Import SpecTNT-Attention model class
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util, sys

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_spectnt_mod  = _import_module(
    'spectnt_mod',
    '/Scratch/repository/msa/MSATSUNGPING/spectnt-training.py'  # <- UPDATE
)
SpecTNTModel = _spectnt_mod.FunctionalSegmentModel

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint paths   <- UPDATE ALL FIVE
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH = "/Scratch/repository/msa/MSATSUNGPING/"
CONFIG_PATH    = "/Scratch/repository/msa/MSATSUNGPING/my_dataset_selection_beatles_salami_rwc.json"

CKPT_VOCALS = '/Scratch/repository/msa/MSATSUNGPING/vocals_F_rwc/best_models/best-epoch-58-82'
CKPT_OTHER = '/Scratch/repository/msa/MSATSUNGPING/others_F_rwc/best_models/best-epoch-44-67'
CKPT_DRUMS = '/Scratch/repository/msa/MSATSUNGPING/drums_F_rwc/best_models/best-epoch-101-127'
CKPT_BASS = '/Scratch/repository/msa/MSATSUNGPING/bass_F_rwc/best_models/best-epoch-69-91'
CKPT_SPECTNT   = '/Scratch/repository/msa/MSATSUNGPING/beatles_salami_spectnt_baseline/best_models/best-76'

# ─────────────────────────────────────────────────────────────────────────────
# SIMI expert: encode methods (unchanged from attention-towers-mmoe.py)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_frozen_method(self, spec, chromagram, stem_spec, stem_chromagram, valid_len):
    spec        = tf.math.log(1 + 100 * tf.nn.relu(spec        + 80))
    spec        = tf.expand_dims(spec,        axis=-1)
    chromagram  = tf.expand_dims(chromagram,  axis=-1)
    stem_spec   = tf.math.log(1 + 100 * tf.nn.relu(stem_spec   + 80))

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

    enc_S   = tf.concat([enc_spec, enc_chroma], axis=2)
    enc_FCT = self.fct_dense(tf.reduce_mean(enc_S, axis=2, keepdims=True))
    enc_FCT = self.fct_dense_norm(enc_FCT)
    enc_S  += self.fpe_S
    enc_FCT += self.fpe_FCT

    for specTNT in self.specTNT_layers[:-1]:
        # Defensive unpacking — handles 3-value and 4-value SpecTNT returns
        result  = specTNT(enc_S, enc_FCT, stem_features, valid_len=valid_len, training=False)
        enc_S   = result[0]
        enc_FCT = result[1]

    return (tf.stop_gradient(enc_S),
            tf.stop_gradient(enc_FCT),
            tf.stop_gradient(stem_features))


def _encode_last_layer_method(self, enc_S, enc_FCT, stem_features, valid_len, training=False):
    last_specTNT = self.specTNT_layers[-1]
    # Defensive unpacking — handles 3-value and 4-value SpecTNT returns
    result  = last_specTNT(enc_S, enc_FCT, stem_features, valid_len=valid_len, training=training)
    enc_S   = result[0]
    enc_FCT = result[1]
    return tf.squeeze(enc_FCT, axis=2)   # [B, T, 80]


def patch_simi_model_with_encode(model):
    model._encode_frozen     = types.MethodType(_encode_frozen_method,     model)
    model._encode_last_layer = types.MethodType(_encode_last_layer_method, model)
    return model


def load_frozen_expert(checkpoint_path, stem_name):
    """Load and partially unfreeze a SIMI expert."""
    print(f"🔄 Loading {stem_name} expert from {checkpoint_path} ...")
    model = FunctionalSegmentModel(
        max_len=935, n_units=80, n_heads=8, n_layers=2,
        cnn_dropout_rate=0.0, attn_dropout_rate=0.0,
        use_boundary_fusion=False,
    )
    dummy_spec   = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_len    = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_spec, dummy_chroma, dummy_len, training=False)

    ckpt   = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path).expect_partial()
    try:
        status.assert_existing_objects_matched()
        print(f"  ✅ {stem_name}: checkpoint restored")
    except Exception as e:
        print(f"  ⚠️  {stem_name}: {e}")

    for layer in model.layers:
        layer.trainable = False
    model.trainable = True

    last_spectnt = model.specTNT_layers[-1]
    last_spectnt.trainable = True
    n_unfrozen = sum(tf.size(v).numpy() for v in last_spectnt.trainable_variables)
    print(f"  🔓 {stem_name}: last SpecTNT layer unfrozen ({n_unfrozen:,} params)")

    model.expert_type = 'simi'
    patch_simi_model_with_encode(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SpecTNT-Attention expert: encode methods (from baselines.py)
# Takes only mixture (spec, chromagram) — no stem inputs
# Key difference from Dual-Attention: enc_FCT initialised with zeros+fpe_FCT (no fct_dense)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_spectnt_frozen(self, spec, chromagram, valid_len):
    spec       = tf.math.log(1 + 100 * tf.nn.relu(spec + 80))
    spec       = tf.expand_dims(spec,       axis=-1)
    chromagram = tf.expand_dims(chromagram, axis=-1)

    spec = self.spec_prenorm(spec, valid_len, training=False)

    enc_spec   = self.specCNNBase(spec,        valid_len, training=False)
    enc_spec   = self.specCNN(enc_spec,         valid_len, training=False)
    enc_chroma = self.chromaCNNBase(chromagram, valid_len, training=False)
    enc_chroma = self.chromaCNN(enc_chroma,      valid_len, training=False)

    enc_spec_res   = self.sepc_res_conv(tf.transpose(enc_spec,   [0,1,3,2]))
    enc_spec_res   = tf.reduce_mean(enc_spec_res,   axis=[2,3])
    enc_chroma_res = self.chroma_res_conv(tf.transpose(enc_chroma, [0,1,3,2]))
    enc_chroma_res = tf.reduce_mean(enc_chroma_res, axis=[2,3])

    enc_spec   = self.spec_transition(enc_spec)
    enc_spec   = self.spec_transition_norm(enc_spec)
    enc_chroma = self.chroma_transition(enc_chroma)
    enc_chroma = self.chroma_transition_norm(enc_chroma)

    enc_S   = tf.concat([enc_spec, enc_chroma], axis=2)
    b, n, _, _ = shape_list(enc_S)
    enc_S  += self.fpe_S
    # SpecTNT: enc_FCT initialised with zeros + fpe_FCT (no fct_dense)
    enc_FCT = tf.zeros([b, n, 1, self.n_units]) + self.fpe_FCT

    # Defensive unpacking — handles both 2-return and 4-return SpecTNT
    for specTNT in self.specTNT_layers[:-1]:
        result  = specTNT(enc_S, enc_FCT, valid_len=valid_len, training=False)
        enc_S   = result[0]
        enc_FCT = result[1]

    return (tf.stop_gradient(enc_S),
            tf.stop_gradient(enc_FCT),
            tf.stop_gradient(enc_spec_res),
            tf.stop_gradient(enc_chroma_res))


def _encode_spectnt_last_layer(self, enc_S, enc_FCT, valid_len, training=False):
    last_specTNT = self.specTNT_layers[-1]
    # Defensive unpacking
    result  = last_specTNT(enc_S, enc_FCT, valid_len=valid_len, training=training)
    enc_S   = result[0]
    enc_FCT = result[1]
    return tf.squeeze(enc_FCT, axis=2)   # [B, T, 80]


def load_spectnt_expert(checkpoint_path):
    """Load and partially unfreeze the SpecTNT-Attention expert."""
    print(f"🔄 Loading SpecTNT-Attention expert from {checkpoint_path} ...")
    model = SpecTNTModel(
        max_len=935, n_units=80, n_heads=8, n_layers=2,
        cnn_dropout_rate=0.0, attn_dropout_rate=0.0,
    )
    dummy_spec   = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_len    = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_len, training=False)

    ckpt   = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path).expect_partial()
    try:
        status.assert_existing_objects_matched()
        print("  ✅ SpecTNT: checkpoint restored")
    except Exception as e:
        print(f"  ⚠️  SpecTNT: {e}")

    for layer in model.layers:
        layer.trainable = False
    model.trainable = True

    last_spectnt = model.specTNT_layers[-1]
    last_spectnt.trainable = True
    n_unfrozen = sum(tf.size(v).numpy() for v in last_spectnt.trainable_variables)
    print(f"  🔓 SpecTNT: last SpecTNT layer unfrozen ({n_unfrozen:,} params)")

    model.expert_type = 'spectnt'
    model._encode_frozen     = types.MethodType(_encode_spectnt_frozen,     model)
    model._encode_last_layer = types.MethodType(_encode_spectnt_last_layer, model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TaskGate  (n_experts=5)
# ─────────────────────────────────────────────────────────────────────────────

class TaskGate(tf.keras.layers.Layer):
    def __init__(self, n_experts=5, d_model=80, hidden_units=64,
                 dropout_rate=0.1, name='task_gate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_experts = n_experts
        self.hidden    = tf.keras.layers.Dense(hidden_units, activation='relu',
                                               name=f'{name}_hidden')
        self.dropout   = tf.keras.layers.Dropout(dropout_rate,
                                                  name=f'{name}_dropout')
        self.out       = tf.keras.layers.Dense(n_experts, name=f'{name}_out')

    def call(self, x, training=False):
        h      = self.hidden(x)
        h      = self.dropout(h, training=training)
        logits = self.out(h)
        return tf.nn.softmax(logits, axis=-1)   # [B, T, n_experts]


class LightSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=80, num_heads=4, dropout_rate=0.1,
                 name='light_attn', **kwargs):
        super().__init__(name=name, **kwargs)
        self.attn    = tf.keras.layers.MultiHeadAttention(
                           num_heads=num_heads,
                           key_dim=d_model // num_heads,
                           dropout=dropout_rate,
                           name=f'{name}_mha')
        self.norm    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len=None, training=False):
        if valid_len is not None:
            maxlen   = tf.shape(x)[1]
            pad_mask = tf.sequence_mask(valid_len, maxlen=maxlen)
            pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
        else:
            pad_mask = None
        attn_out = self.attn(x, x, attention_mask=pad_mask, training=training)
        attn_out = self.dropout(attn_out, training=training)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# MMoEMusicModel — 5 experts
# ─────────────────────────────────────────────────────────────────────────────

class MMoEMusicModel(tf.keras.Model):
    """
    MMoE with 5 experts:
      experts[0] = Vocals+Mix   (SIMI)
      experts[1] = Drums+Mix    (SIMI)
      experts[2] = Bass+Mix     (SIMI)
      experts[3] = Others+Mix   (SIMI)
      experts[4] = All-In-One   (Kim & Nam 2023)

    Forward pass:
      1. SIMI experts:   (spec, chroma, stem_spec, stem_chroma) → enc_FCT [B,T,80]
         AllInOne expert: instruments_mel [B,4,T,80]            → proj_out [B,T,80]
      2. Stack → [B, T, 5, 80]
      3. Gate context: mean of 5 + position → [B, T, 81]
      4. boundary_gate, label_gate → [B, T, 5] each
      5. Weighted fusion → fused_b, fused_l [B, T, 80]
      6. Attention towers → logits_boun [B,T], logits_func [B,T,7]
    """

    def __init__(
        self,
        expert_vocals,
        expert_drums,
        expert_bass,
        expert_other,
        expert_spectnt,
        n_units=80,
        n_classes=7,
        gate_hidden=64,
        gate_dropout=0.1,
        tower_dropout=0.3,
        steps_per_epoch=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 5 experts
        self.experts      = [expert_vocals, expert_drums, expert_bass,
                             expert_other, expert_spectnt]
        self.expert_names = ['Vocals+Mix', 'Drums+Mix', 'Bass+Mix',
                             'Others+Mix', 'SpecTNT']
        self.n_experts    = 5

        # Gates — n_experts=5
        self.boundary_gate = TaskGate(n_experts=5, d_model=n_units,
                                      hidden_units=gate_hidden,
                                      dropout_rate=gate_dropout,
                                      name='boundary_gate')
        self.label_gate    = TaskGate(n_experts=5, d_model=n_units,
                                      hidden_units=gate_hidden,
                                      dropout_rate=gate_dropout,
                                      name='label_gate')

        # Towers (unchanged)
        self.boun_attn = LightSelfAttention(d_model=n_units, num_heads=4,
                                            dropout_rate=tower_dropout, name='boun_attn')
        self.boun_proj = tf.keras.layers.Dense(n_units, activation='relu', name='boun_proj')
        self.boun_out  = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same', name='boun_out')
        self.boun_drop = tf.keras.layers.Dropout(tower_dropout)

        self.func_attn = LightSelfAttention(d_model=n_units, num_heads=4,
                                            dropout_rate=tower_dropout, name='func_attn')
        self.func_proj = tf.keras.layers.Dense(n_units, activation='relu', name='func_proj')
        self.func_out  = tf.keras.layers.Conv1D(n_classes, kernel_size=11, padding='same', name='func_out')
        self.func_drop = tf.keras.layers.Dropout(tower_dropout)

        # Bookkeeping
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
        print("MMoE 5-EXPERT CONFIGURATION (SIMI x4 + SpecTNT)")
        print("="*80)
        print(f"  Experts     : {self.expert_names}")
        print(f"  n_experts   : {self.n_experts}")
        print(f"  Gate hidden : {gate_hidden}   dropout: {gate_dropout}")
        print(f"  Tower drop  : {tower_dropout}")
        print("="*80 + "\n")

    # ── forward pass ──────────────────────────────────────────────────────────

    def call(self, spec, chromagram,
             vocal_spec,  vocal_chromagram,
             drum_spec,   drum_chromagram,
             bass_spec,   bass_chromagram,
             other_spec,  other_chromagram,
             valid_len, training=False):
        """
        spec, chromagram         : mixture [B,T,80] / [B,T,12]
        vocal/drum/bass/other_spec + chromagram : stem inputs for SIMI experts
        SpecTNT expert also uses spec + chromagram (mixture only)
        """
        stem_inputs = [
            (vocal_spec,  vocal_chromagram),
            (drum_spec,   drum_chromagram),
            (bass_spec,   bass_chromagram),
            (other_spec,  other_chromagram),
        ]

        # Encode 4 SIMI experts
        expert_feats = []
        for expert, (s, c) in zip(self.experts[:4], stem_inputs):
            enc_S, enc_FCT, stem_feats = expert._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            feat = expert._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                             valid_len, training=training)
            expert_feats.append(feat)   # [B, T, 80]

        # Encode SpecTNT-Attention expert (expert 4) — mixture only
        stn_exp             = self.experts[4]
        enc_S, enc_FCT, _, _ = stn_exp._encode_frozen(spec, chromagram, valid_len)
        stn_feat            = stn_exp._encode_last_layer(enc_S, enc_FCT,
                                                         valid_len, training=training)
        expert_feats.append(stn_feat)   # [B, T, 80]

        # Stack → [B, T, 5, 80]
        stacked = tf.stack(expert_feats, axis=2)

        # Gate context: mean of 5 experts + normalised position
        gate_context_feats = tf.reduce_mean(stacked, axis=2)            # [B, T, 80]
        T         = tf.shape(stacked)[1]
        positions = tf.cast(tf.range(T), tf.float32) / tf.cast(T, tf.float32)
        positions = tf.tile(positions[tf.newaxis, :, tf.newaxis],
                            [tf.shape(stacked)[0], 1, 1])                # [B, T, 1]
        gate_context = tf.concat([gate_context_feats, positions], axis=-1)  # [B, T, 81]

        gate_b = self.boundary_gate(gate_context, training=training)     # [B, T, 5]
        gate_l = self.label_gate(gate_context,    training=training)     # [B, T, 5]

        fused_b = tf.reduce_sum(
            tf.expand_dims(gate_b, axis=-1) * stacked, axis=2)          # [B, T, 80]
        fused_l = tf.reduce_sum(
            tf.expand_dims(gate_l, axis=-1) * stacked, axis=2)          # [B, T, 80]

        # Boundary tower
        x = self.boun_attn(fused_b, valid_len=valid_len, training=training)
        x = self.boun_drop(x, training=training)
        x = self.boun_proj(x)
        logits_boun = tf.squeeze(self.boun_out(x), axis=2)              # [B, T]

        # Label tower
        y = self.func_attn(fused_l, valid_len=valid_len, training=training)
        y = self.func_drop(y, training=training)
        y = self.func_proj(y)
        logits_func = self.func_out(y)                                   # [B, T, 7]

        return logits_boun, logits_func, gate_b, gate_l

    # ── call_from_encoded (used inside train_step) ────────────────────────────

    def call_from_encoded(self, stacked, valid_len, training=False):
        """stacked: [B, T, 5, 80]"""
        gate_context_feats = tf.reduce_mean(stacked, axis=2)            # [B, T, 80]
        T         = tf.shape(stacked)[1]
        positions = tf.cast(tf.range(T), tf.float32) / tf.cast(T, tf.float32)
        positions = tf.tile(positions[tf.newaxis, :, tf.newaxis],
                            [tf.shape(stacked)[0], 1, 1])
        gate_context = tf.concat([gate_context_feats, positions], axis=-1)  # [B, T, 81]

        gate_b = self.boundary_gate(gate_context, training=training)    # [B, T, 5]
        gate_l = self.label_gate(gate_context,    training=training)    # [B, T, 5]

        mixed_b = tf.reduce_sum(stacked * gate_b[:, :, :, tf.newaxis], axis=2)
        mixed_l = tf.reduce_sum(stacked * gate_l[:, :, :, tf.newaxis], axis=2)

        x_b = self.boun_attn(mixed_b, valid_len=valid_len, training=training)
        x_b = self.boun_drop(x_b, training=training)
        x_b = self.boun_proj(x_b)
        logits_boun = tf.squeeze(self.boun_out(x_b), axis=-1)           # [B, T]

        x_f = self.func_attn(mixed_l, valid_len=valid_len, training=training)
        x_f = self.func_drop(x_f, training=training)
        x_f = self.func_proj(x_f)
        logits_func = self.func_out(x_f)                                 # [B, T, 7]

        return logits_boun, logits_func, gate_b, gate_l, mixed_b

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self, data):
        (spec, chromagram,
         vocal_spec,  vocal_chromagram,
         drum_spec,   drum_chromagram,
         bass_spec,   bass_chromagram,
         other_spec,  other_chromagram,
         valid_len, boun_ref, func_ref, sec_ref) = data

        stem_inputs = [
            (vocal_spec,  vocal_chromagram),
            (drum_spec,   drum_chromagram),
            (bass_spec,   bass_chromagram),
            (other_spec,  other_chromagram),
        ]

        # Stage 1: frozen layers OUTSIDE tape
        intermediates = []
        for expert, (s, c) in zip(self.experts[:4], stem_inputs):
            enc_S, enc_FCT, stem_feats = expert._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            intermediates.append((enc_S, enc_FCT, stem_feats))

        # SpecTNT frozen stage (mixture only)
        stn_exp = self.experts[4]
        stn_enc_S, stn_enc_FCT, _, _ = stn_exp._encode_frozen(spec, chromagram, valid_len)

        with tf.GradientTape() as tape:
            # Stage 2: last layers INSIDE tape
            enc_list = []
            for expert, (enc_S, enc_FCT, stem_feats) in zip(self.experts[:4], intermediates):
                feat = expert._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                                  valid_len, training=True)
                enc_list.append(feat)

            # SpecTNT last layer inside tape
            stn_feat = stn_exp._encode_last_layer(stn_enc_S, stn_enc_FCT,
                                                   valid_len, training=True)
            enc_list.append(stn_feat)

            stacked = tf.stack(enc_list, axis=2)   # [B, T, 5, 80]

            logits_boun, logits_func, gate_b, gate_l, mixed_b = \
                self.call_from_encoded(stacked, valid_len, training=True)

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

            # Gate entropy regularisation
            gate_entropy_b = -tf.reduce_mean(
                tf.reduce_sum(gate_b * tf.math.log(gate_b + 1e-8), axis=-1))
            gate_entropy_l = -tf.reduce_mean(
                tf.reduce_sum(gate_l * tf.math.log(gate_l + 1e-8), axis=-1))
            gate_entropy   = gate_entropy_b + gate_entropy_l

            gate_cosine_sim = tf.reduce_mean(
                tf.reduce_sum(gate_b * gate_l, axis=-1) /
                (tf.norm(gate_b, axis=-1) * tf.norm(gate_l, axis=-1) + 1e-8)
            )

            # SSM loss
            gt_ssm      = self.compute_gt_ssm(func_ref, valid_len)
            pred_ssm    = self.compute_pred_ssm(mixed_b, valid_len)
            pred_ssm_01 = (pred_ssm + 1.0) / 2.0
            ssm_loss    = tf.reduce_mean(tf.square(pred_ssm_01 - gt_ssm))

            loss = ce_b + ce_f + 0.005 * gate_entropy + 0.1 * ssm_loss

            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)

        # Optimizer setup
        if not hasattr(self, 'optimizer') or self.optimizer is None:
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
            return tf.cond(current_epoch < 30,
                lambda: 1e-4 + (4e-4) * (current_epoch / 30.0),
                lambda: tf.cond(current_epoch < 150,
                    lambda: 5e-4,
                    lambda: tf.cond(current_epoch < 220,
                        lambda: 5e-4 - (4.5e-4) * ((current_epoch - 150.0) / 70.0),
                        lambda: 5e-5)))

        def lr_schedule_expert():
            return tf.cond(current_epoch < 30,
                lambda: 1e-5 + (4e-5) * (current_epoch / 30.0),
                lambda: tf.cond(current_epoch < 150,
                    lambda: 5e-5,
                    lambda: tf.cond(current_epoch < 220,
                        lambda: 5e-5 - (4.5e-5) * ((current_epoch - 150.0) / 70.0),
                        lambda: 5e-6)))

        self.optimizer.learning_rate.assign(lr_schedule())
        self.optimizer_expert.learning_rate.assign(lr_schedule_expert())

        # Split gradients:
        #   All 5 experts (SIMI x4 + SpecTNT) → specTNT_layers[-1]
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

    # ── test_step ─────────────────────────────────────────────────────────────

    def test_step(self, data, log_gates=False):
        (spec, chromagram,
         vocal_spec,  vocal_chromagram,
         drum_spec,   drum_chromagram,
         bass_spec,   bass_chromagram,
         other_spec,  other_chromagram,
         valid_len, boun_ref, func_ref, sec_ref) = data

        stem_inputs = [
            (vocal_spec,  vocal_chromagram),
            (drum_spec,   drum_chromagram),
            (bass_spec,   bass_chromagram),
            (other_spec,  other_chromagram),
        ]

        # Encode all 5 experts (no tape in test)
        expert_feats = []
        for expert, (s, c) in zip(self.experts[:4], stem_inputs):
            enc_S, enc_FCT, stem_feats = expert._encode_frozen(
                spec, chromagram, s, c, valid_len
            )
            feat = expert._encode_last_layer(enc_S, enc_FCT, stem_feats,
                                             valid_len, training=False)
            expert_feats.append(feat)

        stn_exp = self.experts[4]
        stn_enc_S, stn_enc_FCT, _, _ = stn_exp._encode_frozen(spec, chromagram, valid_len)
        stn_feat = stn_exp._encode_last_layer(stn_enc_S, stn_enc_FCT,
                                              valid_len, training=False)
        expert_feats.append(stn_feat)

        stacked = tf.stop_gradient(tf.stack(expert_feats, axis=2))   # [B,T,5,80]

        logits_boun, logits_func, gate_b, gate_l, mixed_b = \
            self.call_from_encoded(stacked, valid_len, training=False)

        prob_boun       = tf.nn.sigmoid(logits_boun)
        boun_est        = self.decode_boundary(prob_boun, valid_len)
        func_est_max    = tf.argmax(logits_func, axis=-1, output_type=tf.int32)
        func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len)

        self.confusion_matrix_test_max  += self.compute_confusion_matrix(func_ref, func_est_max)
        self.confusion_matrix_test_boun += self.compute_confusion_matrix(func_ref, func_est_smooth)

        ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
        ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)

        gate_entropy_b = -tf.reduce_mean(tf.reduce_sum(gate_b * tf.math.log(gate_b + 1e-8), axis=-1))
        gate_entropy_l = -tf.reduce_mean(tf.reduce_sum(gate_l * tf.math.log(gate_l + 1e-8), axis=-1))
        gate_entropy   = gate_entropy_b + gate_entropy_l

        gate_cosine_sim = tf.reduce_mean(
            tf.reduce_sum(gate_b * gate_l, axis=-1) /
            (tf.norm(gate_b, axis=-1) * tf.norm(gate_l, axis=-1) + 1e-8)
        )

        gt_ssm      = self.compute_gt_ssm(func_ref, valid_len)
        pred_ssm    = self.compute_pred_ssm(mixed_b, valid_len)
        pred_ssm_01 = (pred_ssm + 1.0) / 2.0
        ssm_loss    = tf.reduce_mean(tf.square(pred_ssm_01 - gt_ssm))
        loss        = ce_b + ce_f

        score_dict = self.compute_classification_score(func_ref, func_est_max,     valid_len, key='Acc_max')
        score_dict.update(self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth'))
        score_dict.update(self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size))
        score_dict.update(self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size))
        score_dict.update({'loss': loss, 'loss_b': ce_b, 'loss_f': ce_f,
                           'gate_entropy': gate_entropy, 'ssm_loss': ssm_loss,
                           'gate_sim': gate_cosine_sim})
        [self.result[k].append(v) for k, v in score_dict.items()]

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
                    'song':                   song_id,
                    'boun_gate_vocals':        avg_b[0],
                    'boun_gate_drums':         avg_b[1],
                    'boun_gate_bass':          avg_b[2],
                    'boun_gate_other':         avg_b[3],
                    'boun_gate_spectnt':     avg_b[4],
                    'label_gate_vocals':       avg_l[0],
                    'label_gate_drums':        avg_l[1],
                    'label_gate_bass':         avg_l[2],
                    'label_gate_other':        avg_l[3],
                    'label_gate_spectnt':    avg_l[4],
                })

        return boun_est, func_est_smooth

    # ── helper methods (identical to attention-towers-mmoe.py) ─────────────────

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
                    result_dict[k] = tf.reduce_mean(tf.concat(v, axis=0))
                except tf.errors.InvalidArgumentError:
                    result_dict[k] = tf.reduce_mean(tf.stack(v))
        return result_dict

    def bce_from_logits(self, gt, logits, valid_len, pos_weight=0.3):
        gt_expanded = self.expand_boundary(gt, valid_len, value=0.5)
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(gt)[1], dtype=tf.float32)
        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_expanded, logits, pos_weight=pos_weight)
        loss = tf.reduce_sum(wbce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    def cce_from_logits(self, gt, logits, valid_len):
        weights = tf.constant([1.03, 0.27, 0.47, 0.94, 0.88, 3.00, 1.68], tf.float32)
        seq_mask    = tf.sequence_mask(valid_len, maxlen=tf.shape(gt)[1], dtype=tf.float32)
        gt_onehot   = tf.one_hot(gt, depth=self.n_classes)
        ce          = tf.nn.softmax_cross_entropy_with_logits(gt_onehot, logits)
        weighted_ce = ce * tf.gather(weights, gt)
        loss = tf.reduce_sum(weighted_ce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    def decode_boundary(self, prob_boun, valid_len, method='librosa'):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(prob_boun)[1], dtype=tf.float32)
        prob_boun *= seq_mask
        prob_boun_numpy = prob_boun.numpy()
        peaks = np.zeros_like(prob_boun_numpy, dtype=np.int32)
        if method == 'librosa':
            peak_indices = [
                librosa.util.peak_pick(seq, pre_max=10, post_max=10, pre_avg=20,
                                       post_avg=10, delta=0.03, wait=10).astype(int)
                for seq in prob_boun_numpy
            ]
        elif method == 'msaf':
            peak_indices = [peak_picking_MSAF(seq, median_len=7, offset_rel=0.05, sigma=4)
                            for seq in prob_boun_numpy]
        else:
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
            l   = valid_len[i]
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
            axis=-1)
        cond = tf.logical_and((boundary_expanded != boundary),
                              tf.logical_not(tf.cast(boundary, tf.bool)))
        boundary_expanded = tf.where(cond, value, boundary)
        return boundary_expanded * seq_mask

    def compute_segment_score(self, boun_ref, boun_est, valid_len, resolution):
        seq_mask          = tf.sequence_mask(valid_len, maxlen=shape_list(boun_ref)[1], dtype=tf.int32)
        boun_ref_expanded = tf.cast(self.expand_boundary(boun_ref, valid_len, value=1), tf.int32)
        matched           = boun_est * boun_ref_expanded * seq_mask
        precision, recall, fscore    = [], [], []
        precision3, recall3, fscore3 = [], [], []
        for i in range(shape_list(boun_ref)[0]):
            l = valid_len[i].numpy()
            b_ref = boun_ref[i, :l].numpy()
            b_est = boun_est[i, :l].numpy()
            b_ref_in_interval = segmentFrame2interval(b_ref, frame_size=resolution)
            b_est_in_interval = segmentFrame2interval(b_est, frame_size=resolution)
            b_ref_in_second   = np.where(b_ref == 1)[0] * resolution
            b_est_in_second   = np.where(b_est == 1)[0] * resolution
            n_boun_ref = tf.reduce_sum(boun_ref, axis=1)
            n_boun_est = tf.reduce_sum(boun_est, axis=1)
            n_matched  = tf.reduce_sum(matched,  axis=1)
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
            'P_seg':  tf.constant(precision),  'R_seg':  tf.constant(recall),  'F1_seg':  tf.constant(fscore),
            'P_seg3': tf.constant(precision3), 'R_seg3': tf.constant(recall3), 'F1_seg3': tf.constant(fscore3),
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
        labels_row = tf.expand_dims(func_ref, axis=2)
        labels_col = tf.expand_dims(func_ref, axis=1)
        gt_ssm = tf.cast(tf.equal(labels_row, labels_col), tf.float32)
        mask_1d = tf.sequence_mask(valid_len, maxlen=tf.shape(func_ref)[1], dtype=tf.float32)
        mask_2d = tf.einsum('bi,bj->bij', mask_1d, mask_1d)
        return gt_ssm * mask_2d

    def compute_pred_ssm(self, features, valid_len):
        features_norm = tf.nn.l2_normalize(features, axis=-1)
        pred_ssm = tf.matmul(features_norm,
                             tf.transpose(features_norm, perm=[0, 2, 1]))
        mask_1d = tf.sequence_mask(valid_len, maxlen=tf.shape(features)[1], dtype=tf.float32)
        mask_2d = tf.einsum('bi,bj->bij', mask_1d, mask_1d)
        return pred_ssm * mask_2d


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline — exact copy from attention-towers-mmoe.py (no instruments_mel needed)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_files_multistem(dataset_name, song_ids, data_path, include_augmented=False):
    """
    Exact copy from attention-towers-mmoe.py — correct file suffixes:
      mix:   *_spec.npy,        *_chroma.npy
      vocal: *_vocalspec.npy,   *_vocalchroma.npy
      drum:  *_drumspec.npy,    *_drumchroma.npy
      bass:  *_bassspec.npy,    *_basschroma.npy
      other: *_othersspec.npy,  *_otherschroma.npy   ← note: 'others' not 'other'
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
                    other_spec   = np.load(base + '_othersspec.npy')   # 'others' not 'other'
                    other_chroma = np.load(base + '_otherschroma.npy') # 'others' not 'other'
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


def create_datasets(config_path, data_base_path):
    """Exact copy of create_multistem_datasets from attention-towers-mmoe.py."""
    print("🎯 Creating 5-expert MMoE datasets...")
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

    for dataset_name, info in config['training_set'].items():
        if dataset_name == 'summary':
            continue
        song_ids = info['song_ids']
        orig = load_dataset_files_multistem(dataset_name, song_ids,
                                            dataset_paths[dataset_name]['original'],
                                            include_augmented=False)
        aug  = load_dataset_files_multistem(dataset_name, song_ids,
                                            dataset_paths[dataset_name]['aug'],
                                            include_augmented=True)
        for k in all_keys:
            train_data[k].extend(list(orig[k]))
            train_data[k].extend(list(aug[k]))

    for dataset_name, info in config['test_set'].items():
        song_ids = info['song_ids']
        orig = load_dataset_files_multistem(dataset_name, song_ids,
                                            dataset_paths[dataset_name]['original'],
                                            include_augmented=False)
        for k in all_keys:
            test_data[k].extend(list(orig[k]))

    for k in all_keys:
        train_data[k] = np.array(train_data[k], dtype=object)
        test_data[k]  = np.array(test_data[k],  dtype=object)

    print(f"✅ Train: {len(train_data['spec'])} samples   Test: {len(test_data['spec'])} samples")
    return train_data, test_data


def make_tf_dataset(data):
    """
    Exact generator from attention-towers-mmoe.py.
    No instruments_mel needed — SpecTNT uses mixture only.
    """
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
                data['spec'][i].astype(np.float32),              # [T, 80]
                data['chromagram'][i].astype(np.float32),        # [T, 12]
                data['vocal_spec'][i].astype(np.float32),        # [T, 80]
                data['vocal_chromagram'][i].astype(np.float32),  # [T, 12]
                data['drum_spec'][i].astype(np.float32),         # [T, 80]
                data['drum_chromagram'][i].astype(np.float32),   # [T, 12]
                data['bass_spec'][i].astype(np.float32),         # [T, 80]
                data['bass_chromagram'][i].astype(np.float32),   # [T, 12]
                data['other_spec'][i].astype(np.float32),        # [T, 80]
                data['other_chromagram'][i].astype(np.float32),  # [T, 12]
                vlen,
                data['boundary'][i].astype(np.int32),
                data['function'][i].astype(np.int32),
                sec,
            )

    output_signature = (
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),    # spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),    # chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),    # vocal_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),    # vocal_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),    # drum_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),    # drum_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),    # bass_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),    # bass_chromagram
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),    # other_spec
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),    # other_chromagram
        tf.TensorSpec(shape=[],         dtype=tf.int32),      # valid_len
        tf.TensorSpec(shape=[None],     dtype=tf.int32),      # boundary
        tf.TensorSpec(shape=[None],     dtype=tf.int32),      # function
        tf.TensorSpec(shape=[],         dtype=tf.string),     # section
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction saver — 5 expert gate columns
# ─────────────────────────────────────────────────────────────────────────────

class MMoEPredictionSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.pred_dir = os.path.join(save_dir, 'test_predictions')
        os.makedirs(self.pred_dir, exist_ok=True)

    def save_predictions(self, model, test_dataset, test_data, epoch):
        print(f"\n🎯 Saving 5-expert MMoE predictions (epoch {epoch})...")
        model.gate_log.clear()
        epoch_dir   = os.path.join(self.pred_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        sample_idx  = 0
        saved_count = 0

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

            gate_b_np = gate_b.numpy()
            gate_l_np = gate_l.numpy()
            vlen_np   = valid_len.numpy()
            sec_np    = sec_ref.numpy()

            for i in range(gate_b_np.shape[0]):
                if sample_idx >= len(test_data['spec']): break
                v_len   = int(vlen_np[i])
                sec_str = sec_np[i].decode('utf-8') if isinstance(sec_np[i], bytes) else str(sec_np[i])
                avg_b   = gate_b_np[i, :v_len, :].mean(axis=0)
                avg_l   = gate_l_np[i, :v_len, :].mean(axis=0)
                model.gate_log.append({
                    'song':                 sec_str,
                    'boun_gate_vocals':     avg_b[0],
                    'boun_gate_drums':      avg_b[1],
                    'boun_gate_bass':       avg_b[2],
                    'boun_gate_other':      avg_b[3],
                    'boun_gate_spectnt':  avg_b[4],
                    'label_gate_vocals':    avg_l[0],
                    'label_gate_drums':     avg_l[1],
                    'label_gate_bass':      avg_l[2],
                    'label_gate_other':     avg_l[3],
                    'label_gate_spectnt': avg_l[4],
                })
                saved_count += 1
                sample_idx  += 1

        if model.gate_log:
            gate_df = pd.DataFrame(model.gate_log)
            gate_df.to_csv(os.path.join(epoch_dir, 'gate_weights.csv'), index=False)
            print(f"  📊 Gate weights → {epoch_dir}/gate_weights.csv")
            print(f"\n  🔬 Mean gate weights across test set:")
            print(f"  Boundary: V={gate_df['boun_gate_vocals'].mean():.3f}  "
                  f"D={gate_df['boun_gate_drums'].mean():.3f}  "
                  f"B={gate_df['boun_gate_bass'].mean():.3f}  "
                  f"O={gate_df['boun_gate_other'].mean():.3f}  "
                  f"Stn={gate_df['boun_gate_spectnt'].mean():.3f}")
            print(f"  Label:    V={gate_df['label_gate_vocals'].mean():.3f}  "
                  f"D={gate_df['label_gate_drums'].mean():.3f}  "
                  f"B={gate_df['label_gate_bass'].mean():.3f}  "
                  f"O={gate_df['label_gate_other'].mean():.3f}  "
                  f"Stn={gate_df['label_gate_spectnt'].mean():.3f}")
        print(f"✅ Saved {saved_count} entries")
        return saved_count


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_mmoe():
    """Train MMoE with 5 experts: 4 SIMI + 1 All-In-One."""

    # 1. Load data
    train_data, test_data = create_datasets(CONFIG_PATH, DATA_BASE_PATH)

    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE  = 4
    TRAIN_SHUFFLE    = len(train_data['spec'])
    N_EPOCHS         = 300

    tf_train_data = make_tf_dataset(train_data)
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE)

    def create_test_dataset():
        return make_tf_dataset(test_data).padded_batch(TEST_BATCH_SIZE)

    # 2. Load 5 experts
    expert_vocals   = load_frozen_expert(CKPT_VOCALS,   'Vocals+Mix')
    expert_drums    = load_frozen_expert(CKPT_DRUMS,    'Drums+Mix')
    expert_bass     = load_frozen_expert(CKPT_BASS,     'Bass+Mix')
    expert_other    = load_frozen_expert(CKPT_OTHER,    'Others+Mix')
    expert_spectnt = load_spectnt_expert(CKPT_SPECTNT)

    # 3. Create MMoE model
    steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE / TRAIN_BATCH_SIZE))

    model = MMoEMusicModel(
        expert_vocals=expert_vocals,
        expert_drums=expert_drums,
        expert_bass=expert_bass,
        expert_other=expert_other,
        expert_spectnt=expert_spectnt,
        n_units=80,
        n_classes=7,
        gate_hidden=64,
        gate_dropout=0.1,
        tower_dropout=0.3,
        steps_per_epoch=steps_per_epoch,
    )

    # 4. Build with dummy data
    print("🔄 Building 5-expert MMoE model (SIMI x4 + SpecTNT) ...")
    dummy_spec   = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_stem   = tf.zeros((1, 100, 80))
    dummy_stemch = tf.zeros((1, 100, 12))
    dummy_len    = tf.constant([100])
    _ = model(
        dummy_spec, dummy_chroma,
        dummy_stem, dummy_stemch,
        dummy_stem, dummy_stemch,
        dummy_stem, dummy_stemch,
        dummy_stem, dummy_stemch,
        dummy_len, training=False
    )
    print("✅ 5-expert MMoE model built!")

    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Trainable parameters total: {trainable_params:,}")

    # 5. Setup checkpointing
    model_path = './mmoe_spectnt_rwc'
    os.makedirs(f'{model_path}/all_epochs',  exist_ok=True)
    os.makedirs(f'{model_path}/best_models', exist_ok=True)
    checkpoint        = tf.train.Checkpoint(model=model)
    pred_saver        = MMoEPredictionSaver(save_dir=model_path)
    early_stopping    = ImprovedEarlyStopping(patience=80, min_delta=0.0005, restore_best=True)

    all_epoch_results  = []
    best_test_F1       = -1.0
    best_test_epoch    = 0
    supervised_metrics = ['F1_seg']

    print("🚀 Starting 5-expert MMoE training (SIMI x4 + SpecTNT) ...\n")

    # 6. Training loop
    for epoch in range(1, N_EPOCHS + 1):
        import time
        print(f"🔄 Epoch {epoch}/{N_EPOCHS}")
        print(color_text("RED") + "--training phase--" + color_text("END"))

        epoch_start = time.time()
        step        = 0
        total_steps = steps_per_epoch

        for batch in tf_train_data:
            model.train_step(batch)
            step += 1
            if step % 50 == 0 or step == 1:
                elapsed = time.time() - epoch_start
                eta     = (elapsed / step) * (total_steps - step)
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"  step {step:4d}/{total_steps} | "
                      f"{elapsed/step:.2f}s/step | ETA: {eta/60:.1f}min | "
                      f"GPU: {gpu_info['current']/1024**3:.1f}/{gpu_info['peak']/1024**3:.1f}GB")

        print_temp(model.temp)
        result = model.average_result()
        for k in ['F1_seg', 'P_seg', 'R_seg']:
            if k not in result: result[k] = tf.constant(0.0)
        print_confusion_matrix(model.confusion_matrix_train_max.numpy(),
                                model.confusion_matrix_train_boun.numpy())
        model.clear_result()

        try:    train_F1 = sum(float(result[k]) for k in supervised_metrics)
        except: train_F1 = 0.0

        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['loss','loss_b','loss_f'] if k in result))
        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['P_seg','R_seg','F1_seg'] if k in result))

        # Testing phase
        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        for batch in create_test_dataset():
            model.test_step(batch)

        print_temp(model.temp,
                   sample=np.random.randint(min(TEST_BATCH_SIZE, len(test_data['spec']))))
        result = model.average_result()
        for k in ['F1_seg', 'P_seg', 'R_seg']:
            if k not in result: result[k] = tf.constant(0.0)
        print_confusion_matrix(model.confusion_matrix_test_max.numpy(),
                                model.confusion_matrix_test_boun.numpy())
        model.clear_result()

        try:    test_F1 = sum(float(result[k]) for k in supervised_metrics)
        except: test_F1 = 0.0

        # Epoch checkpoint
        try:    checkpoint.save(f'{model_path}/all_epochs/epoch-{epoch}')
        except Exception as e: print(f'⚠️  Epoch checkpoint failed: {e}')

        # CSV logging
        epoch_result = {'epoch': epoch, 'phase': 'test'}
        for k, v in result.items():
            try:    epoch_result[k] = float(v.numpy())
            except: epoch_result[k] = float(v)
        all_epoch_results.append(epoch_result)
        try:    pd.DataFrame(all_epoch_results).to_csv(
                    f'{model_path}/training_results.csv', index=False)
        except Exception as e: print(f'⚠️  CSV save failed: {e}')

        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['loss','loss_b','loss_f']  if k in result))
        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['P_seg','R_seg','F1_seg']  if k in result))
        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['P_seg3','R_seg3','F1_seg3'] if k in result))
        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['P_pair','R_pair','F1_pair'] if k in result))
        print('  '.join(f"{k.ljust(8)} {result[k].numpy():.3f}"
                        for k in ['Acc_max','Acc_smooth']    if k in result))
        print(color_text("CYAN") +
              f'### Best test F1_seg at epoch {best_test_epoch}: {best_test_F1:.3f}' +
              color_text("END"))

        if test_F1 > best_test_F1:
            best_test_F1    = test_F1
            best_test_epoch = epoch
            print(color_text("YELLOW") +
                  f"🏆 NEW BEST at epoch {epoch} — F1_seg={test_F1:.3f}" +
                  color_text("END"))
            try:
                checkpoint.save(f'{model_path}/best_models/best-epoch-{epoch}')
                with open(f'{model_path}/best_models/best_info.json', 'w') as f:
                    json.dump({'epoch': epoch, 'F1_seg': test_F1,
                               'all_metrics': {k: float(v.numpy()) for k,v in result.items()}},
                              f, indent=2)
                pred_saver.save_predictions(model, create_test_dataset(), test_data, epoch)
            except Exception as e:
                print(f'❌ Best model save failed: {e}')
                import traceback; traceback.print_exc()

        if early_stopping(test_F1, model, epoch):
            print(color_text("YELLOW") +
                  f"🛑 Early stopping at epoch {epoch}. Best F1: {early_stopping.best_score:.3f}" +
                  color_text("END"))
            break
        print()

    print("\n" + "="*80)
    print("🎉 5-EXPERT MMoE TRAINING COMPLETE (SIMI x4 + SpecTNT)!")
    print(f"🏆 Best model: epoch {best_test_epoch}   F1_seg={best_test_F1:.3f}")
    print(f"📂 Results: {model_path}/")
    print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_mmoe()
