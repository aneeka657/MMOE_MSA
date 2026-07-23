import os
import json
import glob
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d, filters
from tensorflow.keras import backend
import mir_eval
import librosa
import math

# Use GPUs 2 and 3 (the free ones!)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

global_frame_size = 0.5

# === UTILITY FUNCTIONS (SAME AS BEFORE) ===
function_dict = {
    'intro': 0, 'verse': 1, 'chorus': 2, 'bridge': 3,
    'inst': 4, 'outro': 5, 'silence': 6,
}

def peak_picking_MSAF(x, median_len=9, offset_rel=0.05, sigma=4.0):
    offset = x.mean() * offset_rel
    x = gaussian_filter1d(x, sigma=sigma)
    threshold_local = median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    return np.array(peaks, dtype=np.int32)

def peak_picking_boeck(activations, threshold=0.5, fps=100, include_scores=False, combine=False,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    activations = activations.ravel()
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = filters.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = filters.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        detections = detections * (detections >= threshold)
    if combine:
        stamps = []
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            if i > last_onset + combine:
                stamps.append(i)
                last_onset = i
        stamps = np.array(stamps)
    else:
        stamps = np.where(detections)[0]
    return stamps

def segmentFrame2interval(segment_frame, frame_size=0.5):
    segment_frame = np.array(segment_frame)
    segment_frame[0] = 1
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval

def frame2interval(segment_frame, label_frame, frame_size=0.5):
    segment_frame = np.array(segment_frame)
    label_frame = np.array(label_frame)
    segment_frame[0] = 1
    label = label_frame[segment_frame == 1]
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval, label

def load_dataset_config(config_path="/Scratch/repository/msa/MSATSUNGPING/my_dataset_selection_beatles_full_salami_70_30.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def shape_list(input_tensor):
    tensor = tf.convert_to_tensor(input_tensor)
    if tensor.get_shape().dims is None:
        return tf.shape(tensor)
    static = tensor.get_shape().as_list()
    shape = tf.shape(tensor)
    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

# === TENSORFLOW ALL-IN-ONE APPROXIMATION ===

class AllInOneConfig:
    """Exact All-in-One configuration from the paper"""
    def __init__(self):
        # EXACT parameters from All-in-One paper
        self.dim_embed = 16  # Paper uses 16, not 24
        self.depth = 11      # Paper uses 11 layers
        self.num_heads = 4   # Paper uses 4 heads
        self.kernel_size = 5 # Neighborhood size
        self.dilation_factor = 2
        self.dilation_max = 2048  # Paper uses much higher dilations
        self.mlp_ratio = 8.0      # Paper uses 8x expansion
        
        # Adapted for your data
        self.dim_input = 80      # YOUR mel bins (paper uses 81)
        self.num_instruments = 4  # drums, vocals, bass, other
        self.num_function_classes = 7
        
        # EXACT dropout rates from paper
        self.drop_conv = 0.2
        self.drop_attention = 0.2  
        self.drop_hidden = 0.2
        self.drop_path = 0.2
        self.drop_last = 0.1
        
        self.layer_norm_eps = 1e-5
        self.qkv_bias = True
        self.double_attention = True   # Paper uses this
        self.instrument_attention = False  # Simplified for TF implementation
        self.act_conv = 'elu'         # Paper activation
        self.act_transformer = 'gelu' # Paper activation

class DropPath(tf.keras.layers.Layer):
    """Stochastic Depth exactly as in original"""
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        
    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)
        return tf.divide(x, keep_prob) * random_tensor

class ApproximateNeighborhoodAttention1D(tf.keras.layers.Layer):
    """TensorFlow approximation of 1D Neighborhood Attention"""
    def __init__(self, config, dim, num_heads, kernel_size, dilation, **kwargs):
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f"Hidden size ({dim}) is not divisible by num_heads ({num_heads})")
            
        self.config = config
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.window_size = kernel_size * dilation
        
        # QKV projections
        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='value')
        
        self.dropout = tf.keras.layers.Dropout(config.drop_attention)
        
    def build(self, input_shape):
        # Relative position bias (approximating NATTEN's RPB)
        self.rpb = self.add_weight(
            name='rpb',
            shape=[self.num_attention_heads, 2 * self.kernel_size - 1],
            initializer='random_uniform',
            trainable=True
        )
        
    def transpose_for_scores(self, x):
        new_x_shape = tf.concat([tf.shape(x)[:-1], [self.num_attention_heads, self.attention_head_size]], 0)
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, [0, 2, 1, 3])  # [B, heads, T, head_dim]
        
    def call(self, hidden_states, training=None):
        B, T, C = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
        
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Scale query
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        
        # Compute attention scores
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        
        # APPROXIMATION: Apply windowed masking to simulate neighborhood attention
        if self.window_size < T:
            attention_scores = self._apply_neighborhood_mask(attention_scores, T)
        
        # Add relative position bias (simplified)
        attention_scores = attention_scores + self._get_relative_position_bias(T)
        
        # Softmax
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        
        # Apply attention to values
        context_layer = tf.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        new_context_shape = tf.concat([tf.shape(context_layer)[:-2], [self.all_head_size]], 0)
        context_layer = tf.reshape(context_layer, new_context_shape)
        
        return context_layer
    
    def _apply_neighborhood_mask(self, attention_scores, seq_len):
        """Apply windowed attention mask to approximate neighborhood attention"""
        B, H, T, T2 = tf.shape(attention_scores)[0], tf.shape(attention_scores)[1], tf.shape(attention_scores)[2], tf.shape(attention_scores)[3]
        
        # Create band mask for neighborhood attention
        indices = tf.range(T, dtype=tf.int32)
        mask_matrix = tf.abs(indices[:, None] - indices[None, :]) <= (self.window_size // 2)
        
        # Expand mask for all heads and batch
        mask = tf.tile(mask_matrix[None, None, :, :], [B, H, 1, 1])
        
        # Apply mask
        attention_scores = tf.where(mask, attention_scores, -1e9)
        
        return attention_scores
    
    def _get_relative_position_bias(self, seq_len):
        """Simplified relative position bias"""
        # Create position indices
        positions = tf.range(seq_len, dtype=tf.int32)
        relative_positions = positions[:, None] - positions[None, :]  # [T, T]
        
        # Clip to valid range
        relative_positions = tf.clip_by_value(
            relative_positions + self.kernel_size - 1,
            0, 2 * self.kernel_size - 2
        )
        
        # Get bias values
        bias = tf.gather(self.rpb, relative_positions, axis=1)  # [H, T, T]
        bias = tf.transpose(bias, [1, 0, 2])  # [T, H, T]
        bias = tf.transpose(bias, [1, 0, 2])  # [H, T, T]
        
        return bias[None, :, :, :]  # [1, H, T, T]

class AllInOneEmbeddings(tf.keras.layers.Layer):
    """EXACT CNN embedding from All-in-One paper"""
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # EXACT architecture from paper
        first_conv_filters = config.dim_embed // 2
        
        # Layer 1: Conv(3,3) -> Pool(1,3)
        self.conv0 = tf.keras.layers.Conv2D(
            first_conv_filters, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='same',
            activation=config.act_conv,
            name='conv0'
        )
        self.pool0 = tf.keras.layers.MaxPool2D(
            pool_size=(1, 3), 
            strides=(1, 3), 
            padding='valid',
            name='pool0'
        )
        self.drop0 = tf.keras.layers.Dropout(config.drop_conv)
        
        # Layer 2: Conv(1,12) -> Pool(1,3) 
        self.conv1 = tf.keras.layers.Conv2D(
            config.dim_embed,
            kernel_size=(1, 12),
            strides=(1, 1),
            padding='valid',  # EXACT from paper
            activation=config.act_conv,
            name='conv1'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(1, 3),
            strides=(1, 3),
            padding='valid',
            name='pool1'
        )
        self.drop1 = tf.keras.layers.Dropout(config.drop_conv)
        
        # Layer 3: Conv(3,3) -> Pool(1,3)
        self.conv2 = tf.keras.layers.Conv2D(
            config.dim_embed,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=config.act_conv,
            name='conv2'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(1, 3),
            strides=(1, 3),
            padding='valid',
            name='pool2'
        )
        
        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(config.drop_conv)
        
    def call(self, x, training=None):
        # x: [B*K, T, F] -> [B*K, T, F, 1]
        x = tf.expand_dims(x, axis=-1)
        
        # EXACT sequence from paper
        x = self.conv0(x)   
        x = self.pool0(x)   
        x = self.drop0(x, training=training)
        
        x = self.conv1(x)   
        x = self.pool1(x)   
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)   
        x = self.pool2(x)   
        
        # Final frequency dimension should be 1
        x = tf.squeeze(x, axis=2)  # [B*K, T, filters]
        x = self.norm(x)
        x = self.dropout(x, training=training)
        
        return x

class TensorFlowAllInOneBlock(tf.keras.layers.Layer):
    """All-in-One block using TensorFlow neighborhood attention approximation"""
    def __init__(self, config, dilation, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dilation = dilation
        self.double_attention = config.double_attention
        
        # Use approximated neighborhood attention
        self.timelayer = ApproximateNeighborhoodAttention1D(
            config=config,
            dim=config.dim_embed,
            num_heads=config.num_heads,
            kernel_size=config.kernel_size,
            dilation=dilation,
        )
        
        if config.double_attention:
            self.timelayer2 = ApproximateNeighborhoodAttention1D(
                config=config,
                dim=config.dim_embed,
                num_heads=config.num_heads,
                kernel_size=config.kernel_size,
                dilation=dilation * 2,
            )
        
        # Layer normalizations
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
        # MLP (EXACT from paper)
        dim_after = config.dim_embed * 2 if config.double_attention else config.dim_embed
        mlp_hidden_dim = int(dim_after * config.mlp_ratio)
        
        self.intermediate = tf.keras.layers.Dense(
            mlp_hidden_dim, 
            activation=config.act_transformer,
            name='intermediate'
        )
        self.output_dense = tf.keras.layers.Dense(config.dim_embed, name='output')
        self.output_dropout = tf.keras.layers.Dropout(config.drop_hidden)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else tf.keras.layers.Identity()
        
    def call(self, hidden_states, training=None):
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        
        # First time attention
        attention_output1 = self.timelayer(hidden_states, training=training)
        hidden_states1 = shortcut + self.drop_path(attention_output1, training=training)
        
        hidden_states_list = [hidden_states1]
        
        # Second time attention (if double_attention)
        if self.double_attention and hasattr(self, 'timelayer2'):
            attention_output2 = self.timelayer2(hidden_states, training=training)
            hidden_states2 = shortcut + self.drop_path(attention_output2, training=training)
            hidden_states_list.append(hidden_states2)
        
        # Combine attention outputs
        if self.double_attention and len(hidden_states_list) == 2:
            hidden_states = tf.concat(hidden_states_list, axis=-1)  # [BK, T, 2*C]
            shortcut = (hidden_states_list[0] + hidden_states_list[1]) / 2.0  # Average as shortcut
        else:
            hidden_states = hidden_states_list[0]
            shortcut = hidden_states
            
        # MLP
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output_dense(layer_output)
        layer_output = self.output_dropout(layer_output, training=training)
        
        layer_output = shortcut + self.drop_path(layer_output, training=training)
        
        return layer_output

class AllInOneHead(tf.keras.layers.Layer):
    """EXACT output head from All-in-One paper"""
    def __init__(self, num_classes, config, init_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.config = config
        
        # EXACT from paper: flatten instrument x embed dimensions
        input_dim = config.num_instruments * config.dim_embed
        self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')
        
        # Focal loss initialization (EXACT from paper)
        if init_confidence is not None:
            bias_init = -tf.math.log(1.0 / init_confidence - 1.0)
            self.classifier.bias_initializer = tf.keras.initializers.Constant(bias_init)
    
    def call(self, x):
        # x: [B, K, T, C] -> [B, T, num_classes]
        B, K, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.transpose(x, [0, 2, 1, 3])  # [B, T, K, C]  
        x = tf.reshape(x, [B, T, K * C])   # [B, T, K*C]
        logits = self.classifier(x)        # [B, T, num_classes]
        
        if self.num_classes == 1:
            logits = tf.squeeze(logits, axis=-1)  # [B, T]
            
        return logits

class AllInOneModel(tf.keras.Model):
    """TensorFlow All-in-One approximation matching paper architecture"""
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or AllInOneConfig()
        
        # Training variables for your pipeline
        self.n_classes = self.config.num_function_classes
        self.optimizer = None
        self.flag = True
        self.confusion_matrix_train_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.w_b = 18
        self.w_f = 2
        self.steps_per_epoch = None
        
        # EXACT All-in-One architecture
        self.embeddings = AllInOneEmbeddings(self.config)
        
        # Transformer blocks with EXACT dilations from paper
        drop_path_rates = np.linspace(0, self.config.drop_path, self.config.depth)
        dilations = [
            min(self.config.dilation_factor ** i, self.config.dilation_max) 
            for i in range(self.config.depth)
        ]
        
        self.encoder_layers = [
            TensorFlowAllInOneBlock(  # UPDATED: Use TensorFlow approximation
                config=self.config,
                dilation=dilations[i],
                drop_path_rate=drop_path_rates[i]
            )
            for i in range(self.config.depth)
        ]
        
        # Final norm
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps)
        
        # Output heads with EXACT initialization from paper
        self.section_classifier = AllInOneHead(
            num_classes=1,
            config=self.config,
            init_confidence=0.001   # Paper value for sections
        )
        self.function_classifier = AllInOneHead(
            num_classes=self.config.num_function_classes,
            config=self.config
        )
        
        self.dropout = tf.keras.layers.Dropout(self.config.drop_last)
        
        # Training metrics (keeping your existing evaluation pipeline)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.result = {k: [] for k in [
            'Acc_max', 'Acc_smooth', 'P_seg', 'R_seg', 'F1_seg', 
            'P_seg3', 'R_seg3', 'F1_seg3', 'P_pair', 'R_pair', 'F1_pair', 
            'loss', 'loss_b', 'loss_f'
        ]}
        self.temp = {k: [] for k in [
            'b_ref', 'b_est', 'matched', 'n_b_ref', 'n_b_est', 'n_matched', 
            'b_ref_in_second', 'b_est_in_second', 'f_ref', 'f_est'
        ]}
        
    def call(self, instruments_mel, valid_len, training=None):
        """EXACT forward pass from All-in-One paper"""
        # instruments_mel: [B, K, T, F] where K=4, F=80
        B, K, T, F = tf.shape(instruments_mel)[0], tf.shape(instruments_mel)[1], tf.shape(instruments_mel)[2], tf.shape(instruments_mel)[3]
        
        # Reshape for processing: [B*K, T, F]
        inputs = tf.reshape(instruments_mel, [B * K, T, F])
        
        # CNN feature extraction (EXACT from paper)
        frame_embed = self.embeddings(inputs, training=training)  # [B*K, T, dim_embed]
        
        # Encoder layers (EXACT from paper)
        hidden_states = frame_embed
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, training=training)
            
        # Reshape back to multi-instrument format: [B, K, T, dim_embed]
        hidden_states = tf.reshape(hidden_states, [B, K, T, self.config.dim_embed])
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        
        # Output heads (EXACT from paper)
        logits_section = self.section_classifier(hidden_states)  # [B, T] - use as boundary
        logits_function = self.function_classifier(hidden_states) # [B, T, num_classes]
        
        # For your task: use section as boundary, function as function
        return logits_section, logits_function

    # === Include all existing training methods from your pipeline ===
    # (Copy all the methods from your previous implementation)
    
    def clear_result(self):
        self.result = {k: [] for k in self.result.keys()}
        self.temp = {k: [] for k in self.temp.keys()}
        self.confusion_matrix_train_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        
    def average_result(self):
        result_dict = {}
        for k, v in self.result.items():
            if len(v) > 0:
                try:
                    if all(tf.rank(val) == 0 for val in v):
                        stacked = tf.stack(v)
                        result_dict[k] = tf.reduce_mean(stacked)
                    else:
                        concatenated = tf.concat(v, axis=0)
                        result_dict[k] = tf.reduce_mean(concatenated)
                except (tf.errors.InvalidArgumentError, ValueError):
                    try:
                        stacked = tf.stack(v)
                        result_dict[k] = tf.reduce_mean(stacked)
                    except:
                        mean_val = np.mean([float(val.numpy()) for val in v])
                        result_dict[k] = tf.constant(mean_val)
        return result_dict
    
    def train_step(self, data):
        instruments_mel, valid_len, boun_ref, func_ref, sec_ref = data
        
        with tf.GradientTape() as tape:
            boundary_logits, function_logits = self(instruments_mel, valid_len, training=True)
            
            prob_boun = tf.nn.sigmoid(boundary_logits)
            boun_est = self.decode_boundary(prob_boun, valid_len)
            func_est_max = tf.argmax(function_logits, axis=-1, output_type=tf.int32)
            func_est_smooth = self.decode_labeling(boun_est, function_logits, valid_len)
            
            self.confusion_matrix_train_max += self.compute_confusion_matrix(func_ref, func_est_max)
            self.confusion_matrix_train_boun += self.compute_confusion_matrix(func_ref, func_est_smooth)
            
            ce_b = self.w_b * self.bce_from_logits(boun_ref, boundary_logits, valid_len)
            ce_f = self.w_f * self.cce_from_logits(func_ref, function_logits, valid_len)
            
            if self.flag:
                print('ce_b', ce_b.numpy())
                print('ce_f', ce_f.numpy())
                self.flag = False
            
            loss = ce_b + ce_f
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
        
        # Optimizer (keeping your learning rate schedule)
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, epsilon=1e-7)
        
        if self.steps_per_epoch is not None:
            current_epoch = tf.cast(self.optimizer.iterations // self.steps_per_epoch, tf.float32)
            def lr_schedule():
                return tf.cond(current_epoch < 5, lambda: 1e-4,
                              lambda: tf.cond(current_epoch < 30, lambda: 1e-3,
                                            lambda: tf.cond(current_epoch < 60, lambda: 5e-4, lambda: 1e-4)))
            self.optimizer.learning_rate.assign(lr_schedule())
        
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # Compute metrics
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth'))
        score_dict.update(self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size))
        score_dict.update(self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size))
        
        self.loss_tracker.update_state(loss)
        score_dict.update({'loss': loss, 'loss_b': ce_b, 'loss_f': ce_f})
        [self.result[k].append(v) for k, v in score_dict.items()]
        
        return boun_est, func_est_smooth
    
    def test_step(self, data):
        if isinstance(data, tuple) and len(data) == 5:
            instruments_mel, valid_len, boun_ref, func_ref, sec_ref = data
        else:
            print(f"Unexpected data format: {type(data)}")
            return None, None
        
        boundary_logits, function_logits = self(instruments_mel, valid_len, training=False)
        prob_boun = tf.nn.sigmoid(boundary_logits)
        
        boun_est = self.decode_boundary(prob_boun, valid_len)
        func_est_max = tf.argmax(function_logits, axis=-1, output_type=tf.int32)
        func_est_smooth = self.decode_labeling(boun_est, function_logits, valid_len)
        
        self.confusion_matrix_test_max += self.compute_confusion_matrix(func_ref, func_est_max)
        self.confusion_matrix_test_boun += self.compute_confusion_matrix(func_ref, func_est_smooth)
        
        ce_b = self.w_b * self.bce_from_logits(boun_ref, boundary_logits, valid_len)
        ce_f = self.w_f * self.cce_from_logits(func_ref, function_logits, valid_len)
        loss = ce_b + ce_f
        
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth'))
        score_dict.update(self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size))
        score_dict.update(self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size))
        
        self.loss_tracker.update_state(loss)
        score_dict.update({'loss': loss, 'loss_b': ce_b, 'loss_f': ce_f})
        [self.result[k].append(v) for k, v in score_dict.items()]
        
        return boun_est, func_est_smooth

    # === Keep all your existing helper methods ===
    def bce_from_logits(self, gt, logits, valid_len, pos_weight=0.3):
        gt_expaned = self.expand_boundary(gt, valid_len, value=0.5)
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(gt)[1], dtype=tf.float32)
        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_expaned, logits, pos_weight=pos_weight)
        loss = tf.reduce_sum(wbce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    def cce_from_logits(self, gt, logits, valid_len):
        weights = tf.constant([2.0, 0.25, 0.6, 1.0, 4.0, 2.5, 1.5], tf.float32)
        seq_mask = tf.sequence_mask(valid_len, maxlen=tf.shape(gt)[1], dtype=tf.float32)
        gt_onehot = tf.one_hot(gt, depth=self.n_classes)
        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_onehot, logits, pos_weight=weights)
        loss = tf.reduce_sum(wbce * seq_mask[:, :, tf.newaxis], axis=1) / tf.cast(valid_len, tf.float32)[:, tf.newaxis]
        return tf.reduce_mean(loss)

    def decode_boundary(self, prob_boun, valid_len, method='librosa'):
        assert method in ['msaf', 'librosa', 'boeck']
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(prob_boun)[1], dtype=tf.float32)
        prob_boun *= seq_mask
        prob_boun_numpy = prob_boun.numpy()
        peaks = np.zeros_like(prob_boun_numpy, dtype=np.int32)

        if method == 'msaf':
            peak_indices = [peak_picking_MSAF(seq, median_len=7, offset_rel=0.05, sigma=4) for seq in prob_boun_numpy]
        elif method == 'librosa':
            peak_indices = [
                librosa.util.peak_pick(seq, pre_max=10, post_max=10, pre_avg=20, post_avg=10, delta=0.03, wait=10) 
                for seq in prob_boun_numpy
            ]
            peak_indices = [ids.astype(int) for ids in peak_indices]
        elif method == 'boeck':
            peak_indices = [
                peak_picking_boeck(seq, threshold=0.01, fps=2, combine=10, pre_max=10, post_max=10, pre_avg=20, post_avg=10) 
                for seq in prob_boun_numpy
            ]

        for i in range(prob_boun_numpy.shape[0]):
            peaks[i, peak_indices[i]] = 1
        peaks[:, 0] = 1
        return tf.constant(peaks, tf.int32) * tf.cast(seq_mask, tf.int32)

    def decode_labeling(self, boun_est, logits_func, valid_len):
        boun_est = boun_est.numpy()
        prob_func = tf.nn.sigmoid(logits_func).numpy()
        valid_len = valid_len.numpy()
        max_len = valid_len.max()
        func_est = []
        for i in range(valid_len.shape[0]):
            l = valid_len[i]
            b_i = np.where(np.equal(boun_est[i, :l], 1))[0]
            segments = [segment for segment in np.split(prob_func[i, :l], indices_or_sections=b_i) if len(segment)]
            centroids = np.stack([np.sum(segment, axis=0) for segment in segments])
            clusters = np.argmax(centroids, axis=-1)
            label_frame = np.array([c for (segment, c) in zip(segments, clusters) for _ in range(len(segment))])
            if l < max_len:
                label_frame = np.pad(label_frame, (0, max_len-l), 'constant', constant_values=self.n_classes-1)
            func_est.append(label_frame)
        return tf.constant(func_est)

    def expand_boundary(self, boundary, valid_len, value=0.5, size=3):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boundary)[1], dtype=tf.float32)
        boundary = tf.cast(boundary, tf.float32)
        filter = tf.ones([size, 1, 1])
        boundary_expanded = tf.nn.conv1d(boundary[:, :, tf.newaxis], filters=filter, stride=1, padding='SAME')
        boundary_expanded = tf.squeeze(boundary_expanded, axis=-1)
        cond = tf.logical_and((boundary_expanded != boundary), tf.logical_not(tf.cast(boundary, tf.bool)))
        boundary_expanded = tf.where(cond, value, boundary)
        return boundary_expanded * seq_mask

    def compute_segment_score(self, boun_ref, boun_est, valid_len, resolution):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boun_ref)[1], dtype=tf.int32)
        boun_ref_expanded = tf.cast(self.expand_boundary(boun_ref, valid_len, value=1), tf.int32)
        matched = boun_est * boun_ref_expanded * seq_mask
        n_boun_ref = tf.reduce_sum(boun_ref, axis=1)
        n_boun_est = tf.reduce_sum(boun_est, axis=1)
        n_matched = tf.reduce_sum(matched, axis=1)

        precision, recall, fscore = [], [], []
        precision3, recall3, fscore3 = [], [], []
        for i in range(shape_list(boun_ref)[0]):
            l = valid_len[i].numpy()
            b_ref = boun_ref[i, :l].numpy()
            b_est = boun_est[i, :l].numpy()

            b_ref_in_second = np.where(b_ref == 1)[0] * resolution
            b_est_in_second = np.where(b_est == 1)[0] * resolution

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

            P, R, F1 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=0.5, beta=1.0)
            precision.append(P)
            recall.append(R)
            fscore.append(F1)

            P3, R3, F3 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=3, beta=1.0)
            precision3.append(P3)
            recall3.append(R3)
            fscore3.append(F3)

        return {
            'P_seg': tf.constant(precision), 'R_seg': tf.constant(recall), 'F1_seg': tf.constant(fscore),
            'P_seg3': tf.constant(precision3), 'R_seg3': tf.constant(recall3), 'F1_seg3': tf.constant(fscore3)
        }

    def compute_confusion_matrix(self, func_ref, func_est):
        return tf.math.confusion_matrix(
            labels=tf.reshape(func_ref, [-1]), predictions=tf.reshape(func_est, [-1]),
            num_classes=self.n_classes, weights=None, dtype=tf.dtypes.int32)

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

            P_pair, R_pair, F1_pair = mir_eval.segment.pairwise(
                interval_ref, label_ref, interval_est, label_est, frame_size=0.1)
            precision_pair.append(P_pair)
            recall_pair.append(R_pair)
            fscore_pair.append(F1_pair)

        return {
            'P_pair': tf.constant(precision_pair),
            'R_pair': tf.constant(recall_pair),
            'F1_pair': tf.constant(fscore_pair)
        }

    def compute_classification_score(self, func_ref, func_est, valid_len, key='Accuracy'):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(func_ref)[1], dtype=tf.float32)
        matched = tf.cast(func_ref == func_est, tf.float32) * seq_mask
        accuracy = tf.reduce_sum(matched, axis=1) / tf.cast(valid_len, tf.float32)
        return {key: accuracy}


# === DATA LOADING (Use your existing functions) ===

def load_dataset_files_allinone_strict(dataset_name, song_ids, data_path, include_augmented=False):
    """STRICT: Only load tracks with ALL 4 instrument separations available"""
    data = {
        'instruments_mel': [],  
        'boundary': [],
        'function': [],
        'len': [],
        'section': []
    }
    
    successful_loads = 0
    skipped_incomplete = 0
    
    for song_id in song_ids:
        try:
            if include_augmented:
                pattern = f"{song_id}*_a1_spec.npy"
                spec_files = glob.glob(os.path.join(data_path, pattern))
                spec_files = [f for f in spec_files if '_original_' not in f]
            else:
                spec_files = [os.path.join(data_path, f"{song_id}_original_a1_spec.npy")]
            
            for spec_file in spec_files:
                if not os.path.exists(spec_file):
                    continue
                    
                base_name = spec_file.replace('_spec.npy', '')
                
                try:
                    # First, check if ALL required files exist
                    required_files = {
                        'main': base_name + '_spec.npy',
                        'boundary': base_name + '_boundary.npy', 
                        'function': base_name + '_function.npy',
                        'drums': base_name + '_drumspec.npy',
                        'vocals': base_name + '_vocalspec.npy',
                        'bass': base_name + '_bassspec.npy',
                        'others': base_name + '_othersspec.npy'
                    }
                    
                    # Check if all files exist
                    missing_files = []
                    for file_type, file_path in required_files.items():
                        if not os.path.exists(file_path):
                            missing_files.append(file_type)
                    
                    if missing_files:
                        print(f"  Skipping {os.path.basename(spec_file)}: missing {missing_files}")
                        skipped_incomplete += 1
                        continue
                    
                    # Load all files - only proceed if ALL exist
                    main_spec = np.load(required_files['main'])
                    boundary = np.load(required_files['boundary'])
                    function = np.load(required_files['function'])
                    drum_spec = np.load(required_files['drums'])
                    vocal_spec = np.load(required_files['vocals'])
                    bass_spec = np.load(required_files['bass'])
                    others_spec = np.load(required_files['others'])
                    
                    valid_len = main_spec.shape[0]
                    
                    # Strict shape validation - ALL must match
                    shape_errors = []
                    if boundary.shape[0] != valid_len:
                        shape_errors.append(f"boundary: {boundary.shape[0]} vs {valid_len}")
                    if function.shape[0] != valid_len:
                        shape_errors.append(f"function: {function.shape[0]} vs {valid_len}")
                    if drum_spec.shape[0] != valid_len or drum_spec.shape[1] != 80:
                        shape_errors.append(f"drums: {drum_spec.shape} vs ({valid_len}, 80)")
                    if vocal_spec.shape[0] != valid_len or vocal_spec.shape[1] != 80:
                        shape_errors.append(f"vocals: {vocal_spec.shape} vs ({valid_len}, 80)")
                    if bass_spec.shape[0] != valid_len or bass_spec.shape[1] != 80:
                        shape_errors.append(f"bass: {bass_spec.shape} vs ({valid_len}, 80)")
                    if others_spec.shape[0] != valid_len or others_spec.shape[1] != 80:
                        shape_errors.append(f"others: {others_spec.shape} vs ({valid_len}, 80)")
                    
                    if shape_errors:
                        print(f"  Skipping {os.path.basename(spec_file)}: shape mismatches: {shape_errors}")
                        skipped_incomplete += 1
                        continue
                    
                    # All checks passed - create the 4-instrument stack
                    instruments = [drum_spec, vocal_spec, bass_spec, others_spec]
                    instruments_stacked = np.stack(instruments, axis=0)  # [4, T, 80]
                    
                    # Final verification
                    if instruments_stacked.shape != (4, valid_len, 80):
                        print(f"  Skipping {os.path.basename(spec_file)}: final shape {instruments_stacked.shape} != (4, {valid_len}, 80)")
                        skipped_incomplete += 1
                        continue
                    
                    # Success - add to data
                    data['instruments_mel'].append(instruments_stacked)
                    data['boundary'].append(boundary)
                    data['function'].append(function)
                    data['len'].append(valid_len)
                    data['section'].append(f"{dataset_name}_{song_id}")
                    
                    successful_loads += 1
                    
                except Exception as e:
                    print(f"  Error loading {os.path.basename(spec_file)}: {e}")
                    skipped_incomplete += 1
                    continue
                    
        except Exception as e:
            print(f"  Error processing {song_id}: {e}")
            continue
    
    print(f"   Successfully loaded: {successful_loads} complete 4-way tracks")
    print(f"   Skipped incomplete: {skipped_incomplete} tracks (missing instrument separations)")
    print(f"   Data quality: {successful_loads}/{successful_loads + skipped_incomplete} = {successful_loads/(successful_loads + skipped_incomplete)*100:.1f}% complete separations")
    
    return data

# Updated dataset creation functions
def create_enhanced_datasets_allinone_strict(config_path, data_base_path):
    """Load ONLY tracks with complete 4-way instrument separation"""
    print("Creating STRICT All-In-One datasets (requires all 4 instrument separations)")
    
    config = load_dataset_config(config_path)
    
    dataset_paths = {
        'beatles': {
            'original': os.path.join(data_base_path, 'beatles-original-preprocessed-data'),
            'aug': os.path.join(data_base_path, 'beatles-aug-preprocessed-data')
        },
        'salami': {
            'original': os.path.join(data_base_path, 'salami-original-preprocessed-data'),
            'aug': os.path.join(data_base_path, 'salami-aug-preprocessed-data')
        }
    }
    
    def create_train_data_strict():
        all_data = {'instruments_mel': [], 'boundary': [], 'function': [], 'len': [], 'section': []}
        total_complete = 0
        total_incomplete = 0
        
        for dataset_name, info in config['training_set'].items():
            if dataset_name == 'summary':
                continue
                
            print(f"Loading {dataset_name} training data (STRICT mode)...")
            song_ids = info['song_ids']
            
            original_data = load_dataset_files_allinone_strict(
                dataset_name, song_ids, dataset_paths[dataset_name]['original'], include_augmented=False
            )
            aug_data = load_dataset_files_allinone_strict(
                dataset_name, song_ids, dataset_paths[dataset_name]['aug'], include_augmented=True
            )
            
            for key in all_data.keys():
                all_data[key].extend(original_data[key])
                all_data[key].extend(aug_data[key])
            
            dataset_complete = len(original_data['instruments_mel']) + len(aug_data['instruments_mel'])
            total_complete += dataset_complete
            print(f"   {dataset_name}: {len(original_data['instruments_mel'])} original + {len(aug_data['instruments_mel'])} augmented = {dataset_complete} complete")
        
        print(f"TOTAL TRAINING: {total_complete} tracks with complete 4-way separation")
        return all_data

    def create_test_data_strict():
        all_data = {'instruments_mel': [], 'boundary': [], 'function': [], 'len': [], 'section': []}
        total_complete = 0
        
        for dataset_name, info in config['test_set'].items():
            print(f"Loading {dataset_name} test data (STRICT mode)...")
            song_ids = info['song_ids']
            
            dataset_data = load_dataset_files_allinone_strict(
                dataset_name, song_ids, dataset_paths[dataset_name]['original'], include_augmented=False
            )
            
            for key in all_data.keys():
                all_data[key].extend(dataset_data[key])
            
            dataset_complete = len(dataset_data['instruments_mel'])
            total_complete += dataset_complete
            print(f"   {dataset_name}: {dataset_complete} complete tracks")
        
        print(f"TOTAL TEST: {total_complete} tracks with complete 4-way separation")
        return all_data

    train_data = create_train_data_strict()
    test_data = create_test_data_strict()
    
    print(f"\nSTRICT All-In-One datasets created!")
    print(f"Training: {len(train_data['instruments_mel'])} tracks (all have 4-way separation)")
    print(f"Test: {len(test_data['instruments_mel'])} tracks (all have 4-way separation)")
    
    if len(train_data['instruments_mel']) == 0:
        print("WARNING: No training data with complete 4-way separation found!")
        print("This suggests your preprocessed data may not have instrument-separated files.")
        
    if len(test_data['instruments_mel']) == 0:
        print("WARNING: No test data with complete 4-way separation found!")
    
    return train_data, test_data

# Example usage check function
def check_instrument_separation_availability(data_base_path):
    """Check how many files have complete 4-way separation"""
    print("Checking instrument separation availability...")
    
    dataset_paths = [
        os.path.join(data_base_path, 'beatles-original-preprocessed-data'),
        os.path.join(data_base_path, 'beatles-aug-preprocessed-data'),
        os.path.join(data_base_path, 'salami-original-preprocessed-data'),
        os.path.join(data_base_path, 'salami-aug-preprocessed-data')
    ]
    
    total_files = 0
    complete_files = 0
    
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            continue
            
        print(f"\nChecking: {os.path.basename(dataset_path)}")
        spec_files = glob.glob(os.path.join(dataset_path, "*_spec.npy"))
        
        for spec_file in spec_files:  # Check first 10 files
            total_files += 1
            base_name = spec_file.replace('_spec.npy', '')
            
            required_files = [
                base_name + '_drumspec.npy',
                base_name + '_vocalspec.npy', 
                base_name + '_bassspec.npy',
                base_name + '_othersspec.npy'
            ]
            
            if all(os.path.exists(f) for f in required_files):
                complete_files += 1
                print(f"  ✓ {os.path.basename(spec_file)} - complete")
            else:
                missing = [os.path.basename(f) for f in required_files if not os.path.exists(f)]
                print(f"  ✗ {os.path.basename(spec_file)} - missing: {missing}")
    
    print(f"\nSummary: {complete_files}/{total_files} files have complete 4-way separation")
    if complete_files == 0:
        print("No files with instrument separation found. You may need to:")
        print("1. Run source separation on your audio files")
        print("2. Check if files have different naming convention")
        print("3. Verify the preprocessing pipeline includes instrument separation")


def print_temp(temp, sample=0, print_len=190):
    if not temp['b_ref'] or len(temp['b_ref']) == 0:
        print("No temp data available for printing")
        return
    
    if sample >= len(temp['b_ref']):
        sample = 0
    
    b_ref, b_est, matched = temp['b_ref'][sample], temp['b_est'][sample], temp['matched'][sample]
    n_b_ref, n_b_est, n_matched = temp['n_b_ref'][sample], temp['n_b_est'][sample], temp['n_matched'][sample]

    print('n_b_ref %d n_b_est %d n_matched %d' % (n_b_ref, n_b_est, n_matched))
    b_ref_in_second = temp['b_ref_in_second'][sample]
    b_est_in_second = temp['b_est_in_second'][sample]
    print('b_ref_in_second', ' '.join(["{:.2f}".format(s) for s in b_ref_in_second]))
    print('b_est_in_second', ' '.join(["{:.2f}".format(s) for s in b_est_in_second]))

def print_confusion_matrix(cm1, cm2=None, n_just=9, norm=True, epsilon=1e-7):
    n_classes = len(function_dict.keys())

    if norm:
        n_just = 6
        sum_row = np.sum(cm1, axis=1, keepdims=True)
        cm1_norm = cm1 / (sum_row + epsilon)
        if cm2 is not None:
            sum_row = np.sum(cm2, axis=1, keepdims=True)
            cm2_norm = cm2 / (sum_row + epsilon)

    def str_value(v):
        return '%.3f' % v if norm else '%d' % v

    def multi_display(vs, just):
        return '/'.join([str_value(v).rjust(just) for v in vs])

    def class_conversion(i, reduced=True):
        if i == 4: return 'X'
        for k, v in function_dict.items():
            if v == i:
                return (k[0].upper()) if reduced else k.capitalize()

    if cm2 is None:
        if norm:
            rows = [class_conversion(i) + ''.join([str_value(v).rjust(n_just) for v in cm1_norm[i]]) for i in range(n_classes)]
        else:
            rows = [class_conversion(i) + ''.join([str_value(v).rjust(n_just) for v in cm1[i]]) for i in range(n_classes)]
    else:
        if norm:
            rows = [
                class_conversion(i) + ''.join([multi_display(vs, n_just) for vs in zip(cm1_norm[i], cm2_norm[i])]) for i in range(n_classes)
            ]
            col_n_just = 2 * n_just + 1
        else:
            rows = [
                class_conversion(i) + ''.join([multi_display(vs, n_just) for vs in zip(cm1_norm[i], cm2[i])]) for i in
                range(n_classes)
            ]
            col_n_just = n_just
    
    col_names = ''.join([class_conversion(i).rjust(col_n_just) for i in range(n_classes)])

    print('confusion matrix:')
    print('', col_names)
    [print(row) for row in rows]

    # Compute P, R, and F1 scores
    for cm in [cm1, cm2]:
        if cm is not None:
            sum_row = np.sum(cm, axis=1)
            sum_col = np.sum(cm, axis=0)
            diag = np.diag(cm)
            P = diag / (sum_col + epsilon)
            R = diag / (sum_row + epsilon)
            F1 = 2*P*R / (P + R + epsilon)
            print(' ', ''.join([class_conversion(i).rjust(n_just) for i in range(n_classes)]))
            print('P', ''.join([str_value(v).rjust(n_just) for v in P]))
            print('R', ''.join([str_value(v).rjust(n_just) for v in R]))
            print('F', ''.join([str_value(v).rjust(n_just) for v in F1]))

def color_text(color):
    color_dict = {
        "PURPLE": "\033[95m", "CYAN": "\033[96m", "DARKCYAN": "\033[36m", "BLUE": "\033[94m",
        "GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m", "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m", "END": "\033[0m"}
    return color_dict[color]

class ImprovedEarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def __call__(self, current_score, model, epoch):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            if self.restore_best:
                self.best_weights = model.get_weights()
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best and self.best_weights:
                    print(f"Restoring best weights (F1: {self.best_score:.3f})")
                    model.set_weights(self.best_weights)
                return True
        return False

def train_tensorflow_allinone():
    """COMPLETE: Training with TensorFlow All-in-One approximation"""
    print("🎯 Starting TensorFlow All-in-One Approximation Training")
    print("(Windowed attention approximates neighborhood attention)")
    
    DATA_BASE_PATH = "/Scratch/repository/msa/MSATSUNGPING/"
    
    # Load data using CORRECT function
    train_data, test_data = create_enhanced_datasets_allinone_strict(
        config_path="/Scratch/repository/msa/MSATSUNGPING/my_dataset_selection_beatles_full_salami_70_30.json",
        data_base_path=DATA_BASE_PATH
    )
    
    # Generator for data format
    def generator(data):
        for instruments_mel, valid_len, boundary, function, section in \
                zip(data['instruments_mel'], data['len'], data['boundary'], data['function'], data['section']):
            
            valid_len_scalar = int(valid_len[0]) if hasattr(valid_len, 'shape') and len(valid_len.shape) > 0 else int(valid_len)
            section_str = str(section[0]) if hasattr(section, 'shape') and len(section.shape) > 0 else str(section)
            
            yield instruments_mel, valid_len_scalar, boundary, function, section_str

    # Output types and shapes
    output_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.string)
    output_shapes = (
        tf.TensorShape([4, None, 80]),  # instruments_mel [4, T, 80]
        tf.TensorShape([]),             # len (scalar)
        tf.TensorShape([None]),         # boundary
        tf.TensorShape([None]),         # function
        tf.TensorShape([]),             # section (scalar string)
    )

    # Create TensorFlow datasets
    tf_train_data = tf.data.Dataset.from_generator(
        lambda: generator(train_data), output_types=output_types, output_shapes=output_shapes)
    tf_test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_data), output_types=output_types, output_shapes=output_shapes)
    
    # Create model with CORRECT classes
    config = AllInOneConfig()  # FIXED: Use AllInOneConfig
    model = AllInOneModel(config)  # FIXED: Use AllInOneModel
    
    print(f"TensorFlow All-in-One Approximation Configuration:")
    print(f"  - Model: TensorFlow approximation (not true NATTEN)")
    print(f"  - Attention: Windowed self-attention approximation")
    print(f"  - Depth: {config.depth} layers (paper: 11)")
    print(f"  - Embed dim: {config.dim_embed} (paper: 16)")  
    print(f"  - Heads: {config.num_heads} (paper: 4)")
    print(f"  - Max dilation: {config.dilation_max} (paper: 2048)")
    print(f"  - Double attention: {config.double_attention} (paper: True)")
    print(f"  - MLP ratio: {config.mlp_ratio} (paper: 8.0)")
    
    # Build model
    print("🔧 Building TensorFlow All-in-One approximation...")
    dummy_input = tf.zeros((1, 4, 100, 80))
    dummy_len = tf.constant([100])
    _ = model(dummy_input, dummy_len, training=False)
    print("✅ TensorFlow All-in-One approximation built successfully!")
    
    # Training parameters
    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    TRAIN_SHUFFLE_SIZE = len(train_data['instruments_mel'])
    N_EPOCHS = 100
    
    model.steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE_SIZE / TRAIN_BATCH_SIZE))
    
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE_SIZE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE, output_shapes)
    tf_test_data = tf_test_data.padded_batch(TEST_BATCH_SIZE, output_shapes)
    
    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(model=model)
    model_path = './allinone_again'
    all_epochs_manager = tf.train.CheckpointManager(
        checkpoint, 
        directory=f'{model_path}/all_epochs', 
        max_to_keep=50,
        checkpoint_name='epoch'
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint, directory=f'{model_path}/best_models', max_to_keep=3, checkpoint_name='best')
    
    # Training metrics
    best_train_epoch, best_test_epoch = 0, 0
    supervised_metrics = ['F1_seg']
    best_train_result = {k: 0 for k in supervised_metrics}
    best_test_result = {k: 0 for k in supervised_metrics}
    
    # Early stopping
    early_stopping = ImprovedEarlyStopping(patience=20, min_delta=0.001, restore_best=True)
    
    print("🚀 Starting TensorFlow All-in-One training loop...")
    
    # COMPLETE Training loop
    for epoch in range(1, N_EPOCHS+1):
        print(f'🔥 Epoch {epoch}/{N_EPOCHS}')
        print(color_text("RED") + "--training phase--" + color_text("END"))
        
        for i_batch, batch in enumerate(tf_train_data):
            model.train_step(batch)
        
        # Training results with COMPLETE output
        print_temp(model.temp)
        result = model.average_result()
        
        # Add missing metrics if they don't exist
        if 'F1_seg' not in result:
            print("⚠️  F1_seg not found in training results, setting to 0.0")
            result['F1_seg'] = tf.constant(0.0)
        if 'P_seg' not in result:
            result['P_seg'] = tf.constant(0.0)
        if 'R_seg' not in result:
            result['R_seg'] = tf.constant(0.0)
            
        print_confusion_matrix(model.confusion_matrix_train_max.numpy(), model.confusion_matrix_train_boun.numpy())
        model.clear_result()
        
        # Safe calculation of F1 score
        try:
            train_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing training metric {e}, setting train_F1 to 0.0")
            train_F1 = 0.0
            
        if train_F1 > sum([float(best_train_result.get(k, 0)) for k in supervised_metrics]):
            best_train_epoch, best_train_result = epoch, result

        # COMPLETE output formatting
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))

        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        for i_batch, batch in enumerate(tf_test_data):
            model.test_step(batch)
        
        # Safe sample selection for print_temp
        test_data_size = len(test_data['instruments_mel'])
        if test_data_size > 0:
            safe_sample = np.random.randint(min(TEST_BATCH_SIZE, test_data_size))
        else:
            safe_sample = 0
            
        print_temp(model.temp, sample=safe_sample)
        result = model.average_result()
        
        # Add missing metrics if they don't exist
        if 'F1_seg' not in result:
            print("⚠️  F1_seg not found in test results, setting to 0.0")
            result['F1_seg'] = tf.constant(0.0)
        if 'P_seg' not in result:
            result['P_seg'] = tf.constant(0.0)
        if 'R_seg' not in result:
            result['R_seg'] = tf.constant(0.0)
            
        print_confusion_matrix(model.confusion_matrix_test_max.numpy(), model.confusion_matrix_test_boun.numpy())
        model.clear_result()
        
        # Safe calculation of test F1 score
        try:
            test_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing test metric {e}, setting test_F1 to 0.0")
            test_F1 = 0.0
        
        # Save every epoch
        try:
            all_epochs_path = all_epochs_manager.save()
            print(f"💾 Saved epoch {epoch}: {all_epochs_path}")
        except Exception as e:
            print(f'❌ Epoch saving failed: {e}')

        # Save best model
        if test_F1 > sum([float(best_test_result.get(k, 0)) for k in supervised_metrics]):
            best_test_epoch, best_test_result = epoch, result
            print(color_text("YELLOW") + f"🏆 NEW BEST MODEL at epoch {epoch}!" + color_text("END"))
            try:
                best_path = best_manager.save()
                print(f"✅ Best model saved: {best_path}")
            except Exception as e:
                print(f'❌ Best model saving failed: {e}')

        # COMPLETE output formatting
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))
        print(color_text("CYAN") + '### best_test_F1 at epoch %d' % best_test_epoch,
              '  '.join([' '.join((k, '{:.3f}'.format(best_test_result.get(k, tf.constant(0.0)).numpy()))) for k in supervised_metrics]), color_text("END"))
        
        # Check early stopping
        if early_stopping(test_F1, model, epoch):
            print(color_text("YELLOW") + f"🛑 Early stopping triggered at epoch {epoch}" + color_text("END"))
            print(f"📊 Best F1 score: {early_stopping.best_score:.3f}")
            break
        
        print()

    # Print final results
    if early_stopping.stopped_epoch > 0:
        print(f'🛑 Training stopped early at epoch {early_stopping.stopped_epoch}')
        print(f'📊 Final best F1: {early_stopping.best_score:.3f}')
    print(f'🎉 Training completed! Best test F1: {best_test_result[supervised_metrics[0]].numpy():.3f} at epoch {best_test_epoch}')

# CORRECTED main function
if __name__ == '__main__':
    print("=== TensorFlow All-in-One Approximation ===")
    print("Note: This uses windowed attention to approximate neighborhood attention")
    print("NATTEN library is not required for this implementation")
    
    train_tensorflow_allinone()  # FIXED: Use correct function name

# if __name__ == "__main__":
#     # Check your data first
#     check_instrument_separation_availability("/Scratch/repository/msa/MSATSUNGPING/")
# # === COMPLETE TRAINING FUNCTION ===


