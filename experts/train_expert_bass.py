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
import pandas as pd
import csv                    
from datetime import datetime  


# Use GPUs 2 and 3 (the free ones!)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

global_frame_size = 0.5

# === MISSING FUNCTIONS - ADD THESE TO YOUR CODE ===

# 1. Missing function dictionary
function_dict = {
    'intro': 0,
    'verse': 1,
    'chorus': 2,
    'bridge': 3,
    'inst': 4,
    'outro': 5,
    'silence': 6,
}

# 2. Missing peak picking functions
def peak_picking_MSAF(x, median_len=9, offset_rel=0.05, sigma=4.0):
    """Peak picking strategy following MSFA using an adaptive threshold"""
    offset = x.mean() * offset_rel
    x = gaussian_filter1d(x, sigma=sigma)
    threshold_local = median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks, dtype=np.int32)
    return peaks

def peak_picking_boeck(activations, threshold=0.5, fps=100, include_scores=False, combine=False,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """Peak picking method described in Boeck et al."""
    activations = activations.ravel()

    # detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = filters.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = filters.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        detections = detections * (detections >= threshold)

    # convert detected onsets to a list of timestamps
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

# 3. Missing utility functions
def segmentFrame2interval(segment_frame, frame_size=0.5):
    """Convert frame-level segments to interval format"""
    segment_frame = np.array(segment_frame)
    segment_frame[0] = 1
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval

def frame2interval(segment_frame, label_frame, frame_size=0.5):
    """Convert frame-level boundaries and labels to interval format"""
    segment_frame = np.array(segment_frame)
    label_frame = np.array(label_frame)
    segment_frame[0] = 1
    label = label_frame[segment_frame == 1]
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval, label

# 4. Missing mask functions
def get_spectral_mask(n_batch, seq_len, n_head, n_fct=1, n_mel=80, n_chroma=12):
    """Generate spectral attention mask"""
    # Within only
    mel_mask = tf.concat(
        [tf.ones([n_mel, n_mel], dtype=tf.bool), tf.zeros([n_mel, n_chroma], dtype=tf.bool)],
        axis=1,
    ) # [80, 80+12]
    chroma_mask = tf.concat(
        [tf.zeros([n_chroma, n_mel], dtype=tf.bool), tf.ones([n_chroma, n_chroma], dtype=tf.bool)],
        axis=1,
    ) # [12, 80+12]
    mask = tf.concat([mel_mask, chroma_mask], axis=0) # [80+12, 80+12]
    mask = tf.tile(mask[tf.newaxis, :, :], [n_head, 1, 1]) # [h, 80+12, 80+12]

    # FCT see all
    mask = tf.pad(mask, [(0, 0), (0, 0), (n_fct, 0)], constant_values=False) # [h, 80+12, 1+80+12]
    mask = tf.pad(mask, [(0, 0), (n_fct, 0), (0, 0)], constant_values=True) # [h, 1+80+12, 1+80+12]
    mask = tf.tile(mask[tf.newaxis, :, :, :], [n_batch*seq_len, 1, 1, 1]) # [bn, h, 1+80+12, 1+80+12]
    mask = tf.concat(tf.split(mask, n_head, axis=1), axis=0) # [hbn, 1, 1+80+12, 1+80+12]
    return tf.squeeze(mask, axis=1) # [hbn, 1+80+12, 1+80+12]

def get_temporal_mask(valid_len, max_len, n_heads=8):
    """Generate temporal attention mask"""
    def partition_len(l, div):
        assert div < l
        return [l // div + (1 if x < l % div else 0) for x in range(div)]

    valid_len = tf.cast(valid_len, tf.float32)
    b_seq = []
    for l in valid_len:
        h_seq = []
        for i_h in range(n_heads):
            div = i_h // 2 + 1
            ids = tf.concat([tf.ones([part], dtype=tf.int32)*i for i, part in enumerate(partition_len(l, div))], axis=0)
            ids = tf.pad(ids, [(0, max_len - l)], constant_values=-1) # [n]
            h_seq.append(ids)
        b_seq.append(tf.stack(h_seq)) # [h, n]
    b_seq = tf.stack(b_seq) # [b, h, n]
    mask = (b_seq[:, :, :, tf.newaxis] == b_seq[:, :, tf.newaxis, :]) # [b, h, n, n]
    mask = tf.concat(tf.split(mask, n_heads, axis=1), axis=0) # [hb, 1, n, n]
    return tf.squeeze(mask, axis=1) # [hb, n, n]

def load_dataset_config(config_path="/Scratch/repository/msa/MSATSUNGPING/dataset_beatles_salami_splits.json"):
    """Load the dataset selection configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_enhanced_datasets(config_path="/Scratch/repository/msa/MSATSUNGPING/dataset_beatles_salami_splits.json", 
                           data_base_path="/Scratch/repository/msa/MSATSUNGPING/"):  # UPDATE THIS PATH TO YOUR DATA DIRECTORY
    """
    Create train/test datasets following Claude's Beatles-centric strategy
    
    Args:
        config_path: Path to your fixed_dataset_selection.json file
        data_base_path: Parent directory containing your 6 preprocessed folders:
                       - beatles-original-preprocessed-data
                       - beatles-aug-preprocessed-data  
                       - salami-original-preprocessed-data
                       - salami-aug-preprocessed-data
                       - harmonix-original-preprocessed-data
                       - harmonix-aug-preprocessed-data
    """
    print("🎯 Creating datasets following Beatles-centric strategy...")
    
    # Load configuration
    config = load_dataset_config(config_path)
    
    # Data paths - UPDATE THESE IF YOUR FOLDER NAMES ARE DIFFERENT
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
    
    # Verify paths exist
    print("🔍 Verifying data paths...")
    for dataset_name, paths in dataset_paths.items():
        for data_type, path in paths.items():
            if os.path.exists(path):
                print(f"   ✅ {dataset_name}-{data_type}: {path}")
            else:
                print(f"   ❌ {dataset_name}-{data_type}: {path} (NOT FOUND)")
    
    # Create training data (original + augmented)
    train_data = create_train_data(config['training_set'], dataset_paths)
    
    # Create test data (original only)
    test_data = create_test_data(config['test_set'], dataset_paths)
    
    # Save datasets
    print("💾 Saving enhanced datasets...")
    os.makedirs('./vocals_high_quality', exist_ok=True)
    
    np.savez_compressed('./beatles_salami/train_data.npz', **train_data)
    np.savez_compressed('./beatles_salami/test_data.npz', **test_data)
    
    print(f"✅ Enhanced datasets saved!")
    print(f"📊 Training samples: {len(train_data['spec'])}")
    print(f"📊 Test samples: {len(test_data['spec'])}")
    
    return train_data, test_data

def create_train_data(training_config, dataset_paths):
    """Create training dataset with original + augmented data"""
    all_data = {
        'spec': [],
        'chromagram': [],
        'vocal_spec': [],
        'vocal_chromagram': [],
        'boundary': [],
        'function': [],
        'len': [],
        'section': []
    }
    
    for dataset_name, info in training_config.items():
        if dataset_name == 'summary':
            continue
            
        print(f"📁 Loading {dataset_name} training data...")
        song_ids = info['song_ids']
        
        # Load original data
        dataset_data = load_dataset_files(
            dataset_name, song_ids, dataset_paths[dataset_name]['original'], 
            include_augmented=False
        )
        
        # Load augmented data
        aug_data = load_dataset_files(
            dataset_name, song_ids, dataset_paths[dataset_name]['aug'], 
            include_augmented=True
        )
        
        # Combine original + augmented
        for key in all_data.keys():
            all_data[key].extend(dataset_data[key])
            all_data[key].extend(aug_data[key])
        
        print(f"   ✅ {dataset_name}: {len(dataset_data[key])} original + {len(aug_data[key])} augmented")
    
    # Convert to numpy arrays
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key], dtype=object)
    
    return all_data

def create_test_data(test_config, dataset_paths):
    """Create test dataset with original data only"""
    all_data = {
        'spec': [],
        'chromagram': [],
        'vocal_spec': [],
        'vocal_chromagram': [],
        'boundary': [],
        'function': [],
        'len': [],
        'section': []
    }
    
    for dataset_name, info in test_config.items():
        print(f"📁 Loading {dataset_name} test data...")
        song_ids = info['song_ids']
        
        # Load original data only for testing
        dataset_data = load_dataset_files(
            dataset_name, song_ids, dataset_paths[dataset_name]['original'], 
            include_augmented=False
        )
        
        for key in all_data.keys():
            all_data[key].extend(dataset_data[key])
        
        print(f"   ✅ {dataset_name}: {len(dataset_data['spec'])} samples")
    
    # Convert to numpy arrays
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key], dtype=object)
    
    return all_data

def load_dataset_files(dataset_name, song_ids, data_path, include_augmented=False):
    """Load files for a specific dataset"""
    data = {
        'spec': [],
        'chromagram': [],
        'vocal_spec': [],
        'vocal_chromagram': [],
        'boundary': [],
        'function': [],
        'len': [],
        'section': []
    }
    
    successful_loads = 0
    
    for song_id in song_ids:
        try:
            if include_augmented:
                # Find all augmented files for this song
                # Pattern matches: song_id + pitch/preemph variations + _a1_spec.npy
                pattern = f"{song_id}*_a1_spec.npy"
                spec_files = glob.glob(os.path.join(data_path, pattern))
                # Exclude original files (they have '_original_' in the name)
                spec_files = [f for f in spec_files if '_original_' not in f]
            else:
                # Load original file only
                spec_files = [os.path.join(data_path, f"{song_id}_original_a1_spec.npy")]
            
            for spec_file in spec_files:
                if not os.path.exists(spec_file):
                    continue
                    
                # Extract base name for other files
                base_name = spec_file.replace('_spec.npy', '')
                
                # Load all required files
                try:
                    spec = np.load(spec_file)
                    chromagram = np.load(base_name + '_chroma.npy')
                    vocal_spec = np.load(base_name + '_bassspec.npy')
                    vocal_chromagram = np.load(base_name + '_basschroma.npy')
                    boundary = np.load(base_name + '_boundary.npy')
                    function = np.load(base_name + '_function.npy')
                    section = np.load(base_name + '_section.npy')
                    
                    # Calculate length
                    valid_len = spec.shape[0]
                    
                    # Verify shapes
                    if (chromagram.shape[0] != valid_len or 
                        vocal_spec.shape[0] != valid_len or
                        vocal_chromagram.shape[0] != valid_len or
                        boundary.shape[0] != valid_len or 
                        function.shape[0] != valid_len):
                        print(f"⚠️ Shape mismatch in {os.path.basename(spec_file)}, skipping")
                        continue
                    
                    # Add to data
                    data['spec'].append(spec)
                    data['chromagram'].append(chromagram)
                    data['vocal_spec'].append(vocal_spec)
                    data['vocal_chromagram'].append(vocal_chromagram)
                    data['boundary'].append(boundary)
                    data['function'].append(function)
                    data['len'].append(valid_len)
                    data['section'].append(f"{dataset_name}_{song_id}")
                    
                    successful_loads += 1
                    
                except Exception as e:
                    print(f"❌ Error loading {os.path.basename(spec_file)}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing {song_id}: {e}")
            continue
    
    print(f"   📊 Successfully loaded {successful_loads} files from {dataset_name}")
    return data

def shape_list(input_tensor):
    """Return list of dims, statically where possible."""
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

# [Include all your architecture classes here - they remain unchanged]
# This includes: Norm, BatchNorm, DuoConv2D, SE_block, SinusoidalPositionalEncoding, 
# DrumEncoder, CrossAttentionBlock, CAMHSA, FeedForward, Attention, 
# CNNBase2D, ResBlock2D, ChromaCNNBase2D, ChromaCNN2D, SpecTNT_Enhanced, 
# FunctionalSegmentModel, etc.

class Norm(tf.keras.layers.Layer):
    def __init__(self, axes=[1, 2], adaptive=False):
        super().__init__()
        self.axes = axes
        self.rank = None
        self.adpative = adaptive

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.rank = len(input_shape)

        if not self.adpative:
            self.gamma = self.add_weight(name='gamma',
                                         shape=[d],
                                         initializer=tf.keras.initializers.Ones,
                                         trainable=True)

            self.beta = self.add_weight(name='beta',
                                        shape=[d],
                                        initializer=tf.keras.initializers.Zeros,
                                        trainable=True)

    def call(self, inputs, valid_len=None, epsilon=1e-7):
        if 1 in self.axes:
            mask = tf.sequence_mask(valid_len, maxlen=shape_list(inputs)[1], dtype=tf.float32)
            mask = mask[:, :, tf.newaxis] if self.rank == 3 else mask[:, :, tf.newaxis, tf.newaxis]
            mean, variance = tf.nn.weighted_moments(inputs, axes=self.axes, frequency_weights=mask, keepdims=True)
        else:
            mean, variance = tf.nn.moments(inputs, axes=self.axes, keepdims=True)

        normalized = (inputs - mean) * tf.math.rsqrt(variance + epsilon)

        if self.adpative:
            C = 1
            k = 0.1
            adapter = C * (1 - k * tf.stop_gradient(normalized))
            return adapter * normalized
        else:
            return self.gamma * normalized + self.beta

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, axes=[0, 1, 2], momentum=0.99, epsilon=0.001):
        super().__init__()
        self.axes = axes
        self.rank = None
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.rank = len(input_shape)

        self.gamma = self.add_weight(name='gamma',
                                     shape=[d],
                                     initializer=tf.keras.initializers.Ones,
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=[d],
                                    initializer=tf.keras.initializers.Zeros,
                                    trainable=True)

        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=[d],
                                           initializer=tf.keras.initializers.Zeros,
                                           trainable=False)

        self.moving_var = self.add_weight(name='moving_var',
                                           shape=[d],
                                           initializer=tf.keras.initializers.Ones,
                                           trainable=False)

    def call(self, inputs, valid_len, training=False):
        mask = tf.sequence_mask(valid_len, maxlen=shape_list(inputs)[1], dtype=tf.float32)
        mask = mask[:, :, tf.newaxis, tf.newaxis]
        batch_mean, batch_var = tf.nn.weighted_moments(inputs, axes=self.axes, frequency_weights=mask, keepdims=False)

        update_mean = tf.cond(
            training,
            lambda: self.moving_mean * self.momentum + batch_mean * (1 - self.momentum),
            lambda: self.moving_mean
        )
        update_var = tf.cond(
            training,
            lambda: self.moving_var * self.momentum + batch_var * (1 - self.momentum),
            lambda: self.moving_var
        )
        self.moving_mean.assign(update_mean)
        self.moving_var.assign(update_var)

        mean, var = tf.cond(
            training,
            lambda: (batch_mean, batch_var),
            lambda: (self.moving_mean, self.moving_var)
        )
        normalized = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

# [Continue with all other architecture classes from your original code...]
# I'm including the key ones here but you should copy all the architecture classes
# from your my-training-attempt2.py file
class DuoConv2D(tf.keras.layers.Layer):
    def __init__(self, n_units, kernel_size, padding='same'):
        super().__init__()
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_dense = tf.keras.layers.Dense(n_units)

    def build(self, input_shape):
        depth_multiplier = int(self.n_units // input_shape[-1])
        self.conv0 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=(1, 1),
            dilation_rate=(1, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )
        self.conv1 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            dilation_rate=(1, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )
        self.conv2 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            dilation_rate=(2, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )

    def call(self, x):
        '''x = [b, n, f, c]'''
        if self.padding == 'same':
            enc0 = self.conv0(x)
            enc1 = self.conv1(x)
            enc2 = self.conv2(x)
        elif self.padding == 'valid':
            n_pad = self.kernel_size[0] // 2
            enc1 = self.conv1(tf.pad(x, [(0,0), (n_pad,n_pad), (0,0), (0,0)]))
            enc2 = self.conv2(tf.pad(x, [(0,0), (2*n_pad,2*n_pad), (0,0), (0,0)]))
        return self.out_dense(enc0 + enc1 + enc2)


class SE_block(tf.keras.layers.Layer):
    '''Squeeze and excitation block'''
    def __init__(self, alpha=0.5, activation_func='relu', axis=[1]):
        super().__init__()

        self.alpha = alpha
        self.activation_func = activation_func
        self.axis = axis
        self.inner = None
        self.outer = None

    def build(self, input_shape):
        self.inner = tf.keras.layers.Dense(int(input_shape[-1] * self.alpha), activation=self.activation_func)
        self.outer = tf.keras.layers.Dense(int(input_shape[-1]), activation=tf.sigmoid)
        self.shape = input_shape

    def call(self, input, valid_len):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(input)[1], dtype=tf.float32) # [b, n]
        seq_mask = seq_mask[:, :, tf.newaxis] if int(tf.rank(input)) == 3 else seq_mask[:, :, tf.newaxis, tf.newaxis]
        valid_len = valid_len[:,  tf.newaxis, tf.newaxis] if int(tf.rank(input)) == 3 else valid_len[:, tf.newaxis, tf.newaxis, tf.newaxis]

        if self.axis == [1, 2]:
            gap = tf.reduce_sum(input*seq_mask, axis=self.axis, keepdims=True) / (tf.cast(valid_len, tf.float32) * self.shape[2]) # [b, c]
        elif self.axis == [1]:
            gap = tf.reduce_sum(input*seq_mask, axis=self.axis, keepdims=True) / tf.cast(valid_len, tf.float32) # [b, (f), c]
        elif self.axis == [2]:
            gap = tf.reduce_mean(input, axis=self.axis, keepdims=True) # [b, n, c]
        else:
            print('invalid axes.')
            exit(1)
        scale = self.outer(self.inner(gap))
        return scale * input


class SpecCNN(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=128,
        dropout_rate=0,
        activation_func='relu',
        kernel_size=(5, 5),
        is_ssm=False,
        freq_collapse=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.layers.Activation(activation_func)
        self.kernel_size = kernel_size
        self.is_ssm = is_ssm
        self.freq_collapse = freq_collapse

        alpha = 0.5
        self.conv1 = DuoConv2D(n_units//4, kernel_size=kernel_size)
        self.conv2 = DuoConv2D(n_units//2, kernel_size=kernel_size)
        self.conv3 = DuoConv2D(n_units, kernel_size=kernel_size)
        self.se1 = SE_block(alpha=alpha, axis=[2])
        self.se2 = SE_block(alpha=alpha, axis=[2])
        self.se3 = SE_block(alpha=alpha, axis=[2])

        self.norm1 = Norm(axes=[1], adaptive=False)
        self.norm2 = Norm(axes=[1], adaptive=False)
        self.norm3 = Norm(axes=[1], adaptive=False)

        self.reduce_dense = tf.keras.layers.Dense(1, name='reduce_dense')
        self.out_dense = tf.keras.layers.Dense(n_units, name='out_dense')
        self.out_norm_t = Norm(axes=[1], adaptive=False)
        self.out_norm_c = Norm(axes=[-1], adaptive=False)
        self.out_se = SE_block(alpha=alpha, axis=[1])

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def shuffle_channel(self, x, groups):
        b, n, d = shape_list(x)
        output = tf.reshape(x, [b, n, groups, d // groups])
        output = tf.transpose(output, [0, 1, 3, 2])
        return tf.reshape(output, [b, n, d])

    def call(self, input, valid_len):
        # input = [b, n, f, c]

        enc1 = self.conv1(input) # [b, n, f, d]
        enc1 = self.norm1(enc1, valid_len)
        enc1 = self.activation(enc1)
        enc1 = self.se1(enc1, valid_len)

        enc2 = self.conv2(enc1) # [b, n, f, d]
        enc2 = self.norm2(enc2, valid_len)
        enc2 = self.activation(enc2)
        enc2 = self.se2(enc2, valid_len)
        enc2 = self.dropout(enc2)

        enc3 = self.conv3(enc2) # [b, n, f, d]
        enc3 = self.norm3(enc3, valid_len)
        enc3 = self.activation(enc3)
        enc3 = self.se3(enc3, valid_len)

        if not self.freq_collapse:
            return enc3

        if not self.is_ssm:
            # Summarize the frequency dimension
            enc_max = tf.reduce_max(enc3, axis=2) # [b, n, d]
            enc_dense = tf.squeeze(self.reduce_dense(tf.transpose(enc3, [0, 1, 3, 2])), axis=-1) # [b, n, d]
            output = enc_max + enc_dense # [b, n, d]
            output = self.out_dense(output) # [b, n, d]
            output = self.out_norm_t(output, valid_len)
            b, n, d = shape_list(output)
            output = tf.reshape(output, [b, n, 20, d//20])
            output = self.out_norm_c(output, valid_len)
            output = tf.reshape(output, [b, n, d])
            output = self.activation(output)
            output = self.out_se(output, valid_len)
            output = self.dropout(output)
        else:
            output = tf.reduce_max(enc3, axis=2) # [b, n, d]
        return output


class SelfCNN(tf.keras.layers.Layer):
    """Multi-head attention keras layer wrapper"""
    def __init__(
        self,
        n_units=32,
        dropout_rate=0,
        activation_func='relu',
        kernel_size=(7, 7),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        self.activation = tf.keras.activations.deserialize(self.activation_func)

        self.conv1 = tf.keras.layers.Conv2D(2*n_units, kernel_size=kernel_size, activation=None, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(n_units, kernel_size=kernel_size, activation=None, padding='same')

        self.norm1 = Norm(axes=[1,2], adaptive=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, valid_len):
        '''input has shape = [b, n, n, d]'''
        output = self.conv1(input) # [b, n, n, d]
        output = self.norm1(output, valid_len)
        output = self.activation(output)
        output = self.conv2(output)
        return output


# === NEW: Positional Encoding ===
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=935, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


# === NEW: Drum Encoder ===
# REPLACE your entire DrumEncoder class with this:
class UniversalStemEncoder(tf.keras.layers.Layer):
    def __init__(self, mel_dim=80, chroma_dim=12, d_model=256, num_heads=8, 
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.mel_dim = mel_dim
        self.chroma_dim = chroma_dim
        self.d_model = d_model
        
        # Separate projections for mel and chroma
        self.mel_projection = tf.keras.layers.Dense(d_model // 2, name='stem_mel_proj')
        self.chroma_projection = tf.keras.layers.Dense(d_model // 2, name='stem_chroma_proj')
        
        # Input normalization
        self.input_norm = tf.keras.layers.LayerNormalization(name='stem_input_norm')
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, name='stem_pos_enc')
        
        # SIMPLIFIED: Only self-attention (NO feed forward layers)
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='stem_self_attn'
        )
        self.norm = tf.keras.layers.LayerNormalization(name='stem_norm')
        self.dropout = tf.keras.layers.Dropout(dropout, name='stem_dropout')
        
    def call(self, stem_mel, stem_chroma, mask=None, training=None):
        # Project mel and chroma features
        mel_proj = self.mel_projection(stem_mel)      # (B, T, d_model//2)
        chroma_proj = self.chroma_projection(stem_chroma)  # (B, T, d_model//2)
        
        # Concatenate to form full feature vector
        x = tf.concat([mel_proj, chroma_proj], axis=-1)  # (B, T, d_model)
        
        # Add normalization and positional encoding
        x = self.input_norm(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # ONLY self-attention (professor's suggestion: remove feed forward)
        attn_output = self.self_attention(x, x, attention_mask=mask, training=training)
        x = self.norm(x + attn_output)
        
        return x


# === NEW: Cross-Attention Block ===
class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='cross_attn'
        )
        
        self.norm1 = tf.keras.layers.LayerNormalization(name='cross_norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(name='cross_norm2')
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model)
        ], name='cross_ffn')
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        # Cross-attention with drums
        tgt_norm = self.norm1(tgt)
        tgt2, attn_weights = self.cross_attn(
            tgt_norm, memory, attention_mask=memory_mask, 
            return_attention_scores=True, training=training
        )
        tgt = tgt + self.dropout(tgt2, training=training)
        
        # FFN
        tgt_norm = self.norm2(tgt)
        tgt2 = self.ffn(tgt_norm, training=training)
        tgt = tgt + self.dropout(tgt2, training=training)
        
        return tgt, attn_weights


class CAMHSA(tf.keras.layers.Layer):
    """Convolution-Augmented Multi-Head Self-Attention"""
    def __init__(
        self,
        n_units=32,
        n_heads=8,
        max_len=540,
        attn_dropout_rate=0,
        cnn_dropout_rate=0,
        activation_func=None,
        self_mask=False,
        shared_pos=False,
        return_maps=False,
        attn_mask=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads = n_heads
        self.max_len = max_len
        self.attn_dropout_rate = attn_dropout_rate
        self.cnn_dropout_rate = cnn_dropout_rate
        self.activation_func = activation_func
        self.self_mask = self_mask
        self.shared_pos = shared_pos
        self.return_maps = return_maps
        self.attn_mask = attn_mask
        self.pos_clip = 60

        self.proj_q = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_q')
        self.proj_k = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_k')
        self.proj_v = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_v')
        self.proj_h = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_h')
        self.attn_dropout = tf.keras.layers.Dropout(attn_dropout_rate)
        self.cnn_dropout = tf.keras.layers.Dropout(cnn_dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

        self.selfcnn = SelfCNN(
            n_units=n_heads,
            dropout_rate=cnn_dropout_rate,
            activation_func='relu',
            kernel_size=(5, 5),
        )

    def build(self, input_shape):
        if not self.shared_pos:
            # pos embedding
            self.pos_k = self.add_weight(name='pos_k',
                                         shape=[2 * self.pos_clip + 1, self.n_units // self.n_heads],
                                         initializer=tf.random_uniform_initializer,
                                         trainable=True)

    def call(self, query, valid_len, attn_mask=None):
        # Query has shape [b, n, d]
        b, n, d_in = shape_list(query)
        seq_mask = tf.sequence_mask(valid_len, maxlen=n, dtype=tf.bool) # [b, n]

        # Relative position encodings
        max_len = shape_list(query)[1]
        '''Self-Attention with Relative Position Representations (NAACL-HLT 2018)'''
        rel_pos_idx = tf.range(max_len)[tf.newaxis, :] - tf.range(max_len)[:, tf.newaxis] # [n, n]
        rel_pos_idx = tf.clip_by_value(rel_pos_idx, -self.pos_clip, self.pos_clip)
        rel_pos_idx += self.pos_clip
        pos_enc_k = tf.nn.embedding_lookup(self.pos_k, rel_pos_idx) # [n, n, d/h]

        # Projection
        q_emb = self.proj_q(query) # [b, n, d]
        k_emb = self.proj_k(query) # [b, n, d]
        v_emb = self.proj_v(query) # [b, n, d]

        # Head splitting
        q_emb = tf.concat(tf.split(q_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        k_emb = tf.concat(tf.split(k_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        v_emb = tf.concat(tf.split(v_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]

        # Attention computation
        QK = tf.matmul(q_emb, k_emb, transpose_b=True) # [hb, n, n]

        QR_K = tf.matmul(tf.transpose(q_emb, [1,0,2]), pos_enc_k, transpose_b=True) # [n, hb, n]
        QR_K = tf.transpose(QR_K, [1,0,2]) # [hb, n, n]
        attn_map = QK + QR_K # [hb, n, n]
        attn_map = attn_map / (shape_list(k_emb)[-1]**0.5) # [hb, n, n]

        # Convolution on attention maps
        attn_map = tf.stack(tf.split(attn_map, self.n_heads, 0), -1) # [b, n, n, h]
        attn_map = self.selfcnn(attn_map, valid_len) # [b, n, n, h]
        attn_map = tf.concat(tf.split(attn_map, self.n_heads, -1), 0) # [hb, n, n, 1]
        attn_map = tf.squeeze(attn_map, -1) # [hb, n, n]

        # Attention masking
        valid_mask = tf.tile(seq_mask[:, tf.newaxis, :], [self.n_heads, n, 1]) # [hb, n, n]
        if attn_mask is not None:
            valid_mask = tf.logical_and(valid_mask, attn_mask)

        attn_map = tf.where(valid_mask, attn_map, -1e12) # [hb, n, n]

        # Activation
        attn_map = tf.nn.softmax(attn_map) # [hb, n, n]

        # Combinatorial representation
        output = tf.matmul(attn_map, v_emb) # [hb, n, d/h]
        output = tf.concat(tf.split(output, self.n_heads, 0), -1) # [b, n, d]
        output = self.proj_h(output) # [b, n, d]
        output = self.attn_dropout(output)
        output += query # residual connection

        if self.return_maps:
            return self.layer_norm(output), attn_map
        else:
            return self.layer_norm(output)


class FeedForward(tf.keras.layers.Layer):
    """Feedfoward layer of the transformer model.
    Paramters
    ---------
    n_units: list[int, int]
        A two-element integer list. The first integer represents the output embedding size
        of the first convolution layer, and the second integer represents the embedding size
        of the second convolution layer.
    activation_func: str
        Activation function of the first covolution layer. Available options can be found
        from the tensorflow.keras official site.
    dropout_rate: float
        Dropout rate of all dropout layers.
    """
    def __init__(
            self,
            n_units=[1024, 256],
            activation_func='relu',
            dropout_rate=0,
            output_norm=True,
            residual=True,
            kernel_size=None,
    ):
        super().__init__()

        self.n_units = n_units
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.output_norm = output_norm
        self.kernel_size = kernel_size

        self.inner = tf.keras.layers.Dense(n_units[0], activation=activation_func)
        self.outer = tf.keras.layers.Dense(n_units[1], activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

    def call(self, input):
        output = self.inner(input)
        output = self.outer(output)
        output = self.dropout(output)
        if self.residual:
            output += input # residual connection
        return self.layer_norm(output) if self.output_norm else output


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=128,
        n_heads=4,
        max_len=540,
        activation_func=None,
        dropout_rate=0,
        return_maps=False,
        with_pos_enc=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads = n_heads
        self.max_len = max_len
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.pos_clip = max_len - 1
        self.return_maps = return_maps
        self.with_pos_enc = with_pos_enc

        self.proj_q = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_q')
        self.proj_k = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_k')
        self.proj_v = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_v')
        self.proj_h = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_h')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

    def build(self, input_shape):
        if self.with_pos_enc:
            # pos embedding
            self.pos_k = self.add_weight(name='pos_k',
                                         shape=[2 * self.pos_clip + 1, self.n_units // self.n_heads],
                                         initializer=tf.random_uniform_initializer,
                                         trainable=True)

    def call(self, query, key=None, value=None, valid_len=None, attn_mask=None):
        # query = [b, n, d_in]

        if key is None:
            key = query
        if value is None:
            value = key

        b, n, d_in = shape_list(query)

        # Projection
        q_emb = self.proj_q(query) # [b, n, d]
        k_emb = self.proj_k(key) # [b, n, d]
        v_emb = self.proj_v(value) # [b, n, d]

        # Head splitting
        q_emb = tf.concat(tf.split(q_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        k_emb = tf.concat(tf.split(k_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        v_emb = tf.concat(tf.split(v_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]

        # Attention computation
        QK = tf.matmul(q_emb, k_emb, transpose_b=True) # [hb, n, n]
        if self.with_pos_enc:
            # Relative position encodings
            max_len = shape_list(query)[1]
            '''Self-Attention with Relative Position Representations (NAACL-HLT 2018)'''
            rel_pos_idx = tf.range(max_len)[tf.newaxis, :] - tf.range(max_len)[:, tf.newaxis] # [n, n]
            rel_pos_idx = tf.clip_by_value(rel_pos_idx, -self.pos_clip, self.pos_clip)
            rel_pos_idx += self.pos_clip
            pos_enc_k = tf.nn.embedding_lookup(self.pos_k, rel_pos_idx) # [n, n, d/h]

            QR_K = tf.matmul(tf.transpose(q_emb, [1, 0, 2]), pos_enc_k, transpose_b=True)  # [n, hb, n]
            QR_K = tf.transpose(QR_K, [1, 0, 2])  # [hb, n, n]

            attn_map = QK + QR_K # [hb, n, n]
        else:
            attn_map = QK

        attn_map = attn_map / (shape_list(k_emb)[-1]**0.5) # [hb, n, n]

        if valid_len is not None:
            # Attention masking
            seq_mask = tf.sequence_mask(valid_len, maxlen=n, dtype=tf.bool) # [b, n]
            valid_mask = tf.tile(seq_mask[:, tf.newaxis, :], [self.n_heads, n, 1]) # [hb, n, n]
            attn_map = tf.where(valid_mask, attn_map, -1e12) # [hb, n, n]

        if attn_mask is not None:
            attn_map = tf.where(attn_mask, attn_map, -1e12) # [hb, n, n]

        # Activation
        attn_map = tf.nn.softmax(attn_map) # [hb, n, n]

        # Combinatorial representation
        output = tf.matmul(attn_map, v_emb) # [hb, n, d/h]
        output = tf.concat(tf.split(output, self.n_heads, 0), -1) # [b, n, d]
        output = self.proj_h(output) # [b, n, d]
        output = self.dropout(output)
        output += query # residual connection

        if self.return_maps:
            return self.layer_norm(output), attn_map
        else:
            return self.layer_norm(output)


class CNNBase2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units=[40, 80],
            activation_func='relu',
            kernel_size=(3, 3),
            dropout_rate=0,
            padding='same',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.Conv2D(n_units[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(n_units[1], kernel_size=kernel_size, padding=padding)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.half_t = kernel_size[0] // 2
        self.padding = padding

    def call(self, x, valid_len, training=False):
        '''x = [b, n, f, d]'''
        if self.padding == 'valid':
            x = tf.pad(x, [(0,0), (self.half_t, self.half_t), (0,0), (0,0)], constant_values=0)
        enc = self.conv1(x) # [b, n, f, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, f, d1]
        enc = self.activation(enc) # [b, n, f, d1]
        enc = self.dropout(enc) # [b, n, f, d1]

        if self.padding == 'valid':
            enc = tf.pad(enc, [(0,0), (self.half_t, self.half_t), (0,0), (0,0)], constant_values=0)
        enc = self.conv2(enc) # [b, n, f, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, f, d2]
        enc = self.activation(enc) # [b, n, f, d2]
        enc = self.dropout(enc) # [b, n, f, d2]
        return enc


class ResBlock2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units,
            activation_func='relu',
            kernel_size=(3, 3),
            dropout_rate=0,
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', depth_multiplier=1)
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', depth_multiplier=1)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len, training=False):
        '''x = [b, n, f, d]'''
        enc = self.conv1(x) # [b, n, f, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, f, d1]
        enc = self.activation(enc) # [b, n, f, d1]
        enc = self.dropout(enc) # [b, n, f, d1]

        enc = self.conv2(enc) # [b, n, f, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, f, d2]
        enc = self.activation(enc) # [b, n, f, d2]
        enc = self.dropout(enc) # [b, n, f, d2]
        return x + enc # [b, n, f, d2]


class ChromaCNNBase2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units=[40, 80],
            activation_func='relu',
            kernel_size=(5, 12),
            dropout_rate=0,
            padding='valid',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.Conv2D(n_units[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(n_units[1], kernel_size=kernel_size, padding=padding)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.half_w = kernel_size[0] // 2
        self.padding = padding

    def call(self, x, valid_len, training=False):
        '''x = [b, n, 12, d]'''
        # Chroma expansion
        x_extend = tf.concat([x, x[:, :, :11, :]], axis=2) # [b, n, 23, d]
        x_extend = tf.pad(x_extend, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        # 1st conv
        enc = self.conv1(x_extend) # [b, n, 12, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, 12, d1]
        enc = self.activation(enc) # [b, n, 12, d1]
        enc = self.dropout(enc) # [b, n, 12, d1]

        # Chroma expansion
        enc = tf.concat([enc, enc[:, :, :11, :]], axis=2) # [b, n, 23, d]
        enc = tf.pad(enc, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        # 2nd conv
        enc = self.conv2(enc) # [b, n, 12, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, 12, d2]
        enc = self.activation(enc) # [b, n, 12, d2]
        enc = self.dropout(enc) # [b, n, 12, d2]
        return enc


class ChromaCNN2D(tf.keras.layers.Layer):
    def __init__(
            self,
            activation_func='relu',
            dropout_rate=0,
            kernel_size=[5, 12],
            padding='valid',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.half_w = kernel_size[0] // 2
        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, depth_multiplier=1)
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, depth_multiplier=1)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len, training=False):
        '''x = [b, n, 12, d]'''

        # Chroma expansion
        x_extend = tf.concat([x, x[:, :, :11, :]], axis=2) # [b, n, 23, d]
        x_extend = tf.pad(x_extend, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        enc = self.conv1(x_extend) # [b, n, 12, d]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, 12, d]
        enc = self.activation(enc) # [b, n, 12, d]
        enc = self.dropout(enc) # [b, n, 12, d]

        # Chroma expansion
        enc = tf.concat([enc, enc[:, :, :11, :]], axis=2) # [b, n, 23, d]
        enc = tf.pad(enc, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        enc = self.conv2(enc) # [b, n, 12, d]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, 12, d]
        enc = self.activation(enc) # [b, n, 12, d]
        enc = self.dropout(enc) # [b, n, 12, d]
        return x + enc # [b, n, 12, d]


# === MODIFIED: SpecTNT with Cross-Attention ===
# REPLACE your RhythmGuidanceModule with this:
class UniversalCrossModalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Universal feature analysis (works for any stem type)
        self.feature_analyzer = tf.keras.layers.Conv1D(
            filters=num_heads, 
            kernel_size=7, 
            padding='same',
            name='universal_feature_analyzer'
        )
        
        # Change detection (works for any audio feature)

        self.change_detector = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 5, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 3, padding='same', activation='sigmoid')  # Fixed
        ], name='universal_change_detector')

        # # Guided self-attention
        # self.guided_attention = tf.keras.layers.MultiHeadAttention(
        #     num_heads=num_heads,
        #     key_dim=d_model // num_heads,
        #     dropout=dropout,
        #     name='universal_guided_attention'
        # )

        # In EnhancedCrossModalFusion.__init__:
        self.cross_modal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='cross_modal_attention'
        )
        
        self.norm = tf.keras.layers.LayerNormalization(name='universal_norm')
        
    def call(self, harmonic_features, stem_features, training=None):
        # Analyze stem features for patterns (universal for all stem types)
        feature_patterns = self.feature_analyzer(stem_features)  # [b, n, num_heads]
        
        # Detect changes in stem (universal change detection)
        change_strength = self.change_detector(stem_features)  # [b, n, 1]
        change_strength = tf.squeeze(change_strength, axis=-1)  # [b, n]
        
        # Apply stem-guided attention to harmonic content
        attended = self.guided_attention(
            harmonic_features, 
            harmonic_features, 
            training=training
        )
        
        # Modulate with stem guidance
        b, n, d = tf.shape(harmonic_features)[0], tf.shape(harmonic_features)[1], tf.shape(harmonic_features)[2]
        num_heads = tf.shape(feature_patterns)[2]
        head_dim = d // num_heads
        
        # Reshape feature patterns to match feature dimensions
        pattern_bias_expanded = tf.expand_dims(feature_patterns, axis=-1)  # [b, n, heads, 1]
        pattern_bias_tiled = tf.tile(pattern_bias_expanded, [1, 1, 1, head_dim])
        pattern_bias_reshaped = tf.reshape(pattern_bias_tiled, [b, n, d])  # [b, n, d]
        
        # Apply change-based modulation
        change_bias = tf.expand_dims(change_strength, axis=-1)  # [b, n, 1]
        
        # Universal modulation formula
        modulation = 1.0 + 0.2 * tf.nn.tanh(pattern_bias_reshaped) * change_bias
        guided_output = attended * modulation
        
        return self.norm(harmonic_features + guided_output), change_strength

class EnhancedCrossModalFusion(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        
        # Stem-aware feature analysis
        self.stem_feature_analyzer = tf.keras.layers.Conv1D(
            filters=num_heads, 
            kernel_size=7, 
            padding='same',
            name='stem_feature_analyzer'
        )
        
        # Adaptive fusion weights based on stem characteristics
        self.adaptive_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model // 2, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model, activation='sigmoid')
        ], name='adaptive_gate')
        
        # Enhanced change detection with multi-scale analysis
        self.change_detector = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(32, 7, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 5, padding='same', activation='sigmoid')
        ], name='enhanced_change_detector')
        
        # Guided attention with residual connection
        # self.guided_attention = tf.keras.layers.MultiHeadAttention(
        #     num_heads=num_heads,
        #     key_dim=d_model // num_heads,
        #     dropout=dropout,
        #     name='enhanced_guided_attention'
        # )

        # In EnhancedCrossModalFusion.__init__:
        self.cross_modal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='cross_modal_attention'
        )
        
        # Fusion normalization and projection
        self.fusion_norm = tf.keras.layers.LayerNormalization(name='fusion_norm')
        self.output_projection = tf.keras.layers.Dense(d_model, name='output_projection')
        
    def call(self, harmonic_features, stem_features, training=None):
        """
        harmonic_features: [b, n, d] - main song features
        stem_features: [b, n, d] - any stem type features
        """
        # Analyze stem characteristics
        stem_patterns = self.stem_feature_analyzer(stem_features)  # [b, n, num_heads]
        
        # Detect temporal changes in stem
        change_strength = self.change_detector(stem_features)  # [b, n, 1]
        change_strength = tf.squeeze(change_strength, axis=-1)  # [b, n]
        
        # Compute adaptive fusion weights based on stem content
        stem_summary = tf.reduce_mean(stem_features, axis=1, keepdims=True)  # [b, 1, d]
        adaptive_weights = self.adaptive_gate(stem_summary)  # [b, 1, d]
        adaptive_weights = tf.tile(adaptive_weights, [1, tf.shape(harmonic_features)[1], 1])  # [b, n, d]
        
        # Apply stem-guided attention to harmonic content
        # attended_harmonic = self.guided_attention(
        #     harmonic_features, 
        #     harmonic_features, 
        #     training=training
        # )

        cross_attended = self.cross_modal_attention(
            query=harmonic_features,      # F_temporal (from temporal attention)
            key=stem_features,            # F_stem_enc (from stem encoder)
            value=stem_features,          # F_stem_enc (from stem encoder)
            training=training
        )
        
        # Enhanced modulation with adaptive weights
        b, n, d = tf.shape(harmonic_features)[0], tf.shape(harmonic_features)[1], tf.shape(harmonic_features)[2]
        num_heads = tf.shape(stem_patterns)[2]
        head_dim = d // num_heads
        
        # Reshape stem patterns to match feature dimensions
        pattern_bias_expanded = tf.expand_dims(stem_patterns, axis=-1)  # [b, n, heads, 1]
        pattern_bias_tiled = tf.tile(pattern_bias_expanded, [1, 1, 1, head_dim])
        pattern_bias_reshaped = tf.reshape(pattern_bias_tiled, [b, n, d])  # [b, n, d]
        
        # Apply change-based and adaptive modulation
        change_bias = tf.expand_dims(change_strength, axis=-1)  # [b, n, 1]
        
        # Enhanced modulation formula
        stem_guidance = 0.3 * tf.nn.tanh(pattern_bias_reshaped) * change_bias
        adaptive_modulation = adaptive_weights * stem_guidance
        
        # guided_output = attended_harmonic * (1.0 + adaptive_modulation)

        fusion_strength = 1.0  # Default, or pass as parameter
        # guided_output = attended_harmonic * (1.0 + fusion_strength * adaptive_modulation)
        # Replace line 1387 with:
        # guided_output = harmonic_features * (1.0 + fusion_strength * adaptive_modulation)
        # Then apply modulation
        guided_output = cross_attended * (1.0 + fusion_strength * adaptive_modulation)
        
        

        # Residual connection and final projection
        fused_output = harmonic_features + guided_output
        fused_output = self.fusion_norm(fused_output)
        final_output = self.output_projection(fused_output)
        
        return final_output, change_strength

# Enhanced SpecTNT with both self and cross attention
class SpecTNT_Enhanced(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=80,
        n_heads_f=4,
        n_heads_t=8,
        max_len=540,
        attn_dropout_rate=0,
        cnn_dropout_rate=0,
        activation_func=None,
        shared_pos=False,
        return_maps=False,
        n_fct=1,
        n_mel=80,
        n_chroma=12,
        use_boundary_fusion=True,  # ← ADD THIS LINE
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads_f = n_heads_f
        self.n_heads_t = n_heads_t
        self.max_len = max_len
        self.n_fct = n_fct
        self.n_mel = n_mel
        self.n_chroma = n_chroma
        self.attn_dropout_rate = attn_dropout_rate
        self.cnn_dropout_rate = cnn_dropout_rate
        self.activation_func = activation_func
        self.shared_pos = shared_pos
        self.return_maps = return_maps
        self.use_boundary_fusion = use_boundary_fusion  # ← ADD THIS LINE

        # Spectral attention (unchanged)
        self.attn_f = Attention(
            n_units=n_units//2,
            n_heads=n_heads_f,
            max_len=max_len,
            dropout_rate=attn_dropout_rate,
            return_maps=return_maps,
            with_pos_enc=False,
        )
        self.ffn_f = FeedForward(
            n_units=[4 * (n_units//2), n_units//2],
            dropout_rate=attn_dropout_rate,
        )
        self.f2t = tf.keras.layers.Dense(n_units)

        # ENHANCED TEMPORAL PROCESSING: Both self and cross attention
        # 1. Self-attention for temporal context
        self.self_attn_t = CAMHSA(
            n_units=n_units,
            n_heads=n_heads_t,
            max_len=max_len,
            attn_dropout_rate=attn_dropout_rate,
            cnn_dropout_rate=cnn_dropout_rate,
            return_maps=return_maps,
        )
        

        self.enhanced_cross_modal = EnhancedCrossModalFusion(
            d_model=n_units,
            num_heads=n_heads_t,
            dropout=attn_dropout_rate,
        )
                
        # 3. Fusion mechanism
        self.fusion_gate = tf.keras.layers.Dense(n_units, activation='sigmoid', name='fusion_gate')
        self.fusion_norm = tf.keras.layers.LayerNormalization(name='fusion_norm')
        
        # 4. Combined FFN
        self.ffn_t = FeedForward(
            n_units=[4 * n_units, n_units],
            dropout_rate=attn_dropout_rate,
        )
        self.t2f = tf.keras.layers.Dense(n_units//2)

        # Print configuration for debugging
        if use_boundary_fusion:
            print("🟢 SpecTNT: Full Model (Cross-Modal + Boundary-Aware)")
        else:
            print("🔵 SpecTNT: Cross-Modal Only (No Boundary-Aware)")

    def call(self, S, FCT, stem_features, valid_len, training=None):
        # S = [b, n, f, d/2], FCT = [b, n, 1, d], drum_features = [b, n, d_drum]
        b, n, f, half_d = shape_list(S)

        # Spectral attention (unchanged)
        attn_mask_f = get_spectral_mask(
            n_batch=b, seq_len=n, n_head=self.n_heads_f,
            n_fct=self.n_fct, n_mel=self.n_mel, n_chroma=self.n_chroma
        )

        # Temporal attention mask
        attn_mask_t = get_temporal_mask(valid_len, max_len=n, n_heads=self.n_heads_t)

        # Concat and reshape for spectral processing
        enc_FCT = self.t2f(FCT)  # FCT = [b, n, 1, d/2]
        enc = tf.concat([enc_FCT, S], axis=2)  # [b, n, 1+f, d/2]
        enc = tf.reshape(enc, [b*n, 1+f, half_d])  # [b*n, 1+f, d/2]

        # Spectral Attention
        if self.return_maps:
            enc, map_S = self.attn_f(enc, valid_len=None, attn_mask=attn_mask_f)
        else:
            enc = self.attn_f(enc, valid_len=None, attn_mask=attn_mask_f)
        enc = self.ffn_f(enc)
        enc = tf.reshape(enc, [b, n, 1+f, half_d])

        # Split
        enc_FCT = enc[:, :, 0, :]  # [b, n, d/2]
        enc_S = enc[:, :, 1:, :]   # [b, n, f, d/2]

        # ENHANCED TEMPORAL PROCESSING
        enc_FCT = self.f2t(enc_FCT)  # [b, n, d]

        # 1. Self-attention for temporal context
        if self.return_maps:
            enc_self, map_T_self = self.self_attn_t(enc_FCT, valid_len=valid_len, attn_mask=attn_mask_t)
        else:
            enc_self = self.self_attn_t(enc_FCT, valid_len=valid_len, attn_mask=attn_mask_t)

        
        # enc_guided, stem_boundaries = self.universal_cross_modal(enc_self, stem_features, training=training)
        enc_guided, stem_boundaries = self.enhanced_cross_modal(enc_self, stem_features, training=training)


        # NEW CODE with conditional logic:
        if self.use_boundary_fusion:
            # V3: Full model with boundary-aware gating
            boundary_strength = tf.expand_dims(stem_boundaries, axis=-1)
            enc_fused = boundary_strength * enc_guided + (1 - boundary_strength) * enc_self
            enc_fused = self.fusion_norm(enc_fused)
        else:
            # V2: Cross-modal only, skip boundary-aware gating
            enc_fused = self.fusion_norm(enc_guided)

        # 4. Final processing
        enc_FCT = self.ffn_t(enc_fused)  # [b, n, d]
        enc_FCT = tf.expand_dims(enc_FCT, axis=2)  # [b, n, 1, d]

        # TO:
        if self.return_maps:
            return enc_S, enc_FCT, map_S, stem_boundaries
        else:
            return enc_S, enc_FCT, stem_boundaries

        






def class_conversion(i, reduced=True):
    if i == 4: return 'X'
    for k, v in function_dict.items():
        if v == i:
            return (k[0].upper()) if reduced else k.capitalize()


def format_cluster_sequence(seq):
    out = ''
    for c in seq:
        tag = class_conversion(c)
        sty = str(c%7 + 41)
        format = ';'.join([sty])
        out += '\x1b[%sm%s\x1b[0m' % (format, tag)
    return out


def print_temp(temp, sample=0, print_len=190):
    # Check if temp data exists
    if not temp['b_ref'] or len(temp['b_ref']) == 0:
        print("⚠️  No temp data available for printing")
        return
    
    # Ensure sample index is valid
    if sample >= len(temp['b_ref']):
        sample = 0
        print(f"⚠️  Sample index {sample} out of range, using sample 0")
    
    just_len = 15
    b_ref, b_est, matched = temp['b_ref'][sample], temp['b_est'][sample], temp['matched'][sample]
    n_b_ref, n_b_est, n_matched = temp['n_b_ref'][sample], temp['n_b_est'][sample], temp['n_matched'][sample]

    print('n_b_ref %d n_b_est %d n_matched %d' % (n_b_ref, n_b_est, n_matched))
    b_ref_in_second = temp['b_ref_in_second'][sample]
    b_est_in_second = temp['b_est_in_second'][sample]
    print('b_ref_in_second', ' '.join(["{:.2f}".format(s) for s in b_ref_in_second]))
    print('b_est_in_second', ' '.join(["{:.2f}".format(s) for s in b_est_in_second]))

    max_len = b_ref.shape[0]
    n_split = int(np.ceil(max_len/print_len))

    f_ref, f_est = temp['f_ref'][sample], temp['f_est'][sample]

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
        "PURPLE": "\033[95m",
        "CYAN": "\033[96m",
        "DARKCYAN": "\033[36m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
        "END": "\033[0m"}
    return color_dict[color]


# Enhanced main training function

# === COMPLETE TRAINING FUNCTION - REPLACE YOUR INCOMPLETE train() FUNCTION ===

# === MODIFICATION 2: Add Early Stopping Class (Add this before your train() function) ===

class ImprovedEarlyStopping:
    def __init__(self, patience=50, min_delta=0.001, restore_best=True):
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
                    print(f"🔄 Restoring best weights (F1: {self.best_score:.3f})")
                    model.set_weights(self.best_weights)
                return True
        return False
def train_v2_cross_modal_only():
    """
    V2: Cross-Modal Attention Only (No Boundary-Aware Gating)
    """
    print("🔵 TRAINING V2: Cross-Modal Attention Only")
    
    DATA_BASE_PATH = "/Scratch/repository/msa/MSATSUNGPING/"
    
    # Create enhanced datasets
    train_data, test_data = create_enhanced_datasets(
        config_path="/Scratch/repository/msa/MSATSUNGPING/dataset_beatles_salami_splits.json",
        data_base_path=DATA_BASE_PATH
    )
    
    # Generator for enhanced data
    def generator(data):
        for spec, chromagram, vocal_spec, vocal_chromagram, valid_len, boundary, function, section in \
                zip(data['spec'], data['chromagram'], data['vocal_spec'], data['vocal_chromagram'],
                    data['len'], data['boundary'], data['function'], data['section']):
            
            # Ensure valid_len is a scalar integer
            if hasattr(valid_len, 'shape') and len(valid_len.shape) > 0:
                valid_len_scalar = int(valid_len[0]) if len(valid_len) > 0 else int(valid_len.item())
            else:
                valid_len_scalar = int(valid_len)
            
            # Ensure section is a scalar string
            if hasattr(section, 'shape') and len(section.shape) > 0:
                section_str = str(section[0]) if len(section) > 0 else "unknown"
            else:
                section_str = str(section)
            
            yield spec, chromagram, vocal_spec, vocal_chromagram, valid_len_scalar, boundary, function, section_str

    # Create output signature
    output_signature = (
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.string),
    )

    tf_train_data = tf.data.Dataset.from_generator(
        lambda: generator(train_data),
        output_signature=output_signature
    )

    tf_test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_data),
        output_signature=output_signature
    )

    # === CREATE ENHANCED MODEL ===
    model = FunctionalSegmentModel(
        max_len=935,
        n_units=80,
        n_heads=8,
        n_layers=2,
        cnn_dropout_rate=0.5,
        attn_dropout_rate=0.5,
        use_boundary_fusion=False,  # ← V2: NO BOUNDARY-AWARE GATING
    )

    # Build the model
    print("🔄 Building enhanced model...")
    dummy_spec = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_drum_spec = tf.zeros((1, 100, 80))
    dummy_drum_chroma = tf.zeros((1, 100, 12))
    dummy_len = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_drum_spec, dummy_drum_chroma, dummy_len, training=False)
    print("✅ Enhanced model built successfully!")

    # ADD THIS CODE HERE:
    print("\n" + "="*80)
    print("MODEL SUMMARY:")
    print("="*80)

    # Now the model is built, so we can count parameters
    try:
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")
    except Exception as e:
        print(f"Could not count parameters: {e}")
        # Alternative: manually count from trainable_variables
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        print(f"Trainable parameters: {trainable_params:,}")

    print("="*80 + "\n")

    # =========================================================================
    # CHECKPOINT SETUP - FIXED: Use manual saving only
    # =========================================================================
    checkpoint = tf.train.Checkpoint(model=model)
    model_path = './bass_without_boundary_aware_300'
    os.makedirs(f'{model_path}/all_epochs', exist_ok=True)
    os.makedirs(f'{model_path}/best_models', exist_ok=True)

    # Initialize test prediction saver
    test_pred_saver = BestTestPredictionSaver(
        save_dir=model_path,
        n_best=5  # Save top 5 predictions
    )
    
    # INITIALIZE tracking variables
    all_epoch_results = []  # For CSV
    best_test_F1 = -1.0     # Track best F1 score
    best_test_epoch = 0
    best_test_result = {}

    # Training parameters
    TRAIN_BATCH_SIZE = 6
    TEST_BATCH_SIZE = 6
    TRAIN_SHUFFLE_SIZE = len(train_data['spec'])
    N_EPOCHS = 300

    model.steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE_SIZE / TRAIN_BATCH_SIZE))
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE_SIZE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE)
    tf_test_data = tf_test_data.padded_batch(TEST_BATCH_SIZE)

    # Training metrics tracking
    best_train_epoch = 0
    supervised_metrics = ['F1_seg']
    best_train_result = {k: 0 for k in supervised_metrics}
    
    # Early Stopping
    early_stopping = ImprovedEarlyStopping(patience=50, min_delta=0.001, restore_best=True)
    print("🚀 Starting enhanced training loop...")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(1, N_EPOCHS+1):
        print(f'🔄 Epoch {epoch}/{N_EPOCHS}')
        print(color_text("RED") + "--training phase--" + color_text("END"))
        
        # Training phase
        for i_batch, batch in enumerate(tf_train_data):
            model.train_step(batch)
        
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
        
        # Calculate train F1
        try:
            train_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing training metric {e}, setting train_F1 to 0.0")
            train_F1 = 0.0
            
        if train_F1 > sum([float(best_train_result.get(k, 0)) for k in supervised_metrics]):
            best_train_epoch, best_train_result = epoch, result

        # Print training results
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))

        # Testing phase
        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        for i_batch, batch in enumerate(tf_test_data):
            model.test_step(batch)
        
        test_data_size = len(test_data['spec'])
        if test_data_size > 0:
            safe_sample = np.random.randint(min(TEST_BATCH_SIZE, test_data_size))
        else:
            safe_sample = 0
            
        print_temp(model.temp, sample=safe_sample)
        result = model.average_result()
        
        # Add missing metrics
        if 'F1_seg' not in result:
            print("⚠️  F1_seg not found in test results, setting to 0.0")
            result['F1_seg'] = tf.constant(0.0)
        if 'P_seg' not in result:
            result['P_seg'] = tf.constant(0.0)
        if 'R_seg' not in result:
            result['R_seg'] = tf.constant(0.0)
            
        print_confusion_matrix(model.confusion_matrix_test_max.numpy(), model.confusion_matrix_test_boun.numpy())
        model.clear_result()
        
        # Calculate test F1
        try:
            test_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing test metric {e}, setting test_F1 to 0.0")
            test_F1 = 0.0
        
        # =====================================================================
        # SAVE CHECKPOINTS - Using manual save with actual epoch numbers
        # =====================================================================
        
        # Save every epoch with ACTUAL epoch number
        try:
            save_path = checkpoint.save(f'{model_path}/all_epochs/epoch-{epoch}')
            print(f"💾 Saved epoch {epoch}: {save_path}")
        except Exception as e:
            print(f'❌ Epoch saving failed: {e}')
        
        # Save results to list for CSV
        epoch_result = {'epoch': epoch, 'phase': 'test'}
        for k, v in result.items():
            try:
                epoch_result[k] = float(v.numpy())
            except:
                epoch_result[k] = float(v)
        all_epoch_results.append(epoch_result)
        
        # Save CSV after each epoch
        try:
            results_df = pd.DataFrame(all_epoch_results)
            results_df.to_csv(f'{model_path}/training_results.csv', index=False)
            print(f"📊 Results saved to {model_path}/training_results.csv")
        except Exception as e:
            print(f'⚠️  CSV save failed: {e}')
        

        if test_F1 > best_test_F1:
            best_test_F1 = test_F1
            best_test_epoch = epoch
            best_test_result = result
            
            print(color_text("YELLOW") + 
                f"🏆 NEW BEST MODEL at epoch {epoch} with F1_seg={test_F1:.3f}!" + 
                color_text("END"))
            
            try:
                # Save model checkpoint
                best_save_path = checkpoint.save(
                    f'{model_path}/best_models/best-epoch-{epoch}'
                )
                print(f"✅ Best model saved: {best_save_path}")
                
                # Save best model info to JSON
                best_info = {
                    'epoch': epoch,
                    'F1_seg': test_F1,
                    'all_metrics': {k: float(v.numpy()) for k, v in result.items()}
                }
                with open(f'{model_path}/best_models/best_info.json', 'w') as f:
                    json.dump(best_info, f, indent=2)
                print(f"📝 Best info saved to best_info.json")
                
                # ✨ NEW: Save top 5 test predictions
                test_pred_saver.save_test_predictions_simple(
                    model=model,
                    test_dataset=tf_test_data,
                    test_data=test_data,  # Original test data with track IDs
                    epoch=epoch,
                    test_metrics={k: float(v.numpy()) for k, v in result.items()}
                )
                    
            except Exception as e:
                print(f'❌ Best model/predictions saving failed: {e}')
                import traceback
                traceback.print_exc()
        
        
        # Print test results
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))
        
        # Print current best
        print(color_text("CYAN") + 
              f'### Best test F1_seg at epoch {best_test_epoch}: {best_test_F1:.3f}' + 
              color_text("END"))

        # Check early stopping
        if early_stopping(test_F1, model, epoch):
            print(color_text("YELLOW") + f"🛑 Early stopping triggered at epoch {epoch}" + color_text("END"))
            print(f"📊 Best F1 score: {early_stopping.best_score:.3f}")
            break
        
        print()

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    
    if early_stopping.stopped_epoch > 0:
        print(f'🛑 Training stopped early at epoch {early_stopping.stopped_epoch}')
        print(f'📊 Final best F1: {early_stopping.best_score:.3f}')
    
    print(f"\n🏆 Best model: epoch {best_test_epoch} with F1_seg={best_test_F1:.3f}")
    print(f"📂 Checkpoints saved in: {model_path}/")
    print(f"📊 Results CSV: {model_path}/training_results.csv")
    print(f"📝 Best model info: {model_path}/best_models/best_info.json")
    print("\n" + "="*80)

def train_v3_full_model():
    """
    V3: Full Model (Cross-Modal Attention + Boundary-Aware Gating)
    """
    print("🟢 TRAINING V3: Full Model")
    
    DATA_BASE_PATH = "/Scratch/repository/msa/MSATSUNGPING/"
    
    # Create enhanced datasets
    train_data, test_data = create_enhanced_datasets(
        config_path="/Scratch/repository/msa/MSATSUNGPING/dataset_beatles_salami_splits.json",
        data_base_path=DATA_BASE_PATH
    )
    
    # Generator for enhanced data
    def generator(data):
        for spec, chromagram, vocal_spec, vocal_chromagram, valid_len, boundary, function, section in \
                zip(data['spec'], data['chromagram'], data['vocal_spec'], data['vocal_chromagram'],
                    data['len'], data['boundary'], data['function'], data['section']):
            
            # Ensure valid_len is a scalar integer
            if hasattr(valid_len, 'shape') and len(valid_len.shape) > 0:
                valid_len_scalar = int(valid_len[0]) if len(valid_len) > 0 else int(valid_len.item())
            else:
                valid_len_scalar = int(valid_len)
            
            # Ensure section is a scalar string
            if hasattr(section, 'shape') and len(section.shape) > 0:
                section_str = str(section[0]) if len(section) > 0 else "unknown"
            else:
                section_str = str(section)
            
            yield spec, chromagram, vocal_spec, vocal_chromagram, valid_len_scalar, boundary, function, section_str

    # Create output signature
    output_signature = (
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 80], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 12], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.string),
    )

    tf_train_data = tf.data.Dataset.from_generator(
        lambda: generator(train_data),
        output_signature=output_signature
    )

    tf_test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_data),
        output_signature=output_signature
    )

    # === CREATE ENHANCED MODEL ===
    model = FunctionalSegmentModel(
        max_len=935,
        n_units=80,
        n_heads=8,
        n_layers=2,
        cnn_dropout_rate=0.5,
        attn_dropout_rate=0.5,
        use_boundary_fusion=True,  # ← V3: WITH BOUNDARY-AWARE GATING
    )

    # Build the model
    print("🔄 Building enhanced model...")
    dummy_spec = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_drum_spec = tf.zeros((1, 100, 80))
    dummy_drum_chroma = tf.zeros((1, 100, 12))
    dummy_len = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_drum_spec, dummy_drum_chroma, dummy_len, training=False)
    print("✅ Enhanced model built successfully!")

    # ADD THIS CODE HERE:
    print("\n" + "="*80)
    print("MODEL SUMMARY:")
    print("="*80)

    # Now the model is built, so we can count parameters
    try:
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")
    except Exception as e:
        print(f"Could not count parameters: {e}")
        # Alternative: manually count from trainable_variables
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        print(f"Trainable parameters: {trainable_params:,}")

    print("="*80 + "\n")

    # =========================================================================
    # CHECKPOINT SETUP - FIXED: Use manual saving only
    # =========================================================================
    checkpoint = tf.train.Checkpoint(model=model)
    model_path = './vocals_with_full_model'
    os.makedirs(f'{model_path}/all_epochs', exist_ok=True)
    os.makedirs(f'{model_path}/best_models', exist_ok=True)

    # Initialize test prediction saver
    test_pred_saver = BestTestPredictionSaver(
        save_dir=model_path,
        n_best=5  # Save top 5 predictions
    )
    
    # INITIALIZE tracking variables
    all_epoch_results = []  # For CSV
    best_test_F1 = -1.0     # Track best F1 score
    best_test_epoch = 0
    best_test_result = {}

    # Training parameters
    TRAIN_BATCH_SIZE = 6
    TEST_BATCH_SIZE = 6
    TRAIN_SHUFFLE_SIZE = len(train_data['spec'])
    N_EPOCHS = 300

    model.steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE_SIZE / TRAIN_BATCH_SIZE))
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE_SIZE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE)
    tf_test_data = tf_test_data.padded_batch(TEST_BATCH_SIZE)

    # Training metrics tracking
    best_train_epoch = 0
    supervised_metrics = ['F1_seg']
    best_train_result = {k: 0 for k in supervised_metrics}
    
    # Early Stopping
    early_stopping = ImprovedEarlyStopping(patience=50, min_delta=0.001, restore_best=True)
    print("🚀 Starting enhanced training loop...")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(1, N_EPOCHS+1):
        print(f'🔄 Epoch {epoch}/{N_EPOCHS}')
        print(color_text("RED") + "--training phase--" + color_text("END"))
        
        # Training phase
        for i_batch, batch in enumerate(tf_train_data):
            model.train_step(batch)
        
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
        
        # Calculate train F1
        try:
            train_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing training metric {e}, setting train_F1 to 0.0")
            train_F1 = 0.0
            
        if train_F1 > sum([float(best_train_result.get(k, 0)) for k in supervised_metrics]):
            best_train_epoch, best_train_result = epoch, result

        # Print training results
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))

        # Testing phase
        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        for i_batch, batch in enumerate(tf_test_data):
            model.test_step(batch)
        
        test_data_size = len(test_data['spec'])
        if test_data_size > 0:
            safe_sample = np.random.randint(min(TEST_BATCH_SIZE, test_data_size))
        else:
            safe_sample = 0
            
        print_temp(model.temp, sample=safe_sample)
        result = model.average_result()
        
        # Add missing metrics
        if 'F1_seg' not in result:
            print("⚠️  F1_seg not found in test results, setting to 0.0")
            result['F1_seg'] = tf.constant(0.0)
        if 'P_seg' not in result:
            result['P_seg'] = tf.constant(0.0)
        if 'R_seg' not in result:
            result['R_seg'] = tf.constant(0.0)
            
        print_confusion_matrix(model.confusion_matrix_test_max.numpy(), model.confusion_matrix_test_boun.numpy())
        model.clear_result()
        
        # Calculate test F1
        try:
            test_F1 = sum([float(result[k]) for k in supervised_metrics])
        except KeyError as e:
            print(f"⚠️  Missing test metric {e}, setting test_F1 to 0.0")
            test_F1 = 0.0
        
        # =====================================================================
        # SAVE CHECKPOINTS - Using manual save with actual epoch numbers
        # =====================================================================
        
        # Save every epoch with ACTUAL epoch number
        try:
            save_path = checkpoint.save(f'{model_path}/all_epochs/epoch-{epoch}')
            print(f"💾 Saved epoch {epoch}: {save_path}")
        except Exception as e:
            print(f'❌ Epoch saving failed: {e}')
        
        # Save results to list for CSV
        epoch_result = {'epoch': epoch, 'phase': 'test'}
        for k, v in result.items():
            try:
                epoch_result[k] = float(v.numpy())
            except:
                epoch_result[k] = float(v)
        all_epoch_results.append(epoch_result)
        
        # Save CSV after each epoch
        try:
            results_df = pd.DataFrame(all_epoch_results)
            results_df.to_csv(f'{model_path}/training_results.csv', index=False)
            print(f"📊 Results saved to {model_path}/training_results.csv")
        except Exception as e:
            print(f'⚠️  CSV save failed: {e}')
        
        if test_F1 > best_test_F1:
            best_test_F1 = test_F1
            best_test_epoch = epoch
            best_test_result = result
            
            print(color_text("YELLOW") + 
                f"🏆 NEW BEST MODEL at epoch {epoch} with F1_seg={test_F1:.3f}!" + 
                color_text("END"))
            
            try:
                # Save model checkpoint
                best_save_path = checkpoint.save(
                    f'{model_path}/best_models/best-epoch-{epoch}'
                )
                print(f"✅ Best model saved: {best_save_path}")
                
                # Save best model info to JSON
                best_info = {
                    'epoch': epoch,
                    'F1_seg': test_F1,
                    'all_metrics': {k: float(v.numpy()) for k, v in result.items()}
                }
                with open(f'{model_path}/best_models/best_info.json', 'w') as f:
                    json.dump(best_info, f, indent=2)
                print(f"📝 Best info saved to best_info.json")
                
                # ✨ NEW: Save top 5 test predictions
                test_pred_saver.save_test_predictions_simple(
                    model=model,
                    test_dataset=tf_test_data,
                    test_data=test_data,  # Original test data with track IDs
                    epoch=epoch,
                    test_metrics={k: float(v.numpy()) for k, v in result.items()}
                )
                    
            except Exception as e:
                print(f'❌ Best model/predictions saving failed: {e}')
                import traceback
                traceback.print_exc()
        
        
        # Print test results
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))
        
        # Print current best
        print(color_text("CYAN") + 
              f'### Best test F1_seg at epoch {best_test_epoch}: {best_test_F1:.3f}' + 
              color_text("END"))

        # Check early stopping
        if early_stopping(test_F1, model, epoch):
            print(color_text("YELLOW") + f"🛑 Early stopping triggered at epoch {epoch}" + color_text("END"))
            print(f"📊 Best F1 score: {early_stopping.best_score:.3f}")
            break
        
        print()

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    
    if early_stopping.stopped_epoch > 0:
        print(f'🛑 Training stopped early at epoch {early_stopping.stopped_epoch}')
        print(f'📊 Final best F1: {early_stopping.best_score:.3f}')
    
    print(f"\n🏆 Best model: epoch {best_test_epoch} with F1_seg={best_test_F1:.3f}")
    print(f"📂 Checkpoints saved in: {model_path}/")
    print(f"📊 Results CSV: {model_path}/training_results.csv")
    print(f"📝 Best model info: {model_path}/best_models/best_info.json")
    print("\n" + "="*80)


    # **IMPORTANT**: You need to add your FunctionalSegmentModel class here
    # from your my-training-attempt2.py file
class FunctionalSegmentModel(tf.keras.Model):
    """Modified model with drum encoder and cross-attention"""
    def __init__(
        self,
        max_len=935,
        n_units=80,
        n_heads=8,
        cnn_dropout_rate=0,
        attn_dropout_rate=0,
        n_layers=2,
        steps_per_epoch=None,
        n_classes=7,
        return_maps=True,
        use_boundary_fusion=True,  # ← ADD THIS LINE
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_len = max_len
        self.n_units = n_units
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.cnn_dropout_rate = cnn_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.steps_per_epoch = steps_per_epoch
        self.n_classes = n_classes
        self.return_maps = return_maps
        self.optimizer = None  # Will be created in train_step
        self.flag = True
        self.confusion_matrix_train_max = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_max = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_boun = tf.zeros([n_classes, n_classes], tf.int32)
        self.use_boundary_fusion = use_boundary_fusion  # ← ADD THIS LINE

        self.w_b = 18
        self.w_f = 2

        # === NEW: Drum Encoder ==

        self.stem_encoder = UniversalStemEncoder(
            mel_dim=80,
            chroma_dim=12,
            d_model=n_units,
            num_heads=n_heads,
            dropout=attn_dropout_rate,
        )

        self.spec_prenorm = BatchNorm()

        # CNN feature extraction (unchanged)
        self.specCNNBase = CNNBase2D(
            n_units=[n_units//4, n_units//2],
            activation_func='relu',
            kernel_size=(7, 5),
            dropout_rate=cnn_dropout_rate,
        )
        self.specCNN = ResBlock2D(
            n_units=n_units//2,
            activation_func='relu',
            kernel_size=(7, 5),
            dropout_rate=cnn_dropout_rate,
        )
        self.chromaCNNBase = ChromaCNNBase2D(
            n_units=[n_units//4, n_units//2],
            activation_func='relu',
            kernel_size=(7, 12),
            dropout_rate=cnn_dropout_rate,
        )
        self.chromaCNN = ChromaCNN2D(
            activation_func='relu',
            dropout_rate=cnn_dropout_rate,
            kernel_size=(7, 12),
        )

        self.sepc_res_conv = tf.keras.layers.Conv2D(1, kernel_size=[5, 1], padding='same')
        self.chroma_res_conv = tf.keras.layers.Conv2D(1, kernel_size=[5, 1], padding='same')

        # CNN transition (unchanged)
        self.spec_transition = tf.keras.layers.Dense(n_units//2, name='spec_transition')
        self.spec_transition_norm = Norm(axes=[-1], adaptive=False)
        self.chroma_transition = tf.keras.layers.Dense(n_units//2, name='chroma_transition')
        self.chroma_transition_norm = Norm(axes=[-1], adaptive=False)

        self.fct_dense = tf.keras.layers.Dense(n_units, name='fct_dense')
        self.fct_dense_norm = Norm(axes=[-1], adaptive=False)

        # Spectro-temporal modeling (modified with cross-attention)
        self.specTNT_layers = [SpecTNT_Enhanced(
            n_units=n_units,
            max_len=max_len,
            cnn_dropout_rate=cnn_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            return_maps=return_maps,
            use_boundary_fusion=use_boundary_fusion,  # ← ADD THIS LINE
        ) for _ in range(n_layers)]

        # Output layers (unchanged)
        self.boun1 = tf.keras.layers.Dense(n_units, name='boun1')
        self.boun2 = tf.keras.layers.Dense(n_units, name='boun2')
        self.boun_out = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same')

        self.func1 = tf.keras.layers.Dense(n_units, name='func1')
        self.func2 = tf.keras.layers.Dense(n_units, name='func2')
        self.func_out = tf.keras.layers.Conv1D(self.n_classes, kernel_size=11, padding='same')

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.result = {k: [] for k in
            [
                'Acc_max',
                'Acc_smooth',
                'P_seg', 'R_seg', 'F1_seg',
                'P_seg3', 'R_seg3', 'F1_seg3',
                'P_pair', 'R_pair', 'F1_pair',
                'loss',
                'loss_b', 'loss_f',
            ]
        }
        self.temp = {k: [] for k in
            [
                'b_ref', 'b_est', 'matched', 'n_b_ref', 'n_b_est', 'n_matched', 'b_ref_in_second', 'b_est_in_second',
                'f_ref', 'f_est',
            ]
        }

        # Print model configuration
        print("\n" + "="*80)
        print("MODEL CONFIGURATION:")
        print("="*80)
        if use_boundary_fusion:
            print("📊 Variant: V3 - FULL MODEL (Cross-Modal + Boundary-Aware)")
        else:
            print("📊 Variant: V2 - CROSS-MODAL ONLY (No Boundary-Aware)")
        print("="*80 + "\n")

    def build(self, input_shape):
        # frequency positional embedding
        self.fpe_S = self.add_weight(name='fpe_S',
                                     shape=[1, 1, 92, self.n_units//2],
                                     initializer=tf.random_uniform_initializer,
                                     trainable=True)

        self.fpe_FCT = self.add_weight(name='fpe_FCT',
                                       shape=[1, 1, 1, self.n_units],
                                       initializer=tf.random_uniform_initializer,
                                       trainable=True)

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
                    # First, try to concatenate (works for arrays)
                    concatenated = tf.concat(v, axis=0)
                    result_dict[k] = tf.reduce_mean(concatenated)
                except tf.errors.InvalidArgumentError:
                    # If concat fails, it means we have scalars - use stack instead
                    stacked = tf.stack(v)
                    result_dict[k] = tf.reduce_mean(stacked)
        return result_dict

   
    def call(self, spec, chromagram, stem_spec, stem_chromagram, valid_len, training=False):
        '''
        === MODIFIED: Added drum inputs ===
        spec = [b, n, 80]
        chromagram = [b, n, 12]
        drum_spec = [b, n, 80]
        drum_chromagram = [b, n, 12]
        valid_len = [b]
        '''

        # Log compression (unchanged)
        spec = tf.math.log(1 + 100 * tf.nn.relu(spec + 80))
        spec = tf.expand_dims(spec, axis=-1) # [b, n, 80, 1]
        chromagram = tf.expand_dims(chromagram, axis=-1) # [b, n, 12, 1]

        # === NEW: Process drum features ===

        stem_spec = tf.math.log(1 + 100 * tf.nn.relu(stem_spec + 80))
        stem_features = self.stem_encoder(stem_spec, stem_chromagram, training=training)

        # Pre-Norm (unchanged)
        spec = self.spec_prenorm(spec, valid_len, training=training) # [b, n, 80, 1]

        with tf.name_scope("cnn") as scope_cnn:
            enc_spec = self.specCNNBase(spec, valid_len, training=training) # [b, n, 80, d/2]
            enc_spec = self.specCNN(enc_spec, valid_len, training=training) # [b, n, 80, d/2]
            enc_spec_res = self.sepc_res_conv(tf.transpose(enc_spec, [0, 1, 3, 2])) # [b, n, d/2, 1]
            enc_spec_res = tf.reduce_mean(enc_spec_res, axis=[2, 3]) # [b, n]
            enc_spec = self.spec_transition(enc_spec) # [b, n, 80, d/2]
            enc_spec = self.spec_transition_norm(enc_spec) # [b, n, 80, d/2]

            enc_chroma = self.chromaCNNBase(chromagram, valid_len, training=training) # [b, n, 12, d/2]
            enc_chroma = self.chromaCNN(enc_chroma, valid_len, training=training) # [b, n, 12, d/2]
            enc_chroma_res = self.chroma_res_conv(tf.transpose(enc_chroma, [0, 1, 3, 2])) # [b, n, d/2, 1]
            enc_chroma_res = tf.reduce_mean(enc_chroma_res, axis=[2, 3]) # [b, n]
            enc_chroma = self.chroma_transition(enc_chroma) # [b, n, 12, d/2]
            enc_chroma = self.chroma_transition_norm(enc_chroma) # [b, n, 12, d/2]

        with tf.name_scope("attention") as scope_attn:
            b, n, _, _ = shape_list(spec)
            enc_S = tf.concat([enc_spec, enc_chroma], axis=2) # [b, n, 92, d/2]
            enc_FCT = self.fct_dense(tf.reduce_mean(enc_S, axis=2, keepdims=True)) # [b, n, 1, d]
            enc_FCT = self.fct_dense_norm(enc_FCT)

            enc_S += self.fpe_S # [b, n, 92, d/2]
            enc_FCT += self.fpe_FCT # [b, n, 1, d]

            # WITH this:
            map_S = None
            stem_boundary_scores = None
            for l, specTNT in enumerate(self.specTNT_layers):
                if self.return_maps:
                    enc_S, enc_FCT, map_S, stem_boundaries = specTNT(enc_S, enc_FCT, stem_features, valid_len=valid_len,training=training)
                    if stem_boundary_scores is None:
                        stem_boundary_scores =stem_boundaries
                    else:
                        stem_boundary_scores += stem_boundaries
                else:
                    enc_S, enc_FCT, stem_boundaries = specTNT(enc_S, enc_FCT, stem_features, valid_len=valid_len, training=training)
                    if stem_boundary_scores is None:
                        stem_boundary_scores = stem_boundaries
                    else:
                        stem_boundary_scores += stem_boundaries
            enc = tf.squeeze(enc_FCT, axis=2) # [b, n, d]

        with tf.name_scope('boundary_estimation') as scope_boun:
            
            logits_boun = self.boun1(enc) # [b, n, d]
            logits_boun = self.boun2(logits_boun) # [b, n, d]
            logits_boun = tf.squeeze(self.boun_out(logits_boun), axis=2) # [b, n]
            # ADD THIS: Use drum boundary information to enhance boundary detection
            # You'll need to collect drum_boundaries from your SpecTNT layers
            logits_boun = logits_boun + enc_spec_res + enc_chroma_res # [b, n]

            if stem_boundary_scores is not None:
                logits_boun = logits_boun + 0.5 * stem_boundary_scores
            # logits_boun = logits_boun + enc_spec_res + enc_chroma_res # [b, n]

        with tf.name_scope("function_estimation") as scope_func:
            logits_func = self.func1(enc) # [b, n, d]
            logits_func = self.func2(logits_func) # [b, n, d]
            logits_func = self.func_out(logits_func) # [b, n, k]

        return logits_boun, logits_func, enc_S, map_S

    def train_step(self, data):
        # === MODIFIED: Unpack drum data ===
        # spec, chromagram, drum_spec, drum_chromagram, valid_len, boun_ref, func_ref, sec_ref = data
        spec, chromagram, stem_spec, stem_chromagram, valid_len, boun_ref, func_ref, sec_ref = data

        with tf.GradientTape() as tape:
            # === MODIFIED: Pass drum inputs to model ===
            # logits_boun, logits_func, _, _ = self.call(spec, chromagram, drum_spec, drum_chromagram, valid_len, training=True)
            logits_boun, logits_func, _, _ = self.call(spec, chromagram, stem_spec, stem_chromagram, valid_len, training=True)

            prob_boun = tf.nn.sigmoid(logits_boun) # [b, n]

            # Estimation (unchanged)
            boun_est = self.decode_boundary(prob_boun, valid_len) # [b, n]
            func_est_max = tf.argmax(logits_func, axis=-1, output_type=tf.int32) # [b, n]
            func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len) # [b, n]

            self.confusion_matrix_train_max += self.compute_confusion_matrix(
                func_ref,
                func_est_max,
            )
            self.confusion_matrix_train_boun += self.compute_confusion_matrix(
                func_ref,
                func_est_smooth,
            )

            # Losses (unchanged)
            ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
            ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)
            
            if self.flag:
                print('ce_b', ce_b.numpy())
                print('ce_f', ce_f.numpy())
                self.flag = False

            loss = ce_b + ce_f

            # Compute gradients
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)

        # NEW IMPROVED OPTIMIZER - FIXED VERSION:
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            # Create optimizer on first call
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-4,  # Start with warm-up rate
                clipnorm=1.0,
                epsilon=1e-7
            )
        
        current_epoch = tf.cast(self.optimizer.iterations // self.steps_per_epoch, tf.float32)
        # Progressive fusion strength (starts gentle, becomes stronger)
        fusion_strength = tf.minimum(1.0, current_epoch / 20.0)  # Reaches full strength by epoch 20
        
        def lr_schedule():
            return tf.cond(
                current_epoch < 5,
                lambda: 1e-4,  # Warm up
                lambda: tf.cond(
                    current_epoch < 100,
                    lambda: 1e-3,  # Main training
                    lambda: tf.cond(
                        current_epoch < 150,
                        lambda: 5e-4,  # Gradual decay
                        lambda: 1e-4   # Fine-tuning
                    )
                )
            )
        
        # Update learning rate
        self.optimizer.learning_rate.assign(lr_schedule())
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (unchanged)
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(
            self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth')
        )
        score_dict.update(
            self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size)
        )
        score_dict.update(
            self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size)
        )

        self.loss_tracker.update_state(loss)

        score_dict.update(
            {
                'loss': loss,
                'loss_b': ce_b,
                'loss_f': ce_f,
            }
        )
        [self.result[k].append(v) for k, v in score_dict.items()]
    
    def test_step(self, data):
        # === MODIFIED: Unpack drum data ===
        # spec, chromagram, drum_spec, drum_chromagram, valid_len, boun_ref, func_ref, sec_ref = data
        spec, chromagram, stem_spec, stem_chromagram, valid_len, boun_ref, func_ref, sec_ref = data

        # === MODIFIED: Pass drum inputs to model ===
        # logits_boun, logits_func, enc_S, map_S = self(spec, chromagram, drum_spec, drum_chromagram, valid_len, training=False)
        logits_boun, logits_func, enc_S, map_S = self(spec, chromagram, stem_spec, stem_chromagram, valid_len, training=False)
        prob_boun = tf.nn.sigmoid(logits_boun) # [b, n]

        # Estimation (unchanged)
        boun_est = self.decode_boundary(prob_boun, valid_len) # [b, n]
        func_est_max = tf.argmax(logits_func, axis=-1, output_type=tf.int32) # [b, n]
        func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len) # [b, n]

        self.confusion_matrix_test_max += self.compute_confusion_matrix(
            func_ref,
            func_est_max,
        )
        self.confusion_matrix_test_boun += self.compute_confusion_matrix(
            func_ref,
            func_est_smooth,
        )

        # Losses (unchanged)
        ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
        ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)

        loss = ce_b + ce_f

        # Update metrics (unchanged)
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(
            self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth')
        )
        score_dict.update(
            self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size)
        )
        score_dict.update(
            self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size)
        )

        self.loss_tracker.update_state(loss)

        score_dict.update(
            {
                'loss': loss,
                'loss_b': ce_b,
                'loss_f': ce_f,
            }
        )
        [self.result[k].append(v) for k, v in score_dict.items()]

        return boun_est, func_est_smooth

    # === ALL REMAINING METHODS UNCHANGED ===
    def bce_from_logits(self, gt, logits, valid_len, pos_weight=0.3):
        '''gt, logits = [b, n], valid_len = [b]'''
        gt_expaned = self.expand_boundary(gt, valid_len, value=0.5) # [b, n]
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(gt)[1], dtype=tf.float32) # [b, n]

        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_expaned, logits, pos_weight=pos_weight) # [b, n]

        # Mean over time
        loss = tf.reduce_sum(wbce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32) # [b]
        return tf.reduce_mean(loss)

    

    def cce_from_logits(self, gt, logits, valid_len):
        '''gt = [b, n], logits = [b, n, k], valid_len = [b]'''
        
        # 🔥 CRITICAL FIX: Use weights from YOUR actual data distribution
        weights = tf.constant([
            1.03,  # intro: 9.70% → slightly boost
            0.27,  # verse: 37.70% → heavily reduce (most common!)
            0.47,  # chorus: 21.29% → reduce
            0.94,  # bridge: 10.68% → slightly boost
            0.88,  # inst: 11.35% → slightly boost
            3.00,  # outro: 3.33% → heavily boost (rarest!)
            1.68,  # silence: 5.96% → boost
        ], tf.float32)
        
        seq_mask = tf.sequence_mask(valid_len, maxlen=tf.shape(gt)[1], dtype=tf.float32)
        gt_onehot = tf.one_hot(gt, depth=self.n_classes)
        
        # Weighted cross entropy
        ce = tf.nn.softmax_cross_entropy_with_logits(gt_onehot, logits)
        
        # Apply class weights
        class_weights_expanded = tf.gather(weights, gt)  # [b, n]
        weighted_ce = ce * class_weights_expanded
        
        # Mean over time
        loss = tf.reduce_sum(weighted_ce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32)
        return tf.reduce_mean(loss)

    
    def decode_boundary(self, prob_boun, valid_len, method='librosa'):
        '''prob_boun = [b, n]'''
        assert method in ['msaf', 'librosa', 'boeck']

        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(prob_boun)[1], dtype=tf.float32) # [b, n]
        prob_boun *= seq_mask # [b, n]

        prob_boun_numpy = prob_boun.numpy() # [b, n]
        peaks = np.zeros_like(prob_boun_numpy, dtype=np.int32) # [b, n]

        if method == 'msaf':
            peak_indices = [peak_picking_MSAF(seq, median_len=7, offset_rel=0.05, sigma=4) for seq in prob_boun_numpy]
        elif method == 'librosa':
            peak_indices = [
                librosa.util.peak_pick(seq, pre_max=10, post_max=10, pre_avg=20, post_avg=10, delta=0.03, wait=10) for seq in prob_boun_numpy
            ]
            peak_indices = [ids.astype(int) for ids in peak_indices]
        elif method == 'boeck':
            peak_indices = [
                peak_picking_boeck(seq, threshold=0.01, fps=2, combine=10, pre_max=10, post_max=10, pre_avg=20, post_avg=10) for seq in prob_boun_numpy
            ]

        for i in range(prob_boun_numpy.shape[0]):
            peaks[i, peak_indices[i]] = 1

        # Ensure each sequence begins with 1
        peaks[:, 0] = 1
        assert np.array_equal(prob_boun_numpy.shape, peaks.shape)
        return tf.constant(peaks, tf.int32) * tf.cast(seq_mask, tf.int32)

    def decode_labeling(self, boun_est, logits_func, valid_len):
        # Labeling based on the boundary prediction
        '''boun_est = [b, n], prob_func = [b, n, k]'''

        boun_est = boun_est.numpy()
        prob_func = tf.nn.sigmoid(logits_func).numpy()
        valid_len = valid_len.numpy()
        max_len = valid_len.max()
        func_est = []
        for i in range(valid_len.shape[0]):
            l = valid_len[i]
            b_i = np.where(np.equal(boun_est[i, :l], 1))[0]
            segments = [segment for segment in np.split(prob_func[i, :l], indices_or_sections=b_i) if len(segment)]
            centroids = np.stack([np.sum(segment, axis=0) for segment in segments]) # [n_segments, d]
            clusters = np.argmax(centroids, axis=-1) # [n_segments]
            label_frame = np.array([c for (segment, c) in zip(segments, clusters) for _ in range(len(segment))])

            if l < max_len:
                label_frame = np.pad(label_frame, (0, max_len-l), 'constant', constant_values=self.n_classes-1)
            func_est.append(label_frame)
        return tf.constant(func_est)

    def expand_boundary(self, boundary, valid_len, value=0.5, size=3):
        '''boundary = [b, n], valid_len = [b]'''
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boundary)[1], dtype=tf.float32)
        boundary = tf.cast(boundary, tf.float32) # [b, n]

        filter = tf.ones([size, 1, 1]) # [size, 1, 1]
        boundary_expanded = tf.nn.conv1d(boundary[:, :, tf.newaxis], filters=filter, stride=1, padding='SAME') # [b, n, 1]
        boundary_expanded = tf.squeeze(boundary_expanded, axis=-1) # [b, n]

        cond = tf.logical_and((boundary_expanded != boundary), tf.logical_not(tf.cast(boundary, tf.bool))) # [b, n]
        boundary_expanded = tf.where(cond, value, boundary) # [b, n]
        return boundary_expanded * seq_mask # [b, n]

    def compute_segment_score(self, boun_ref, boun_est, valid_len, resolution):
        # gt = [b, n], pred = [b, n], valid_len = [b]

        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boun_ref)[1], dtype=tf.int32) # [b, n]

        boun_ref_expanded = tf.cast(self.expand_boundary(boun_ref, valid_len, value=1), tf.int32) # [b, n]
        matched = boun_est * boun_ref_expanded * seq_mask # [b, n]

        n_boun_ref = tf.reduce_sum(boun_ref, axis=1) # [b]
        n_boun_est = tf.reduce_sum(boun_est, axis=1) # [b]
        n_matched = tf.reduce_sum(matched, axis=1) # [b]

        precision, recall, fscore = [], [], []
        precision3, recall3, fscore3 = [], [], []
        for i in range(shape_list(boun_ref)[0]):
            l = valid_len[i].numpy()
            b_ref = boun_ref[i, :l].numpy() # [n]
            b_est = boun_est[i, :l].numpy() # [n]

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
            'P_seg': tf.constant(precision),
            'R_seg': tf.constant(recall),
            'F1_seg': tf.constant(fscore),
            'P_seg3': tf.constant(precision3),
            'R_seg3': tf.constant(recall3),
            'F1_seg3': tf.constant(fscore3)
        }

    def compute_confusion_matrix(self, func_ref, func_est):
        return tf.math.confusion_matrix(
            labels=tf.reshape(func_ref, [-1]),
            predictions=tf.reshape(func_est, [-1]),
            num_classes=self.n_classes,
            weights=None,
            dtype=tf.dtypes.int32,
        ) # [k, k]

    def compute_pairwise_score(self, boun_ref, func_ref, boun_est, func_est, valid_len, resolution):
        # boun_ref, func_ref, boun_est, func_est = [b, n]
        precision_pair, recall_pair, fscore_pair = [], [], []

        for i in range(shape_list(func_ref)[0]):
            # Ground truth
            l = valid_len[i].numpy()
            f_ref = func_ref[i, :l].numpy() # [n]
            b_ref = boun_ref[i, :l].numpy() # [n]
            f_est = func_est[i, :l].numpy() # [n]
            assert f_ref.shape == f_est.shape
            b_est = boun_est[i, :l].numpy() # [n]

            # Convert frame-level into interval-level
            interval_ref, label_ref = frame2interval(b_ref, f_ref, frame_size=resolution)
            interval_est, label_est = frame2interval(b_est, f_est, frame_size=resolution)

            self.temp['f_ref'].append(f_ref)
            self.temp['f_est'].append(f_est)

            # Pairwise agreement
            P_pair, R_pair, F1_pair = mir_eval.segment.pairwise(
                interval_ref, label_ref, interval_est, label_est, frame_size=0.1
            )
            precision_pair.append(P_pair)
            recall_pair.append(R_pair)
            fscore_pair.append(F1_pair)

        return {
            'P_pair': tf.constant(precision_pair), # [b]
            'R_pair': tf.constant(recall_pair), # [b]
            'F1_pair': tf.constant(fscore_pair), # [b]
        }

    def compute_classification_score(self, func_ref, func_est, valid_len, key='Accuracy'):
        # func_ref, func_est = [b, n]
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(func_ref)[1], dtype=tf.float32) # [b, n]
        matched = tf.cast(func_ref == func_est, tf.float32) * seq_mask # [b, n]
        accuracy = tf.reduce_sum(matched, axis=1) / tf.cast(valid_len, tf.float32) # [b]
        return {key: accuracy}

    
    # Create enhanced model
    # model = FunctionalSegmentModel(
    #     max_len=935,
    #     n_units=80,
    #     n_heads=8,
    #     n_layers=2,
    #     cnn_dropout_rate=0.5,
    #     attn_dropout_rate=0.5,
    # )

    print("🎯 Please add your FunctionalSegmentModel class and complete the training loop")
    print("📂 Data loading is ready - train_data and test_data are created successfully!")



class BestTestPredictionSaver:
    """
    SIMPLIFIED: Just saves the intervals and labels that are ALREADY computed
    in your model's compute_segment_score() and compute_pairwise_score()!
    """
    
    def __init__(self, save_dir, n_best=5):
        self.save_dir = save_dir
        self.n_best = n_best
        self.predictions_dir = os.path.join(save_dir, 'best_test_predictions')
        os.makedirs(self.predictions_dir, exist_ok=True)
        print(f"📁 Prediction saver initialized: {self.predictions_dir}")
    
    def save_test_predictions_simple(self, model, test_dataset, test_data, epoch, test_metrics):
        """
        SIMPLE VERSION: Save the intervals and labels that model already computes!
        """
        print(f"\n🎯 Collecting test predictions for epoch {epoch}...")
        
        all_predictions = []
        sample_idx = 0
        
        for batch_idx, batch in enumerate(test_dataset):
            spec, chromagram, stem_spec, stem_chromagram, valid_len, boun_ref, func_ref, sec_ref = batch
            
            # Get predictions
            logits_boun, logits_func, _, _ = model(
                spec, chromagram, stem_spec, stem_chromagram, valid_len, training=False
            )
            prob_boun = tf.nn.sigmoid(logits_boun)
            boun_est = model.decode_boundary(prob_boun, valid_len)
            func_est_smooth = model.decode_labeling(boun_est, logits_func, valid_len)
            
            # Convert to numpy
            boun_ref_np = boun_ref.numpy()
            func_ref_np = func_ref.numpy()
            boun_est_np = boun_est.numpy()
            func_est_np = func_est_smooth.numpy()
            valid_len_np = valid_len.numpy()
            
            batch_size = boun_ref_np.shape[0]
            for i in range(batch_size):
                if sample_idx >= len(test_data['spec']):
                    break
                
                track_id = test_data['section'][sample_idx]
                v_len = int(valid_len_np[i])
                
                # ✨ KEY: Get the SAME intervals/labels that model uses for evaluation!
                # This calls the SAME functions your model uses internally
                interval_ref, label_ref = self._get_intervals_and_labels(
                    boun_ref_np[i, :v_len], 
                    func_ref_np[i, :v_len]
                )
                interval_est, label_est = self._get_intervals_and_labels(
                    boun_est_np[i, :v_len], 
                    func_est_np[i, :v_len]
                )
                
                # Compute F1 (same as model does)
                try:
                    import mir_eval
                    P, R, F = mir_eval.segment.pairwise(
                        interval_ref, label_ref, 
                        interval_est, label_est, 
                        frame_size=0.5
                    )
                    pairwise_f1 = float(F)
                except:
                    pairwise_f1 = 0.0
                
                all_predictions.append({
                    'sample_idx': sample_idx,
                    'track_id': track_id,
                    'pairwise_f1': pairwise_f1,
                    # ✨ THESE ARE THE EXACT VALUES PASSED TO MIR_EVAL!
                    'interval_ref': interval_ref,
                    'label_ref': label_ref,
                    'interval_est': interval_est,
                    'label_est': label_est,
                    'valid_len': v_len
                })
                
                sample_idx += 1
        
        print(f"   ✅ Collected {len(all_predictions)} test predictions")
        
        # Sort by pairwise F1 and get top N
        all_predictions.sort(key=lambda x: x['pairwise_f1'], reverse=True)
        top_predictions = all_predictions[:self.n_best]
        
        # Save them!
        self._save_mireval_format(top_predictions, epoch, test_metrics)
        
        print(f"✅ Saved top {self.n_best} predictions")
        print(f"   📊 Best: {top_predictions[0]['track_id']}, F1: {top_predictions[0]['pairwise_f1']:.4f}")
        print(f"   📊 5th: {top_predictions[-1]['track_id']}, F1: {top_predictions[-1]['pairwise_f1']:.4f}")
        
        return top_predictions
    
    def _get_intervals_and_labels(self, boundaries, labels):
        """
        Extract intervals and labels EXACTLY like your model does.
        This is the same logic used in segmentFrame2interval() and frame2interval()
        """
        # Find boundaries
        boundaries = np.array(boundaries)
        labels = np.array(labels)
        
        # Ensure first frame is boundary
        boundaries[0] = 1
        boundaries = np.append(boundaries, [1])  # Add end boundary
        
        boundary_indices = np.where(boundaries == 1)[0]
        
        # Create intervals (in seconds, assuming 0.5s frame size)
        intervals = []
        segment_labels = []
        
        for i in range(len(boundary_indices) - 1):
            start_idx = boundary_indices[i]
            end_idx = boundary_indices[i + 1]
            
            # Convert to seconds
            start_time = start_idx * 0.5
            end_time = end_idx * 0.5
            intervals.append([start_time, end_time])
            
            # Label at segment start
            segment_labels.append(int(labels[start_idx]))
        
        return np.array(intervals), segment_labels
    
    def _save_mireval_format(self, predictions, epoch, test_metrics):
        """
        Save EXACTLY what mir_eval uses - simple and clean!
        """
        epoch_dir = os.path.join(self.predictions_dir, f'epoch_{epoch:03d}')
        mireval_dir = os.path.join(epoch_dir, 'mireval_data')
        os.makedirs(mireval_dir, exist_ok=True)
        
        # Save each track
        for rank, pred in enumerate(predictions, 1):
            safe_track_id = pred['track_id'].replace('/', '_').replace('\\', '_')
            
            # === GROUND TRUTH ===
            ref_file = os.path.join(mireval_dir, 
                                   f"rank{rank:02d}_{safe_track_id}_GT.csv")
            with open(ref_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Ground Truth - EXACT input to mir_eval.segment.pairwise()'])
                writer.writerow(['# Track:', pred['track_id']])
                writer.writerow(['# Pairwise F1:', f"{pred['pairwise_f1']:.4f}"])
                writer.writerow([])
                writer.writerow(['Start_Time', 'End_Time', 'Duration', 'Label_ID', 'Label_Name'])
                
                for interval, label in zip(pred['interval_ref'], pred['label_ref']):
                    duration = interval[1] - interval[0]
                    label_name = class_conversion(label, reduced=False)
                    writer.writerow([
                        f"{interval[0]:.2f}",
                        f"{interval[1]:.2f}",
                        f"{duration:.2f}",
                        label,
                        label_name
                    ])
            
            # === PREDICTED ===
            est_file = os.path.join(mireval_dir,
                                   f"rank{rank:02d}_{safe_track_id}_PRED.csv")
            with open(est_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Predicted - EXACT input to mir_eval.segment.pairwise()'])
                writer.writerow(['# Track:', pred['track_id']])
                writer.writerow(['# Pairwise F1:', f"{pred['pairwise_f1']:.4f}"])
                writer.writerow([])
                writer.writerow(['Start_Time', 'End_Time', 'Duration', 'Label_ID', 'Label_Name'])
                
                for interval, label in zip(pred['interval_est'], pred['label_est']):
                    duration = interval[1] - interval[0]
                    label_name = class_conversion(label, reduced=False)
                    writer.writerow([
                        f"{interval[0]:.2f}",
                        f"{interval[1]:.2f}",
                        f"{duration:.2f}",
                        label,
                        label_name
                    ])
            
            # === SIDE-BY-SIDE COMPARISON ===
            comp_file = os.path.join(mireval_dir,
                                    f"rank{rank:02d}_{safe_track_id}_COMPARE.csv")
            with open(comp_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Ground Truth vs Predicted'])
                writer.writerow(['# Track:', pred['track_id']])
                writer.writerow(['# Pairwise F1:', f"{pred['pairwise_f1']:.4f}"])
                writer.writerow([])
                writer.writerow(['Seg', 'GT_Start', 'GT_End', 'GT_Dur', 'GT_Label',
                               'Pred_Start', 'Pred_End', 'Pred_Dur', 'Pred_Label', 'Match'])
                
                max_len = max(len(pred['label_ref']), len(pred['label_est']))
                for i in range(max_len):
                    row = [i + 1]
                    
                    # Ground truth
                    if i < len(pred['label_ref']):
                        gt_int = pred['interval_ref'][i]
                        gt_label = pred['label_ref'][i]
                        gt_name = class_conversion(gt_label, reduced=False)
                        row.extend([
                            f"{gt_int[0]:.2f}",
                            f"{gt_int[1]:.2f}",
                            f"{gt_int[1] - gt_int[0]:.2f}",
                            f"{gt_name}({gt_label})"
                        ])
                    else:
                        row.extend(['', '', '', ''])
                    
                    # Predicted
                    if i < len(pred['label_est']):
                        pred_int = pred['interval_est'][i]
                        pred_label = pred['label_est'][i]
                        pred_name = class_conversion(pred_label, reduced=False)
                        row.extend([
                            f"{pred_int[0]:.2f}",
                            f"{pred_int[1]:.2f}",
                            f"{pred_int[1] - pred_int[0]:.2f}",
                            f"{pred_name}({pred_label})"
                        ])
                    else:
                        row.extend(['', '', '', ''])
                    
                    # Match
                    if (i < len(pred['label_ref']) and i < len(pred['label_est'])):
                        match = '✓' if pred['label_ref'][i] == pred['label_est'][i] else '✗'
                        row.append(match)
                    else:
                        row.append('')
                    
                    writer.writerow(row)
        
        # === SUMMARY FILE ===
        summary_file = os.path.join(mireval_dir, 'SUMMARY.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TOP {self.n_best} PREDICTIONS - EPOCH {epoch}\n")
            f.write("="*80 + "\n\n")
            
            f.write("These are the EXACT intervals and labels passed to mir_eval.segment.pairwise()\n\n")
            
            f.write(f"{'Rank':<6} {'Track ID':<40} {'Pairwise F1':<12} {'#Segs GT':<10} {'#Segs Pred':<10}\n")
            f.write("-"*80 + "\n")
            for rank, pred in enumerate(predictions, 1):
                f.write(f"{rank:<6} {pred['track_id']:<40} {pred['pairwise_f1']:<12.4f} "
                       f"{len(pred['label_ref']):<10} {len(pred['label_est']):<10}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            f.write("Files saved:\n")
            for rank, pred in enumerate(predictions, 1):
                safe_id = pred['track_id'].replace('/', '_').replace('\\', '_')
                f.write(f"\nRank {rank}: {pred['track_id']}\n")
                f.write(f"  - rank{rank:02d}_{safe_id}_GT.csv (ground truth)\n")
                f.write(f"  - rank{rank:02d}_{safe_id}_PRED.csv (predicted)\n")
                f.write(f"  - rank{rank:02d}_{safe_id}_COMPARE.csv (comparison)\n")
        
        print(f"   💾 mir_eval data saved to {mireval_dir}/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train V2 or V3 model')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['v2', 'v3', 'both'],
                       help='Which variant to train: v2 (cross-modal only), v3 (full model), or both')
    args = parser.parse_args()
    
    if args.variant == 'v2':
        print("\n" + "🔵"*40)
        print("TRAINING V2: Cross-Modal Attention Only")
        print("🔵"*40 + "\n")
        train_v2_cross_modal_only()
        
    elif args.variant == 'v3':
        print("\n" + "🟢"*40)
        print("TRAINING V3: Full Model")
        print("🟢"*40 + "\n")
        train_v3_full_model()
        
    elif args.variant == 'both':
        print("\n" + "🎯"*40)
        print("TRAINING BOTH VARIANTS")
        print("🎯"*40 + "\n")
        
        print("\n" + "🔵"*40)
        print("PHASE 1: V2 - Cross-Modal Only")
        print("🔵"*40 + "\n")
        train_v2_cross_modal_only()
        
        print("\n" + "🟢"*40)
        print("PHASE 2: V3 - Full Model")
        print("🟢"*40 + "\n")
        train_v3_full_model()
        
        print("\n" + "✅"*40)
        print("BOTH VARIANTS COMPLETE!")
        print("✅"*40 + "\n")
