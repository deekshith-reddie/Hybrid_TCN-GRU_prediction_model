"""
SUPRA TCN v5 - Hybrid Thermoacoustic Instability Suppressor
=============================================================


# ==========================================
# STDLIB / THIRD-PARTY IMPORTS
# ==========================================
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Any

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---- Banner ----
print("=" * 65)
print("SUPRA TCN v5 - Hybrid Thermoacoustic Instability Suppressor")
print("=" * 65)
print("\nLoading TensorFlow/Keras...")

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Dense,
    Dropout,
    GRU,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

print(f"TensorFlow version : {tf.__version__}")
print(f"Keras version      : {tf.keras.__version__}")


# ==========================================
# CONFIG
# ==========================================
RANDOM_SEED: int = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -- Data --
SAMPLE_RATE_HZ: int   = 10_000
LOOK_BACK: int        = 200
HORIZON_MS: float     = 20.0
HORIZON: int          = int(SAMPLE_RATE_HZ * HORIZON_MS / 1000)
PRESSURE_COL: str     = 'pressure_pa'

# -- Envelope features (Step 1) --
# Set True to auto-compute and inject multi-scale RMS features into the model.
# These give the TCN explicit visibility into amplitude trends across 3 timescales.
USE_ENVELOPE_FEATURES: bool = True
# Any additional columns from the CSV you want to include (leave empty for auto-only)
EXTRA_FEATURE_COLS: list[str] = []

# -- Forecasting --
FORECAST_STEPS: int   = 1

# -- TCN architecture --
TCN_FILTERS: int         = 32
TCN_KERNEL_SIZE: int     = 3
TCN_DILATIONS: list[int] = [1, 2, 4, 8, 16, 32]
DROPOUT_RATE: float      = 0.2
USE_LAYER_NORM: bool     = True
USE_SE_BLOCK: bool       = True
SE_RATIO: int            = 4
L2_REG: float            = 1e-4

# -- GRU tail (Step 2) --
# Downsamples TCN output (200 -> 200//GRU_DOWNSAMPLE) before feeding to GRU.
# This keeps GRU latency tiny while still capturing amplitude trends.
USE_GRU_TAIL: bool    = True
GRU_UNITS: int        = 32
GRU_DOWNSAMPLE: int   = 8    # 200 samples -> 25 steps before GRU

# -- Self-Attention (Step 3) --
# Applied BETWEEN TCN and GRU. Helps detect mode-switching events.
# Adds ~0.1ms latency. Disable if latency budget is very tight.
USE_SELF_ATTENTION: bool = True
ATT_HEADS: int           = 2
ATT_KEY_DIM: int         = TCN_FILTERS // ATT_HEADS   # = 16

# -- Training --
EPOCHS: int           = 50
BATCH_SIZE: int       = 64
LEARNING_RATE: float  = 1e-3
GRAD_CLIP_NORM: float = 1.0
LR_PATIENCE: int      = 5
LR_FACTOR: float      = 0.5
LR_MIN: float         = 1e-6
LR_WARMUP_EPOCHS: int = 3
EARLY_STOP_PAT: int   = 12
TRAIN_SPLIT: float    = 0.8
USE_MIXED_PRECISION: bool = True

# -- Latency --
CYCLE_PERIOD_MS: float    = 1000.0 / 200.0
TARGET_MARGIN_FACTOR: int = 3
TARGET_LATENCY_MS: float  = CYCLE_PERIOD_MS / TARGET_MARGIN_FACTOR
N_TIMING_RUNS: int        = 100
N_WARMUP_RUNS: int        = 10
MAX_LATENCY_LOG_ROWS: int = 50_000

# -- Prediction --
PREDICT_BATCH_SIZE: int = 512

# -- Plotting --
N_PLOT: int             = 1000
FIGDPI: int             = 150
WELCH_NPERSEG_FINE: int = 2048


# ==========================================
# UTILITY: receptive field
# ==========================================
def compute_receptive_field(kernel_size: int, dilations: list[int]) -> int:
    return 1 + sum(2 * (kernel_size - 1) * d for d in dilations)


rf = compute_receptive_field(TCN_KERNEL_SIZE, TCN_DILATIONS)
print(f"\nTCN receptive field : {rf} samples  (need >= LOOK_BACK={LOOK_BACK})")
if rf < LOOK_BACK:
    raise ValueError(
        f"Receptive field {rf} < LOOK_BACK {LOOK_BACK}. "
        "Add more/larger dilations before training."
    )
margin_pct = 100.0 * (rf - LOOK_BACK) / LOOK_BACK
print(
    f"Margin              : +{margin_pct:.1f}%  "
    f"({'comfortable' if margin_pct > 15 else 'tight - consider one more block'})"
)
print(
    f"Hybrid config       : "
    f"Envelope={USE_ENVELOPE_FEATURES}  "
    f"GRU={USE_GRU_TAIL}  "
    f"Attention={USE_SELF_ATTENTION}"
)


# ==========================================
# DEVICE / PRECISION SETUP
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs visible        : {[g.name for g in gpus]}")
        print("Memory growth       : enabled")
    except RuntimeError as e:
        print(f"GPU config warning  : {e}")

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision     : float16 compute / float32 variables")
else:
    print("GPUs visible        : NONE - running on CPU")
    print("CPU thread config   : setting intra/inter-op parallelism")
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    USE_MIXED_PRECISION = False


# ==========================================
# RESOLVE PATHS
# ==========================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

run_id    = time.strftime('%Y%m%d_%H%M%S')
arch_tag  = (
    f"tcn_v5"
    f"{'_env' if USE_ENVELOPE_FEATURES else ''}"
    f"{'_attn' if USE_SELF_ATTENTION else ''}"
    f"{'_gru' if USE_GRU_TAIL else ''}"
    f"_f{TCN_FILTERS}_d{len(TCN_DILATIONS)}"
    f"{'_se' if USE_SE_BLOCK else ''}"
    f"_{run_id}"
)
artifact_dir = os.path.join(script_dir, 'artifacts', run_id)
os.makedirs(artifact_dir, exist_ok=True)
print(f"\nArtifact directory  : {artifact_dir}")


# ==========================================
# MATPLOTLIB STYLE (fault-tolerant)
# ==========================================
def _apply_plot_style() -> None:
    candidates = ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']
    for s in candidates:
        try:
            plt.style.use(s)
            return
        except OSError:
            continue
    plt.rcParams.update({
        'axes.facecolor': '#f5f5f5',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linewidth': 0.8,
    })


_apply_plot_style()


# ==========================================
# 1. LOAD + VALIDATE DATA
# ==========================================
print("\n" + "=" * 40)
print("STEP 1 - DATA LOADING")
print("=" * 40)

file_name = os.path.join(script_dir, 'thermoacoustic-preprocessed-data.csv')
if not os.path.exists(file_name):
    raise FileNotFoundError(
        f"Could not find '{file_name}'.\n"
        f"Looked in : {script_dir}\n"
        f"Files present: {os.listdir(script_dir)}"
    )

data = pd.read_csv(file_name)
print(f"Loaded  : {len(data):,} rows x {len(data.columns)} columns")
print(f"Columns : {list(data.columns)}")

if PRESSURE_COL not in data.columns:
    raise KeyError(
        f"'{PRESSURE_COL}' not found. Available: {list(data.columns)}"
    )

# NaN / Inf cleaning
n_nan = int(data[PRESSURE_COL].isna().sum())
n_inf = int(np.isinf(data[PRESSURE_COL].values).sum())
if n_nan or n_inf:
    print(
        f"WARNING: {n_nan} NaNs and {n_inf} Infs in {PRESSURE_COL} - "
        "forward-filling then clipping."
    )
    data[PRESSURE_COL] = data[PRESSURE_COL].ffill().bfill()
    med = data[PRESSURE_COL].median()
    data[PRESSURE_COL] = data[PRESSURE_COL].replace([np.inf, -np.inf], med)

# ----------------------------------------------------------
# STEP 1b: ENVELOPE FEATURES (Step 1 of hybrid upgrade)
# Compute multi-scale RMS and amplitude derivative BEFORE
# windowing, so each window sees the amplitude context.
# ----------------------------------------------------------
_envelope_cols: list[str] = []
if USE_ENVELOPE_FEATURES:
    print("\nComputing multi-scale envelope features...")

    # RMS at 3 timescales:
    #   rms_5ms   = 5ms  context  → instantaneous loudness    (window=50  @ 10kHz)
    #   rms_50ms  = 50ms context  → cycle-level amplitude     (window=500 @ 10kHz)
    #   rms_500ms = 500ms context → slow envelope drift       (window=5000@ 10kHz)
    data['rms_5ms']   = data[PRESSURE_COL].rolling(50,   min_periods=1).std().ffill().bfill()
    data['rms_50ms']  = data[PRESSURE_COL].rolling(500,  min_periods=1).std().ffill().bfill()
    data['rms_500ms'] = data[PRESSURE_COL].rolling(5000, min_periods=1).std().ffill().bfill()

    # Amplitude derivative: rate of change of cycle-level RMS (Pa/s)
    # Positive = amplitude growing, Negative = amplitude shrinking
    data['amp_deriv'] = (data['rms_50ms'].diff() * SAMPLE_RATE_HZ).ffill().bfill()

    _envelope_cols = ['rms_5ms', 'rms_50ms', 'rms_500ms', 'amp_deriv']
    print(f"  Added columns : {_envelope_cols}")
    print(f"  rms_5ms   : mean={data['rms_5ms'].mean():.2f} Pa  std={data['rms_5ms'].std():.2f} Pa")
    print(f"  rms_50ms  : mean={data['rms_50ms'].mean():.2f} Pa  std={data['rms_50ms'].std():.2f} Pa")
    print(f"  rms_500ms : mean={data['rms_500ms'].mean():.2f} Pa  std={data['rms_500ms'].std():.2f} Pa")
    print(f"  amp_deriv : mean={data['amp_deriv'].mean():.2f} Pa/s  (>0=growing, <0=shrinking)")

# Build final feature list:
# envelope cols first (they are already in data), then any user-specified extras
available_extras = [c for c in (EXTRA_FEATURE_COLS + _envelope_cols) if c in data.columns]
# Remove duplicates while preserving order
seen = set()
available_extras_dedup = []
for c in available_extras:
    if c not in seen:
        seen.add(c)
        available_extras_dedup.append(c)
missing_extras = [c for c in EXTRA_FEATURE_COLS if c not in data.columns]
if missing_extras:
    print(f"NOTE: Extra feature columns not found, skipping: {missing_extras}")

feature_cols: list[str] = [PRESSURE_COL] + available_extras_dedup
N_FEATURES: int          = len(feature_cols)
print(f"\nFeatures used       : {feature_cols}  ({N_FEATURES} channel(s))")

# Per-feature scaling
raw_arrays: list[np.ndarray] = []
scalers: dict[str, MinMaxScaler] = {}
for col in feature_cols:
    sc  = MinMaxScaler(feature_range=(0, 1))
    arr = data[col].values.reshape(-1, 1)
    raw_arrays.append(sc.fit_transform(arr))
    scalers[col] = sc
    joblib.dump(sc, os.path.join(artifact_dir, f'scaler_{col}.pkl'))

scaled_data: np.ndarray = np.hstack(raw_arrays)
pressure_scaler         = scalers[PRESSURE_COL]
joblib.dump(pressure_scaler, os.path.join(artifact_dir, 'scaler_pressure.pkl'))
print("Scalers saved.")

p = data[PRESSURE_COL].values
p_std_safe = float(p.std()) + 1e-12
p_mean_abs = float(np.abs(p).mean()) + 1e-12
print(f"\nPressure stats (Pa):")
print(
    f"  mean={p.mean():.2f}  std={p.std():.2f}  "
    f"min={p.min():.2f}  max={p.max():.2f}"
)
print(
    f"  Dynamic range : "
    f"{20 * np.log10(p_std_safe / p_mean_abs):.1f} dB (approx)"
)


# ==========================================
# 2. BUILD WINDOWS (stride-trick, no Python loop)
# ==========================================
print("\n" + "=" * 40)
print("STEP 2 - WINDOWING")
print("=" * 40)


def make_windows_strided(
    data_2d: np.ndarray,
    look_back: int,
    horizon: int,
    forecast_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, f = data_2d.shape
    n_windows = n - look_back - horizon - forecast_steps + 2

    s_n, s_f = data_2d.strides
    X = np.lib.stride_tricks.as_strided(
        data_2d,
        shape=(n_windows, look_back, f),
        strides=(s_n, s_n, s_f),
    ).copy()

    target_start = look_back + horizon - 1
    y_source = data_2d[target_start: target_start + n_windows + forecast_steps - 1, 0]
    y = np.lib.stride_tricks.as_strided(
        y_source,
        shape=(n_windows, forecast_steps),
        strides=(y_source.strides[0], y_source.strides[0]),
    ).copy()

    return X.astype(np.float32), y.astype(np.float32)


X, y = make_windows_strided(scaled_data, LOOK_BACK, HORIZON, FORECAST_STEPS)

if FORECAST_STEPS == 1:
    y = y.squeeze(-1)

split   = int(TRAIN_SPLIT * len(X))
X_train = X[:split];  X_test  = X[split:]
y_train = y[:split];  y_test  = y[split:]

print(f"Total windows : {len(X):,}")
print(f"Train         : {len(X_train):,}")
print(f"Test          : {len(X_test):,}")
print(f"X shape       : {X.shape}  (windows, timesteps, features={N_FEATURES})")
print(f"y shape       : {y.shape}  (windows[, forecast_steps])")

AUTO = tf.data.AUTOTUNE

ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=min(len(X_train), 10_000), seed=RANDOM_SEED)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
ds_val = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
print("tf.data pipelines   : created (shuffle + cache + prefetch)")


# ==========================================
# 3. CUSTOM SERIALIZABLE LAYERS
# ==========================================
# FIX: tf.keras.saving is NOT exposed in TF 2.x bundled Keras.
# Version-safe wrapper tries all known paths then falls back to no-op.
# The custom_objects dict passed to load_model() is always the real fallback.
try:
    _reg = tf.keras.saving.register_keras_serializable
except AttributeError:
    try:
        _reg = tf.keras.utils.register_keras_serializable
    except AttributeError:
        def _reg(package='Custom'):
            return lambda cls: cls


@_reg(package='SUPRA')
class LastStepSelector(tf.keras.layers.Layer):
    """Selects final timestep: (batch, T, F) -> (batch, F). Serializable."""

    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()


@_reg(package='SUPRA')
class ScalarSqueeze(tf.keras.layers.Layer):
    """Squeezes (batch, 1) -> (batch,). Serializable."""

    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


@_reg(package='SUPRA')
class DownsampleStride(tf.keras.layers.Layer):
    """
    Takes every `stride`-th timestep along the time axis.
    (batch, T, F) -> (batch, T//stride, F)

    Used to reduce 200 TCN output steps to 25 steps before feeding
    to the GRU tail, cutting sequential GRU cost by ~8x.
    """

    def __init__(self, stride: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride

    def call(self, inputs):
        return inputs[:, ::self.stride, :]

    def get_config(self):
        cfg = super().get_config()
        cfg['stride'] = self.stride
        return cfg


_CUSTOM_OBJECTS: dict = {
    'LastStepSelector': LastStepSelector,
    'ScalarSqueeze'   : ScalarSqueeze,
    'DownsampleStride': DownsampleStride,
}


# ==========================================
# 4. MODEL CONSTRUCTION
# ==========================================
print("\n" + "=" * 40)
print("STEP 3 - MODEL CONSTRUCTION")
print("=" * 40)


def squeeze_excitation(x, filters, ratio, block_id):
    se = GlobalAveragePooling1D(name=f"b{block_id}_se_gap")(x)
    se = Dense(max(1, filters // ratio), activation='relu',
               name=f"b{block_id}_se_fc1")(se)
    se = Dense(filters, activation='sigmoid', name=f"b{block_id}_se_fc2")(se)
    se = Reshape((1, filters), name=f"b{block_id}_se_reshape")(se)
    return Multiply(name=f"b{block_id}_se_scale")([x, se])


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate,
              use_layer_norm=True, use_se=True, se_ratio=4,
              l2_reg=1e-4, block_id=0):
    prev = x
    pfx  = f"b{block_id}_d{dilation_rate}"
    reg  = l2(l2_reg) if l2_reg > 0 else None

    out = LayerNormalization(name=f"{pfx}_ln1")(x) if use_layer_norm else x
    out = Conv1D(filters, kernel_size, padding='causal',
                 dilation_rate=dilation_rate, kernel_regularizer=reg,
                 name=f"{pfx}_conv1")(out)
    out = Activation('relu', name=f"{pfx}_act1")(out)
    out = Dropout(dropout_rate, name=f"{pfx}_drop1")(out)

    out = LayerNormalization(name=f"{pfx}_ln2")(out) if use_layer_norm else out
    out = Conv1D(filters, kernel_size, padding='causal',
                 dilation_rate=dilation_rate, kernel_regularizer=reg,
                 name=f"{pfx}_conv2")(out)
    out = Activation('relu', name=f"{pfx}_act2")(out)
    out = Dropout(dropout_rate, name=f"{pfx}_drop2")(out)

    if use_se:
        out = squeeze_excitation(out, filters, se_ratio, block_id)

    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding='same', name=f"{pfx}_skip_proj")(prev)

    return Add(name=f"{pfx}_add")([prev, out])


# ---- Build graph ----
inputs = Input(shape=(LOOK_BACK, N_FEATURES), name="pressure_window")
x = inputs

# TCN backbone — same 6 residual blocks as v4
for block_idx, dilation in enumerate(TCN_DILATIONS):
    x = tcn_block(
        x,
        filters=TCN_FILTERS,
        kernel_size=TCN_KERNEL_SIZE,
        dilation_rate=dilation,
        dropout_rate=DROPOUT_RATE,
        use_layer_norm=USE_LAYER_NORM,
        use_se=USE_SE_BLOCK,
        se_ratio=SE_RATIO,
        l2_reg=L2_REG,
        block_id=block_idx,
    )
# x shape: (batch, 200, TCN_FILTERS)

# ---- Step 3: Multi-Head Self-Attention ----
# Runs BEFORE GRU. Each timestep can attend to ALL other timesteps,
# making it sensitive to where in the window amplitude transitions occur.
if USE_SELF_ATTENTION:
    attn_out = MultiHeadAttention(
        num_heads=ATT_HEADS,
        key_dim=ATT_KEY_DIM,
        dropout=DROPOUT_RATE / 4,
        name="self_attn",
    )(x, x)                              # self-attention: query=key=value=x
    x = Add(name="attn_residual")([x, attn_out])
    x = LayerNormalization(name="attn_ln")(x)
# x shape: (batch, 200, TCN_FILTERS)

# ---- Step 2: GRU tail ----
# DownsampleStride reduces 200 steps -> 25 steps before the GRU,
# so GRU only processes 25 sequential steps instead of 200 (8x faster).
# GRU returns only the LAST hidden state -> (batch, GRU_UNITS).
if USE_GRU_TAIL:
    x = DownsampleStride(stride=GRU_DOWNSAMPLE, name="gru_downsample")(x)
    # x shape: (batch, 25, TCN_FILTERS)
    x = GRU(
        GRU_UNITS,
        return_sequences=False,
        dropout=DROPOUT_RATE / 4,
        recurrent_dropout=0.0,
        name="gru_tail",
    )(x)
    # x shape: (batch, GRU_UNITS)
    head_in_dim = GRU_UNITS
else:
    # Original v4 path: just take the final TCN timestep
    x = LastStepSelector(name="last_step_select")(x)
    # x shape: (batch, TCN_FILTERS)
    head_in_dim = TCN_FILTERS

# ---- Dense prediction head ----
x       = Dense(head_in_dim,      activation='relu', name="head_dense1")(x)
x       = Dropout(DROPOUT_RATE / 2,                  name="head_drop1")(x)
x       = Dense(head_in_dim // 2, activation='relu', name="head_dense2")(x)
x       = Dropout(DROPOUT_RATE / 4,                  name="head_drop2")(x)
raw_out = Dense(
    FORECAST_STEPS if FORECAST_STEPS > 1 else 1,
    dtype='float32',
    name="forecast_out",
)(x)

if FORECAST_STEPS == 1:
    outputs = ScalarSqueeze(name="scalar_squeeze")(raw_out)
else:
    outputs = raw_out

model = Model(inputs, outputs, name="SUPRA_TCN_v5_Hybrid")

opt_kwargs: dict[str, Any] = {'learning_rate': LEARNING_RATE}
if GRAD_CLIP_NORM > 0:
    opt_kwargs['clipnorm'] = GRAD_CLIP_NORM

model.compile(
    optimizer=Adam(**opt_kwargs),
    loss='mean_squared_error',
    metrics=['mae'],
)
model.summary(line_length=80)

total_params = model.count_params()
print(f"\nTotal parameters    : {total_params:,}")
print(f"Approx model size   : {total_params * 4 / 1024:.1f} KB (float32)")
print(f"Architecture        : TCN"
      f"{' + Attention' if USE_SELF_ATTENTION else ''}"
      f"{' + GRU' if USE_GRU_TAIL else ''}"
      f" + Dense Head")


# ==========================================
# 5. CALLBACKS
# ==========================================
best_weights_path = os.path.join(artifact_dir, 'best_weights.keras')
csv_log_path      = os.path.join(artifact_dir, 'training_log.csv')


class WarmUpLR(tf.keras.callbacks.Callback):
    """Linear LR warm-up. FIX: direct assignment works in Keras 2 and 3."""

    def __init__(self, warmup_epochs, target_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr     = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            try:
                self.model.optimizer.learning_rate = lr
            except AttributeError:
                self.model.optimizer.lr.assign(lr)
            print(f"\n[WarmUpLR] epoch {epoch+1}/{self.warmup_epochs} -> lr={lr:.2e}")


callbacks = [
    EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PAT,
                  restore_best_weights=True, verbose=1, min_delta=1e-6),
    ModelCheckpoint(filepath=best_weights_path, monitor='val_loss',
                    save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1),
    CSVLogger(csv_log_path, append=False),
]
if LR_WARMUP_EPOCHS > 0:
    callbacks.insert(0, WarmUpLR(LR_WARMUP_EPOCHS, LEARNING_RATE))


# ==========================================
# 6. TRAIN
# ==========================================
print("\n" + "=" * 40)
print("STEP 4 - TRAINING")
print("=" * 40)
print(f"Device        : {'GPU' if gpus else 'CPU'}")
print(f"Mixed prec.   : {USE_MIXED_PRECISION}")
print(f"Epochs        : up to {EPOCHS} (EarlyStopping patience={EARLY_STOP_PAT})")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Initial LR    : {LEARNING_RATE}  (warm-up: {LR_WARMUP_EPOCHS} epochs)")
print(f"Grad clip     : {GRAD_CLIP_NORM if GRAD_CLIP_NORM > 0 else 'disabled'}\n")

t_train_start = time.perf_counter()
history = model.fit(
    ds_train, epochs=EPOCHS,
    validation_data=ds_val,
    callbacks=callbacks,
    verbose=1,
)
t_train_end  = time.perf_counter()
train_time_s = t_train_end - t_train_start
actual_epochs = len(history.history['loss'])

print(
    f"\nTraining time   : {train_time_s:.1f}s  "
    f"({train_time_s / actual_epochs:.2f}s/epoch, {actual_epochs} epochs)"
)


# ==========================================
# 7. SAVE + ROUND-TRIP CHECK
# ==========================================
model_path = os.path.join(artifact_dir, f'{arch_tag}.keras')
model.save(model_path)
print(f"\nModel saved to  : {model_path}")

reloaded = tf.keras.models.load_model(model_path, custom_objects=_CUSTOM_OBJECTS)
print("Save/load round trip : OK")

_sample_np  = X_test[0:4].astype(np.float32)
_pred_orig  = model.predict(_sample_np, verbose=0)
_pred_rel   = reloaded.predict(_sample_np, verbose=0)
assert np.allclose(_pred_orig, _pred_rel, atol=1e-4), \
    "Reloaded model gave different predictions!"
print("Numerical consistency (original vs reloaded): PASS")


# ==========================================
# 8. LATENCY ANALYSIS
# ==========================================
print("\n" + "=" * 40)
print("STEP 5 - LATENCY ANALYSIS")
print("=" * 40)

single_window_np = X_test[0:1].astype(np.float32)


def measure_latency(device_name, x_np, model_to_use):
    with tf.device(device_name):
        x_t = tf.constant(x_np)

        @tf.function(reduce_retracing=True)
        def fast_infer(inp):
            return model_to_use(inp, training=False)

        warmup_lats = []
        for _ in range(N_WARMUP_RUNS):
            t0 = time.perf_counter()
            fast_infer(x_t)
            warmup_lats.append((time.perf_counter() - t0) * 1e3)

        lats = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            fast_infer(x_t)
            lats.append((time.perf_counter() - t0) * 1e3)

    arr = np.array(lats)
    return {
        'device'      : device_name,
        'n_runs'      : N_TIMING_RUNS,
        'n_warmup'    : N_WARMUP_RUNS,
        'warmup_mean' : float(np.mean(warmup_lats)),
        'p50'         : float(np.percentile(arr, 50)),
        'p95'         : float(np.percentile(arr, 95)),
        'p99'         : float(np.percentile(arr, 99)),
        'mean'        : float(np.mean(arr)),
        'max'         : float(np.max(arr)),
        'std'         : float(np.std(arr)),
        'jitter'      : float(np.std(arr)),
        'raw'         : arr,
    }


latency_results: dict = {}
latency_results['CPU'] = measure_latency('/CPU:0', single_window_np, model)
if gpus:
    try:
        latency_results['GPU'] = measure_latency('/GPU:0', single_window_np, model)
    except Exception as exc:
        print(f"GPU timing failed ({exc}) - CPU results only.")

print(f"\nCycle budget    : {CYCLE_PERIOD_MS:.2f} ms  (200 Hz loop)")
print(f"Target (<1/{TARGET_MARGIN_FACTOR} budget): {TARGET_LATENCY_MS:.2f} ms\n")
print(f"{'Device':<6}  {'mean':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  "
      f"{'max':>7}  {'jitter':>7}  {'verdict'}")
print("-" * 75)

latency_summary: dict = {}
for dev, res in latency_results.items():
    p99     = res['p99']
    verdict = (
        "PASS"     if p99 < TARGET_LATENCY_MS else
        "MARGINAL" if p99 < CYCLE_PERIOD_MS   else
        "FAIL"
    )
    print(
        f"{dev:<6}  {res['mean']:>6.3f}ms  {res['p50']:>6.3f}ms  "
        f"{res['p95']:>6.3f}ms  {res['p99']:>6.3f}ms  "
        f"{res['max']:>6.3f}ms  {res['jitter']:>6.3f}ms  {verdict}"
    )
    latency_summary[dev] = {
        key: float(val) if isinstance(val, (float, np.floating, np.float32, np.float64)) else val
        for key, val in res.items() if key != 'raw'
    }


# ==========================================
# 9. LATENCY LOG
# ==========================================
try:
    log_path   = os.path.join(script_dir, 'latency_log.csv')
    log_exists = os.path.exists(log_path)
    if log_exists:
        with open(log_path, 'r') as _f:
            existing_rows = sum(1 for _ in _f) - 1
        if existing_rows > MAX_LATENCY_LOG_ROWS:
            print(f"\nWARNING: latency_log.csv has {existing_rows:,} rows. Consider archiving.")
    with open(log_path, 'a') as f:
        if not log_exists:
            f.write('timestamp,run_id,device,model_tag,latency_ms\n')
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        for dev, res in latency_results.items():
            for sample_ms in res['raw']:
                f.write(f'{ts},{run_id},{dev},{arch_tag},{sample_ms:.4f}\n')
    n_logged = sum(len(r['raw']) for r in latency_results.values())
    print(f"\nLogged {n_logged} latency samples -> {log_path}")
except PermissionError:
    print("\nWARNING: latency_log.csv locked (Excel / Controlled Folder Access?).")
except Exception as exc:
    print(f"\nWARNING: latency logging failed ({exc}).")


# ==========================================
# 10. ACCURACY METRICS
# ==========================================
print("\n" + "=" * 40)
print("STEP 6 - ACCURACY + CANCELLATION")
print("=" * 40)

pred_scaled = model.predict(X_test, verbose=0, batch_size=PREDICT_BATCH_SIZE)

if FORECAST_STEPS > 1:
    pred_scaled_1 = pred_scaled[:, 0:1]
    y_test_1      = y_test[:, 0:1]
else:
    pred_scaled_1 = pred_scaled.reshape(-1, 1)
    y_test_1      = y_test.reshape(-1, 1)

pred_pa   = pressure_scaler.inverse_transform(pred_scaled_1).flatten()
actual_pa = pressure_scaler.inverse_transform(y_test_1).flatten()

rmse_val   = float(np.sqrt(np.mean((pred_pa - actual_pa) ** 2)))
mae_val    = float(mean_absolute_error(actual_pa, pred_pa))
r2_val     = float(r2_score(actual_pa, pred_pa))
sig_std    = float(np.std(actual_pa))
corr, _    = pearsonr(actual_pa, pred_pa)

print(f"\nTest-set accuracy:")
print(f"  RMSE          : {rmse_val:.4f} Pa")
print(f"  MAE           : {mae_val:.4f} Pa")
print(f"  Signal std    : {sig_std:.4f} Pa")
print(f"  R2            : {r2_val:.4f}  (sklearn r2_score)")
print(f"  Pearson r     : {float(corr):.4f}")

residuals     = pred_pa - actual_pa
residual_std  = float(np.std(residuals))
residual_mean = float(np.mean(residuals))
print(f"\nResidual analysis:")
print(
    f"  Mean residual : {residual_mean:.4f} Pa  "
    f"({'unbiased' if abs(residual_mean) < 0.01 * sig_std else 'BIASED - check scaling'})"
)
print(f"  Residual std  : {residual_std:.4f} Pa")

e       = residuals
dw_stat = float(np.sum(np.diff(e) ** 2) / (np.dot(e, e) + 1e-30))
print(
    f"  Durbin-Watson : {dw_stat:.3f}  "
    f"({'OK' if 1.5 < dw_stat < 2.5 else 'autocorrelated - consider larger model'})"
)


# ==========================================
# 11. PHASE ERROR ANALYSIS (FFT-based)
# ==========================================
def normalise(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-12)


xcorr    = scipy_signal.correlate(
    normalise(actual_pa), normalise(pred_pa), mode='full', method='fft'
)
lags     = scipy_signal.correlation_lags(len(actual_pa), len(pred_pa), mode='full')
peak_lag = int(lags[np.argmax(xcorr)])
peak_lag_ms = peak_lag / SAMPLE_RATE_HZ * 1000.0

freqs, psd = scipy_signal.welch(
    actual_pa, fs=SAMPLE_RATE_HZ,
    nperseg=min(WELCH_NPERSEG_FINE, len(actual_pa) // 4)
)
dom_bin       = int(np.argmax(psd[1:]) + 1)
dom_freq      = float(freqs[dom_bin])
dom_period_ms = 1000.0 / dom_freq
phase_deg     = (peak_lag_ms / dom_period_ms) * 360.0

print(f"\nPhase analysis:")
print(f"  Dominant instability freq  : {dom_freq:.1f} Hz  (period {dom_period_ms:.3f} ms)")
print(f"  Peak cross-correlation lag : {peak_lag} samples  ({peak_lag_ms:.3f} ms)")
print(f"  Estimated phase error      : {phase_deg:.1f} deg")

abs_phase = abs(phase_deg)
if   abs_phase < 10: print("  Phase verdict : EXCELLENT - near-perfect timing")
elif abs_phase < 30: print("  Phase verdict : GOOD - acceptable for cancellation")
elif abs_phase < 60: print("  Phase verdict : MARGINAL - cancellation will be partial")
else:                print("  Phase verdict : POOR - phase error dominates")


# ==========================================
# 12. CANCELLATION SIMULATION
# ==========================================
anti_wave     = -pred_pa
resultant     = actual_pa + anti_wave
original_rms  = float(np.std(actual_pa))
resultant_rms = float(np.std(resultant))
reduction_db  = (
    float(20.0 * np.log10(original_rms / (resultant_rms + 1e-30)))
    if resultant_rms > 0 else float('inf')
)

print(f"\nTime-domain cancellation (assumes perfect actuation):")
print(f"  Original RMS  : {original_rms:.4f} Pa")
print(f"  Resultant RMS : {resultant_rms:.4f} Pa")
print(f"  Reduction     : {reduction_db:.1f} dB")


# ==========================================
# 13. FREQUENCY-DOMAIN CANCELLATION
# ==========================================
nperseg = min(1024, len(actual_pa) // 4)
f_orig,  Pxx_orig  = scipy_signal.welch(actual_pa, fs=SAMPLE_RATE_HZ, nperseg=nperseg)
f_resid, Pxx_resid = scipy_signal.welch(resultant, fs=SAMPLE_RATE_HZ, nperseg=nperseg)

with np.errstate(divide='ignore', invalid='ignore'):
    per_bin_db = np.where(
        Pxx_resid > 0,
        10.0 * np.log10(Pxx_orig / (Pxx_resid + 1e-30)),
        60.0,
    )

dom_bin_fd       = int(np.argmin(np.abs(f_orig - dom_freq)))
reduction_at_dom = float(per_bin_db[dom_bin_fd])
quality_label    = (
    'excellent' if reduction_at_dom > 15 else
    'moderate'  if reduction_at_dom > 6  else
    'poor'
)
print(f"\nFrequency-domain cancellation at {dom_freq:.1f} Hz:")
print(f"  {reduction_at_dom:.1f} dB  ({quality_label})")


# ==========================================
# 14. JSON EXPORT
# ==========================================
results_dict: dict = {
    'run_id': run_id, 'model_tag': arch_tag,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'look_back': LOOK_BACK, 'horizon_ms': HORIZON_MS,
        'tcn_filters': TCN_FILTERS, 'tcn_dilations': TCN_DILATIONS,
        'receptive_field': rf, 'dropout': DROPOUT_RATE,
        'layer_norm': USE_LAYER_NORM, 'se_block': USE_SE_BLOCK,
        'n_features': N_FEATURES, 'feature_cols': feature_cols,
        'use_envelope': USE_ENVELOPE_FEATURES,
        'use_attention': USE_SELF_ATTENTION, 'att_heads': ATT_HEADS,
        'use_gru': USE_GRU_TAIL, 'gru_units': GRU_UNITS,
        'gru_downsample': GRU_DOWNSAMPLE,
    },
    'training': {
        'actual_epochs': actual_epochs,
        'train_time_s': round(train_time_s, 2),
        'best_val_loss': float(min(history.history['val_loss'])),
    },
    'accuracy': {
        'rmse_pa': round(rmse_val, 5), 'mae_pa': round(mae_val, 5),
        'r2': round(r2_val, 5), 'pearson_r': round(float(corr), 5),
        'durbin_watson': round(dw_stat, 4),
    },
    'phase': {
        'dominant_freq_hz': round(dom_freq, 2),
        'phase_error_deg': round(phase_deg, 2),
    },
    'cancellation': {
        'reduction_db_td': round(reduction_db, 2),
        'reduction_db_fd': round(reduction_at_dom, 2),
    },
    'latency': latency_summary,
}
json_path = os.path.join(artifact_dir, 'results.json')
with open(json_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\nFull results saved : {json_path}")


# ==========================================
# 15. PLOTS (8 figures)
# ==========================================
print("\n" + "=" * 40)
print("STEP 7 - PLOTTING")
print("=" * 40)


def save_fig(fig, name):
    path = os.path.join(artifact_dir, name)
    fig.savefig(path, dpi=FIGDPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# A: Training dashboard
fig_hist = plt.figure(figsize=(14, 8), dpi=FIGDPI)
gs       = fig_hist.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
ep_axis  = range(1, actual_epochs + 1)

ax = fig_hist.add_subplot(gs[0, 0])
ax.plot(ep_axis, history.history['loss'],     label='Train', color='royalblue')
ax.plot(ep_axis, history.history['val_loss'], label='Val',   color='tomato')
ax.set_title('Loss (MSE)'); ax.set_yscale('log'); ax.legend()

ax = fig_hist.add_subplot(gs[0, 1])
ax.plot(ep_axis, history.history['mae'],     label='Train', color='royalblue')
ax.plot(ep_axis, history.history['val_mae'], label='Val',   color='tomato')
ax.set_title('MAE'); ax.legend()

ax = fig_hist.add_subplot(gs[1, 0])
lrs = history.history.get('lr', [LEARNING_RATE] * actual_epochs)
ax.plot(ep_axis, lrs, color='darkorange')
ax.set_title('Learning rate'); ax.set_yscale('log')

ax = fig_hist.add_subplot(gs[1, 1])
ax.plot(ep_axis, history.history['val_loss'], color='tomato')
best_ep = int(np.argmin(history.history['val_loss'])) + 1
ax.axvline(best_ep, color='green', linewidth=1.5, linestyle='--',
           label=f'Best: ep {best_ep}')
ax.set_title('Val loss + best epoch'); ax.set_yscale('log'); ax.legend()

fig_hist.suptitle(f'{arch_tag}\nTraining Dashboard', fontsize=11, fontweight='bold')
save_fig(fig_hist, 'training_history.png')


# B: Cancellation time-domain
sl     = slice(0, min(N_PLOT, len(actual_pa)))
t_axis = np.arange(sl.stop) / SAMPLE_RATE_HZ * 1000

fig_td, ax = plt.subplots(figsize=(13, 5), dpi=FIGDPI)
ax.plot(t_axis, actual_pa[sl], label='Original',
        color='steelblue', alpha=0.85)
ax.plot(t_axis, anti_wave[sl], label='Anti-wave',
        color='seagreen', alpha=0.85, linestyle='--')
ax.plot(t_axis, resultant[sl], label=f'Resultant ({reduction_db:.1f} dB)',
        color='firebrick', alpha=0.95, linewidth=1.5)
ax.axhline(0, color='grey', linewidth=0.5)
ax.set_title(
    f'{arch_tag}  |  {reduction_db:.1f} dB  |  phase error {phase_deg:.1f} deg'
)
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Pressure (Pa)'); ax.legend()
fig_td.tight_layout()
save_fig(fig_td, 'cancellation_timedomain.png')


# C: PSD comparison
fig_fd, ax = plt.subplots(figsize=(11, 4), dpi=FIGDPI)
ax.semilogy(f_orig,  Pxx_orig,  label='Original',  color='steelblue', alpha=0.8)
ax.semilogy(f_resid, Pxx_resid, label='Resultant', color='firebrick', alpha=0.8)
ax.axvline(dom_freq, color='orange', linewidth=1.2, linestyle=':',
           label=f'Dom. {dom_freq:.1f} Hz')
ax.set_xlim(0, min(SAMPLE_RATE_HZ / 2, 2000))
ax.set_title('PSD before/after cancellation')
ax.set_xlabel('Frequency (Hz)'); ax.legend()
fig_fd.tight_layout()
save_fig(fig_fd, 'cancellation_psd.png')


# D: Per-bin reduction
fig_red, ax = plt.subplots(figsize=(11, 4), dpi=FIGDPI)
ax.plot(f_orig, per_bin_db, color='purple', alpha=0.8)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', label='0 dB')
ax.axvline(dom_freq, color='orange', linewidth=1.2, linestyle=':',
           label=f'{dom_freq:.1f} Hz  ({reduction_at_dom:.1f} dB)')
ax.fill_between(f_orig, per_bin_db, 0, where=(per_bin_db > 0),
                alpha=0.25, color='green', label='Reduction')
ax.fill_between(f_orig, per_bin_db, 0, where=(per_bin_db < 0),
                alpha=0.25, color='red', label='Amplification')
ax.set_xlim(0, min(SAMPLE_RATE_HZ / 2, 2000))
ax.set_ylim(-20, max(40, float(per_bin_db.max()) + 5))
ax.set_title('Per-frequency cancellation (dB)')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8)
fig_red.tight_layout()
save_fig(fig_red, 'per_freq_reduction.png')


# E: Latency distributions
n_dev = len(latency_results)
fig_lat, axes = plt.subplots(1, n_dev, figsize=(6 * n_dev, 4),
                              dpi=FIGDPI, squeeze=False)
for ax_i, (dev, res) in enumerate(latency_results.items()):
    ax = axes[0][ax_i]
    ax.hist(res['raw'], bins=30, color='steelblue',
            edgecolor='white', alpha=0.85, density=True)
    ax.axvline(res['p50'], color='green',  linewidth=1.5,
               label=f"p50 {res['p50']:.2f}ms")
    ax.axvline(res['p95'], color='orange', linewidth=1.5,
               label=f"p95 {res['p95']:.2f}ms")
    ax.axvline(res['p99'], color='red',    linewidth=1.5,
               label=f"p99 {res['p99']:.2f}ms")
    ax.axvline(TARGET_LATENCY_MS, color='black', linewidth=1.2,
               linestyle='--', label=f"target {TARGET_LATENCY_MS:.2f}ms")
    ax.set_title(f'Latency - {dev}')
    ax.set_xlabel('Latency (ms)'); ax.legend(fontsize=8)
fig_lat.tight_layout()
save_fig(fig_lat, 'latency_distribution.png')


# F: Predicted vs Actual scatter
fig_sc, ax = plt.subplots(figsize=(6, 6), dpi=FIGDPI)
lim = [
    min(float(actual_pa.min()), float(pred_pa.min())),
    max(float(actual_pa.max()), float(pred_pa.max())),
]
ax.scatter(actual_pa[::10], pred_pa[::10], alpha=0.3, s=4, color='steelblue')
ax.plot(lim, lim, 'r--', linewidth=1.2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f'Pred vs Actual  R2={r2_val:.3f}  r={float(corr):.3f}')
ax.set_xlabel('Actual (Pa)'); ax.set_ylabel('Predicted (Pa)')
ax.set_aspect('equal', adjustable='box'); ax.legend(fontsize=8)
fig_sc.tight_layout()
save_fig(fig_sc, 'pred_vs_actual_scatter.png')


# G: Residual autocorrelation
max_lag_acf = min(200, len(residuals) // 4)
acf_lags    = np.arange(max_lag_acf)
acf_vals    = np.array([
    np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] if lag > 0 else 1.0
    for lag in acf_lags
])
conf_95 = 1.96 / np.sqrt(len(residuals))

fig_acf, ax = plt.subplots(figsize=(12, 4), dpi=FIGDPI)
ax.bar(acf_lags, acf_vals, color='steelblue', alpha=0.7, width=1.0)
ax.axhline(+conf_95, color='red', linewidth=1.2, linestyle='--',
           label=f'95% CI (+/-{conf_95:.3f})')
ax.axhline(-conf_95, color='red', linewidth=1.2, linestyle='--')
ax.axhline(0, color='black', linewidth=0.8)
dw_ok = 1.5 < dw_stat < 2.5
ax.set_title(
    f'Residual ACF  |  DW={dw_stat:.3f}  '
    f'({"white noise OK" if dw_ok else "autocorrelated - consider larger model"})'
)
ax.set_xlabel('Lag (samples)'); ax.set_ylabel('ACF')
ax.set_xlim(-1, max_lag_acf); ax.set_ylim(-0.3, 1.05)
ax.legend(fontsize=9)
fig_acf.tight_layout()
save_fig(fig_acf, 'residual_autocorrelation.png')


# H: Error distribution + residuals over time
fig_err, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=FIGDPI)
ax = axes[0]
ax.hist(residuals, bins=60, color='steelblue',
        edgecolor='white', alpha=0.85, density=True)
ax.axvline(residual_mean, color='red', linewidth=1.5,
           label=f'Mean {residual_mean:.4f} Pa')
ax.axvline(+residual_std, color='orange', linewidth=1.2, linestyle='--',
           label=f'+/-std {residual_std:.4f} Pa')
ax.axvline(-residual_std, color='orange', linewidth=1.2, linestyle='--')
ax.set_title('Residual distribution'); ax.legend(fontsize=8)

ax = axes[1]
sl_err = slice(0, min(N_PLOT, len(residuals)))
ax.plot(
    np.arange(sl_err.stop) / SAMPLE_RATE_HZ * 1000,
    residuals[sl_err],
    color='steelblue', alpha=0.7, linewidth=0.7,
)
ax.axhline(0,             color='grey',   linewidth=0.8, linestyle='--')
ax.axhline(+residual_std, color='orange', linewidth=1.0,
           linestyle='--', label='+/-std')
ax.axhline(-residual_std, color='orange', linewidth=1.0, linestyle='--')
ax.set_title('Residuals over time')
ax.set_xlabel('Time (ms)'); ax.legend(fontsize=8)

fig_err.suptitle('Prediction Error Analysis', fontsize=11, fontweight='bold')
fig_err.tight_layout()
save_fig(fig_err, 'error_analysis.png')


# ==========================================
# 16. FINAL SUMMARY
# ==========================================
print("\n" + "=" * 65)
print("RUN COMPLETE - SUMMARY")
print("=" * 65)
print(f"  Run ID              : {run_id}")
print(f"  Model tag           : {arch_tag}")
print(f"  Architecture        : TCN"
      f"{' + Attention' if USE_SELF_ATTENTION else ''}"
      f"{' + GRU' if USE_GRU_TAIL else ''}"
      f" + Dense Head")
print(f"  Input features      : {N_FEATURES}  {feature_cols}")
print(f"  Receptive field     : {rf} samples  (LOOK_BACK={LOOK_BACK})")
print(f"  Parameters          : {total_params:,}")
print(f"  Training epochs     : {actual_epochs} / {EPOCHS}")
print(f"  Best val_loss       : {min(history.history['val_loss']):.6f}")
print(f"  RMSE                : {rmse_val:.4f} Pa")
print(f"  MAE                 : {mae_val:.4f} Pa")
print(f"  R2                  : {r2_val:.4f}")
print(f"  Pearson r           : {float(corr):.4f}")
print(f"  Durbin-Watson       : {dw_stat:.3f}")
print(f"  Phase error         : {phase_deg:.1f} deg")
print(f"  TD cancellation     : {reduction_db:.1f} dB")
print(f"  FD cancellation     : {reduction_at_dom:.1f} dB  (at {dom_freq:.1f} Hz)")
for dev, res in latency_results.items():
    print(
        f"  Latency ({dev:3s}) p99  : {res['p99']:.3f} ms"
        f"  (target < {TARGET_LATENCY_MS:.2f} ms)"
    )
print(f"\n  Artifacts dir       : {artifact_dir}")
print("=" * 65)
print("\nNEXT STEPS:")
print("  1. If phase error > 30 deg  -> increase HORIZON_MS")
print("  2. If latency FAIL          -> set USE_SELF_ATTENTION=False first")
print("  3. If latency still FAIL    -> set USE_GRU_TAIL=False")
print("  4. If DW stat < 1.5         -> increase TCN_FILTERS or GRU_UNITS")
print("  5. Compare results.json with v4 run to see improvement")
