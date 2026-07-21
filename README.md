# 🔥 SUPRA — Suppression of Pressure Oscillations Using Artificial Intelligence

<p align="center">
  <img src="artifacts/cancellation_psd.png" width="750"/>
  <br/>
  <em>23.06 dB acoustic power suppression at the dominant thermoacoustic instability frequency</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/R²%20Score-0.988-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Cancellation-23.06%20dB-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Latency-6.89%20ms-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

---

<table>
<tr>
<td><b>👨‍💻 Developer</b></td>
<td>Dodla Deekshith Reddy</td>
</tr>
<tr>
<td><b>🏛️ Institute</b></td>
<td>Indian Institute of Technology (ISM), Dhanbad</td>
</tr>
<tr>
<td><b>🏢 Department</b></td>
<td>Mechanical Engineering</td>
</tr>
<tr>
<td><b>👨‍🏫 Supervisor</b></td>
<td>Prof. Rabindra Nath Hota</td>
</tr>
<tr>
<td><b>🔬 Co-Supervisor</b></td>
<td>Dr. Nandan Kumar Jha</td>
</tr>
<tr>
<td><b>📅 Programme</b></td>
<td>Summer Research Internship Scheme (SRIS) 2026</td>
</tr>
</table>

---

## 📌 About the Project

Thermoacoustic instability is one of the most dangerous and difficult-to-control phenomena in modern engineering. In gas turbines, jet engines, and rocket combustors, heat released by the flame can lock in phase with the chamber's natural acoustic modes, creating a **runaway positive feedback loop** that causes:

- 💥 Structural fatigue of turbine blades and combustor walls  
- 🔥 Flame blowout or flashback  
- ⚙️ Complete mechanical failure of the combustion hardware  

**Conventional passive solutions** (Helmholtz resonators, acoustic liners) only suppress instability at a fixed, narrow frequency range. They completely fail when operating conditions change.

**SUPRA** solves this using a **Hybrid Temporal Convolutional Network + Gated Recurrent Unit (TCN-GRU)** deep learning model that:
1. Continuously reads the live combustion pressure signal at 10,000 Hz
2. **Predicts the pressure wave 20 ms into the future**
3. Issues a phase-inverted counter-wave to the acoustic actuator early enough to physically cancel the instability

> **The core insight:** The system doesn't react to the wave — it predicts it. By the time the actuator's counter-wave physically arrives at the combustion chamber, it arrives in perfect destructive interference with the instability.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **Signal Processing** | NumPy, SciPy |
| **Data & Visualisation** | Pandas, Matplotlib |
| **Future Hardware** | NI LabVIEW, NI DAQ, FPGA |

---

## 🏗️ System Architecture

<p align="center">
  <img src="artifacts/supra_flowchart.png" width="800"/>
  <br/>
  <em>End-to-end SUPRA control pipeline</em>
</p>

### How it works — step by step:

| Step | What happens |
|------|-------------|
| 1️⃣ Sense | Piezoelectric transducer samples combustion pressure at 10,000 Hz |
| 2️⃣ Buffer | Most recent 200 samples (20 ms window) assembled in a circular RAM buffer |
| 3️⃣ Features | 3 multi-scale RMS envelopes computed on the fly (5ms, 50ms, 500ms) |
| 4️⃣ Predict | Hybrid TCN-GRU model predicts the pressure 20 ms ahead (~6.89ms inference time) |
| 5️⃣ Invert | Predicted value multiplied by -1 (phase inversion) |
| 6️⃣ Cancel | Inverted signal sent to acoustic actuator — physical waves cancel each other out ✅ |

---

## 🧠 Neural Network Architecture (Hybrid TCN-GRU)

<p align="center">
  <img src="artifacts/supra_nn_architecture.jpg" width="720"/>
  <br/>
  <em>The Hybrid TCN-GRU model: 4-channel input → 6 dilated TCN blocks → SE-Attention → GRU tail → pressure prediction</em>
</p>

### Why Hybrid? Why not just LSTM or TCN alone?

| Model | Problem |
|-------|---------|
| **Pure LSTM** | Processes data one timestep at a time — far too slow for 10,000 Hz real-time control |
| **Pure TCN** | Parallel and fast, but treats all timesteps independently — cannot detect "the instability is growing louder over time" |
| **Hybrid TCN-GRU** ✅ | TCN extracts waveform shape features fast and in parallel. GRU tail adds sequential memory to track the amplitude growth trend |

### Architecture Details

- **Input:** 4-channel tensor of shape `(200, 4)` — raw pressure + RMS at 5ms/50ms/500ms timescales
- **TCN Backbone:** 6 residual blocks with dilation rates `d = [1, 2, 4, 8, 16, 32]`, kernel size 3, 32 filters, SE-channel attention
- **Receptive Field:** 253 samples (covers the full 200-sample window + 53 samples headroom)
- **Self-Attention Layer:** 2-head, captures long-range phase relationships
- **GRU Tail:** Stride-8 downsampling → 25 steps → 32-unit GRU for sequential amplitude memory
- **Output:** Single scalar — predicted pressure at `t + 20ms`
- **Total Parameters:** ~21,500

---

## 📊 Results

> All results evaluated on a **completely held-out 15% test set** — data the model never saw during training.

### Performance Metrics

| Metric | Value |
|--------|-------|
| RMSE | 134.34 Pa |
| MAE | 109.31 Pa |
| **R² Score** | **0.988** |
| Pearson r | 0.996 |
| Durbin-Watson | 1.25 |
| Phase Error | **0.0°** at dominant mode |
| Time-Domain RMS Reduction | **19.55 dB** (~9.5× amplitude) |
| Frequency-Domain Suppression | **23.06 dB** (~200× power) |
| Mean Inference Latency (CPU) | **6.89 ms** |
| p99 Inference Latency (CPU) | 11.04 ms |

---

### Training History

<p align="center">
  <img src="artifacts/training_history.png" width="700"/>
</p>

Training and validation MSE tracked tightly for all 50 epochs with no overfitting. The four kinks are `ReduceLROnPlateau` halving events at epochs 23, 32, 42, and 47.

---

### Prediction Accuracy — Actual vs Predicted

<p align="center">
  <img src="artifacts/pred_vs_actual_scatter.png" width="600"/>
</p>

Tight diagonal clustering with R² = 0.988. No amplitude compression at peaks — a common failure of pure-TCN models that the GRU tail corrects.

---

### Residual Analysis

<p align="center">
  <img src="artifacts/residual_autocorrelation.png" width="45%"/>
  <img src="artifacts/error_analysis.png" width="45%"/>
</p>

ACF of residuals decays rapidly to zero (near white-noise errors). Error distribution is Gaussian and centred at zero — no systematic directional bias.

---

### Active Cancellation — Time Domain

<p align="center">
  <img src="artifacts/cancellation_timedomain.png" width="800"/>
</p>

**19.55 dB RMS reduction.** The amplitude of the instability is reduced by a factor of ~9.5×.

---

### Active Cancellation — Frequency Domain

<p align="center">
  <img src="artifacts/cancellation_psd.png" width="800"/>
</p>

**23.06 dB suppression at ~1000 Hz** — a 200× reduction in acoustic power, completely breaking the Rayleigh feedback loop.

---

### Per-Frequency Reduction

<p align="center">
  <img src="artifacts/per_freq_reduction.png" width="700"/>
</p>

Broadband suppression confirmed across the fundamental mode and all harmonic frequencies — not just a single narrowband spike.

---

### Inference Latency

<p align="center">
  <img src="artifacts/latency_distribution.png" width="600"/>
</p>

Mean 6.89 ms, p99 11.04 ms across 100 CPU benchmark runs. Tight, deterministic distribution confirms readiness for LabVIEW/FPGA deployment.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/SUPRA.git
cd SUPRA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src/supra_train.py

# 4. Evaluate and generate cancellation plots
python src/supra_evaluate.py
