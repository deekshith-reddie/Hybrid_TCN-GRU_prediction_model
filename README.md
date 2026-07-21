# SUPRA: Real-Time Suppression of Thermoacoustic Instability Using AI

<p align="center">
  <img src="artifacts/cancellation_psd.png" width="700"/>
</p>

> **B.Tech Summer Research Internship Scheme (SRIS) 2026 Project**  
> **Institute:** Indian Institute of Technology (Indian School of Mines), Dhanbad  
> **Department:** Mechanical Engineering  
> **Supervised by:** Prof. Rabindra Nath Hota & Dr. Nandan Kumar Jha  
> **Developed by:** Dodla Deekshith Reddy  

---

## 📌 Abstract / Project Objective
Thermoacoustic instability causes violent pressure oscillations in gas turbines and rocket engines, often leading to complete mechanical failure. Conventional passive controllers fail when engine conditions change. 

**SUPRA** is an active noise cancellation framework that uses a **Hybrid TCN-GRU Neural Network** to predict combustion chamber pressure 20 ms into the future. By predicting the wave before it arrives, the system compensates for hardware latency and fires a phase-inverted counter-wave to cancel the instability in real-time.

---

## 🛠️ Tech Stack & Requirements

**Software Requirements:**
- Python 3.8+
- TensorFlow / Keras 2.x
- NumPy, Pandas, Matplotlib, SciPy
- Jupyter Notebook (for EDA)

**Proposed Hardware Setup (Future Scope):**
- NI LabVIEW Data Acquisition System
- Piezoelectric Dynamic Pressure Transducer
- Acoustic Actuator (Loudspeaker)

---

## 🏗️ System Architecture (Block Diagram)

<p align="center">
  <img src="artifacts/supra_flowchart.png" width="800"/>
</p>

### Project Modules:
1. **Data Acquisition Module:** Reads 10,000 Hz pressure signals into a 20 ms sliding window buffer.
2. **Feature Engineering Module:** Calculates multi-scale RMS amplitude envelopes (5ms, 50ms, 500ms) on the fly.
3. **Deep Learning Controller (Hybrid TCN-GRU):** 
   - **TCN Blocks:** Extracts fast waveform shape features using 6 dilated convolutional layers.
   - **GRU Tail:** Remembers the sequential growth/decay trend of the instability amplitude.
4. **Phase Inverter & Actuation:** Multiplies the predicted pressure by -1 and sends the voltage to the actuator.

---

## 🧠 Neural Network Model (Hybrid TCN-GRU)

<p align="center">
  <img src="artifacts/supra_nn_architecture.jpg" width="700"/>
</p>

We designed a hybrid model because pure LSTMs are too slow for real-time 10,000 Hz processing, and pure TCNs fail to track long-term amplitude growth. 
* **Parameters:** ~21,500
* **Receptive Field:** 253 samples (fully covers the 200-sample input window)
* **Inference Speed:** 6.89 ms on CPU (perfect for real-time control)

---

## 📊 Results & Screenshots

### 1. Model Accuracy
The model was evaluated on a completely unseen 15% Test Dataset:
* **R² Score:** 0.988
* **Phase Error:** 0.0°
* **Mean Absolute Error (MAE):** 109.31 Pa

<p align="center">
  <img src="artifacts/pred_vs_actual_scatter.png" width="600"/>
  <br><em>Actual vs Predicted Pressure Plot</em>
</p>

### 2. Time-Domain Cancellation
When the predicted anti-wave is injected into the combustion chamber, it achieves a **19.55 dB reduction** in RMS pressure (amplitude reduced by ~9.5x).

<p align="center">
  <img src="artifacts/cancellation_timedomain.png" width="800"/>
</p>

### 3. Frequency-Domain Cancellation
The Rayleigh feedback loop is completely broken, achieving a **23.06 dB drop** in acoustic power at the dangerous 1000 Hz frequency mode.

<p align="center">
  <img src="artifacts/per_freq_reduction.png" width="700"/>
</p>

---

## 🚀 How to Run the Project

**Step 1: Clone the Repository**
```bash
git clone https://github.com/YOUR-USERNAME/SUPRA.git
cd SUPRA
