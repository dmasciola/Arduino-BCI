# Real-Time Adaptive Alpha-Wave Brain-Computer Interface

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)[cite: 6]
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)[cite: 6]

## Overview
This repository contains a robust, real-time Brain-Computer Interface (BCI) designed to detect Alpha wave (8-12 Hz) blocking (the Berger effect) and drive external hardware. Built on a Python digital signal processing (DSP) backend and an Arduino-controlled hardware layer, this system utilizes an isolated electromechanical actuation loop, a multimodal computer vision synchronization thread, and dynamic statistical thresholding to adapt to the unique baseline neurophysiology of each individual user.

## Hardware Architecture & Noise Mitigation
This project mitigates electromechanical feedback and baseline noise at the hardware level before applying software filters.
*   **Acquisition:** EEG data is captured via an OLIMEX SHIELD EKG-EMG biosignal amplifier at a 250Hz sampling rate. The analog pipeline features an Instrumental Amplifier and a Besselworth 40Hz Lowpass Filter running into an OAmp, providing a total gain of $G_{\text{total}} = 2848$.
*   **Electrodes:** Gold-plated (Au) electrodes are utilized for durability and low impedance, secured with KONIX EEG Paste. Placement follows a 3-channel configuration: Oz (Channel +), Mastoid (Channel -), and Fz (Ground/Reference).
*   **Actuation:** An Arduino UNO handles physical state changes (e.g., servo movement), converting the analog stream to a digital 4-byte packet over serial. 
*   **Noise Isolation:** To prevent PWM-induced ground bounce and electrical noise from contaminating the EEG signal, the hardware utilizes an isolated common-ground power supply powered by an independent 6V battery.

## Digital Signal Processing Pipeline
The software backend relies on continuous windowed analysis to extract Alpha power from the raw signal.
1.  **Filtering:** A 4th-order Butterworth bandpass filter (8-12 Hz) isolates the Alpha band.
2.  **Power Spectral Density (PSD):** Welch's method computes the PSD in real-time[cite: 6] on 2-second windows. 
3.  **Multimodal Synchronization:** A parallel computer vision thread utilizing `mediapipe` Face Mesh calculates the Eye Aspect Ratio (EAR) at 15 FPS to establish an objective ground truth for eye closures. The video feed and the 250 Hz EEG stream are synchronized offline via a common monotonic hardware clock (`time.perf_counter()`).
4.  **State Machine (Artifact Rejection):** To mitigate servo twitching, the state evaluation system requires three stable windows: the servo only actuates if $P_{\text{new}} == P_{\text{current}}$ AND $P_{\text{new}} \neq P_{\text{past}}$. A 2-second debounce refractory period is implemented to protect the Arduino serial buffer from jitter and prevent rapid, erroneous state switching[cite: 6].

## Dynamic Calibration & Statistical Thresholding
To account for physiological variance between different users, the BCI does not rely on a static, hardcoded amplitude threshold[cite: 6]. It features an automated initialization phase and a continuous rolling baseline:
*   **Baseline Establishment:** The system records a 15-second "Eyes Opened" baseline and a 15-second "Eyes Closed" activation state[cite: 6]. The first 3 seconds of each phase are skipped to ignore transitional movement artifacts.
*   **Hybrid Median-MAD Thresholding:** The final transition threshold ($T$) is determined by a linear combination of a geometric mean ($T_2$) and a 3-Sigma confidence interval ($T_1$)[cite: 6]. The weight ($\alpha$) is derived from the Median Absolute Deviation (MAD) of the user's active state, bounded by empirical constants ($C_L = 0.1$, $C_H = 0.4$):
$$T_1 = m_{\text{active}} + 3\cdot s_{\text{active}}$$
$$T_2 = \sqrt{M_{\text{active}}\cdot M_{\text{rest}}}$$
$$C = \frac{\operatorname{median}(\vert{}PSD_{\text{active}}-M_{\text{active}}\vert{})}{M_{\text{active}}}$$
$$\alpha=\frac{C-C_L}{C_H-C_L}$$
$$T = \alpha T_1+(1-\alpha)T_2$$
*   **Rolling Baseline:** The system continuously tracks the last 15 valid samples of high PSD and low PSD to evaluate the threshold dynamically from scratch, adapting to physiological habituation over time.

## Offline Validation
Extensive offline MATLAB analysis validates the real-time pipeline[cite: 6]. Over five empirical validation trials, the system demonstrated:
*   **Signal-to-Noise Ratio (SNR):** Average linear SNR of 3.05 (4.62 dB).
*   **Precision:** 81.5% correct actuations over total actuations.
*   **Latency:** A median actuation latency of 3.91 seconds, representing the deliberate mathematical aggregation of the biological habituation time, the 2-second Welch periodogram duration, tick quantization, and the temporal debounce mechanism.
*   **Event-Detection Accuracy:** 100% True Positive catch rate. The hybrid thresholding architecture successfully identified every intended physiological state change across all trials.

## Installation & Usage


**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/dmasciola/Arduino-BCI.git
cd Arduino-BCI
```

**2. Create a virtual environment and install dependencies**
*On Ubuntu/macOS:*

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

*On Windows:*

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r python/requirements.txt
```

**3. Hardware Setup:**
* Connect the Olimex shield and ensure electrodes are properly placed (e.g. Oz for Alpha detection).
* Connect the Arduino via USB and verify the serial port assignment in `python/data_analysis.py` (default: `/dev/ttyACM0`).

**4. Run the BCI:**
```bash
python python/data_analysis.py
```
*Follow the terminal prompts to complete the 10-second calibration phase before the PyQtGraph GUI initializes.*

## License
This project is licensed under the GNU General Public License v3.0 - see the **LICENSE** file for details. Copyright (c) 2026 Davide Masciola.
