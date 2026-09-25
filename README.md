# Real-Time Adaptive Alpha-Wave Brain-Computer Interface

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview
This repository contains a robust, real-time Brain-Computer Interface (BCI) designed to detect Alpha wave (8-12 Hz) blocking (the Berger effect) and drive external hardware. Built on a Python digital signal processing (DSP) backend and an Arduino-controlled hardware layer, this system utilizes dynamic statistical thresholding to adapt to the unique baseline neurophysiology of each individual user.

## Hardware Architecture
This project mitigates electromechanical feedback and baseline noise at the hardware level before applying software filters.
*   **Acquisition:** EEG data is captured via an Olimex shield at a 250Hz sampling rate.
*   **Actuation:** An Arduino handles the physical state changes (e.g., servo movement).
*   **Noise Isolation:** To prevent PWM-induced ground bounce and electrical noise from contaminating the EEG signal, the hardware utilizes an isolated common-ground power supply powered by an independent 6V battery.

## Digital Signal Processing Pipeline
The software backend relies on continuous windowed analysis to extract Alpha power from the raw signal.
1.  **Filtering:** A 4th-order Butterworth bandpass filter (8-12 Hz) isolates the Alpha band.
2.  **Power Spectral Density (PSD):** Welch's method computes the PSD in real-time. To ensure accurate averaging and to avoid initial filter transient ringing, the system operates on a 5-second rolling buffer, utilizing a 4-second processing window for calibration and a 2-second overlapping window for real-time inference.
3.  **State Machine:** A 2-second debounce refractory period is implemented to protect the Arduino serial buffer from jitter and prevent rapid, erroneous state switching.

## Dynamic Calibration & Statistical Thresholding
To account for physiological variance between different users, the BCI does not rely on a static, hardcoded amplitude threshold. Instead, it features an automated initialization phase:
*   **Baseline Establishment:** The system records a 5-second "Eyes Opened" baseline and a 5-second "Eyes Closed" activation state.
*   **3-Sigma Confidence Interval:** The transition threshold is calculated as the `mean + 3*standard_deviation` of the user's resting Alpha power. This strict >99% confidence interval ensures the hardware is only triggered by genuine Alpha spindles, heavily suppressing false positives from eye blinks or muscle tension.

## Offline Validation
Extensive offline analysis validates the real-time pipeline. Using continuous spectrogram generation (accessible in the `examples/` directory via Jupyter Notebooks), the dynamic calibration successfully generalized across multiple test subjects, consistently achieving a Signal-to-Noise Ratio (SNR) of 4.00 and adapting seamlessly to users with differing baseline Alpha amplitudes.

## Installation & Usage

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/DavideMasciola/Arduino-BCI.git](https://github.com/DavideMasciola/Arduino-BCI.git)
cd Arduino-BCI
pip install -r requirements.txt
```

**2. Hardware Setup:**

* Connect the Olimex shield and ensure electrodes are properly placed (e.g. Oz for Alpha detection).
* Connect the Arduino via USB and verify the serial port assignment in `data_analysis.py` (default: `/dev/ttyACM0`).

**3. Run the BCI:**

```bash
python data_analysis.py
```
*Follow the terminal prompts to complete the 10-second calibration phase before the PyQtGraph GUI initializes.*

## License
This project is licensed under the GNU General Public License v3.0 - see the **LICENSE** file for details. Copyright (c) 2026 Davide Masciola.