import json
import os
import queue
import threading
import time

import numpy as np
import pyqtgraph as pg
import scipy.signal as signal
import serial
import csv

from eye_state_module import SharedEyeState, EyeStateThread, EyeEvent

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

port_name = "/dev/ttyACM0"


class EEGData:
    def __init__(self, frequency=250, duration=15, shared_eye_state=None, t0=None):

        self.frequency = frequency
        self.rec_len = 5 * 60 * self.frequency
        self.rec = np.zeros(self.rec_len)          # raw EEG values
        self.rec_eye = np.zeros(self.rec_len)       # eye state (0/1) at the same sample
        self.rec_time = np.zeros(self.rec_len)      # timestamp (perf_counter relative to t0)
        self.rec_idx = 0
        self.record = False
        self.duration = duration
        self.data = np.zeros(self.duration * self.frequency)
        self.data_index = 0
        self.main_timer = 0
        self.state = 'D'
        self.past_state = 'D'
        self.state_refractory_time = time.perf_counter()
        self.state_refractory_period = 2

        self.psd_hist_closed = np.zeros(15)
        self.psd_closed_idx = 0
        self.flag_closed_full = False

        self.psd_hist_open = np.zeros(15)
        self.psd_open_idx = 0
        self.flag_open_full = False

        self.freq_low = 8
        self.freq_high = 12
        self.bandpass_filter = signal.butter(4, [self.freq_low, self.freq_high], btype='bandpass',
                                              fs=self.frequency, output='sos')

        self.serial_port = serial.Serial(port_name, baudrate=115200, timeout=1 / self.frequency)
        self.calibration_threshold = 0

        # --- Eye-state integration ---
        # t0 is the COMMON time base shared between the EEG/GUI thread and the
        # video thread: both compute their own timestamps as
        # time.perf_counter() - t0, so they stay comparable even though they
        # run at different frequencies (250 Hz vs ~15 fps). perf_counter() is
        # monotonic and unaffected by system clock adjustments, which makes
        # it the right choice for synchronizing two data sources within the
        # same process.
        self.shared_eye_state = shared_eye_state
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.eye_events = []          # completed prolonged-closure events
        self._event_queue = None      # set by main() if available
        self.servo_events = []        # (timestamp_s, state) whenever a command is actually sent to the servo

    def read_eeg_data(self):
        sensor_value = getattr(self, 'last_sensor_value', 300)  # Default to last value if no new data is read
        if self.serial_port.is_open:
            while self.serial_port.in_waiting < 4:
                pass
            try:
                if self.serial_port.read(1) == b'\xA5':
                    packet = self.serial_port.read(3)
                    high_byte, low_byte, stop_byte = packet
                    if stop_byte == 0x5A:
                        sensor_value = (high_byte << 8) | low_byte
            except serial.SerialException:
                print("Error reading from serial port. Please check the connection with Arduino.")
                self.serial_port.close()
                self.serial_port = None

        return sensor_value

    def _current_eye_state(self):
        if self.shared_eye_state is None:
            return 0
        state, _, _ = self.shared_eye_state.get()
        return state

    def update_data(self):
        new_value = self.read_eeg_data()
        ts = time.perf_counter() - self.t0
        eye_state = self._current_eye_state()

        if self.record and self.rec_idx < self.rec_len:
            self.rec[self.rec_idx] = new_value
            self.rec_eye[self.rec_idx] = eye_state
            self.rec_time[self.rec_idx] = ts
            self.rec_idx += 1

        if self.data_index >= len(self.data):
            self.data[:-1] = self.data[1:]
            self.data[-1] = new_value
        else:
            self.data[self.data_index] = new_value
            self.data_index += 1

        return self.data

    def filter_data(self, filt_type):
        if filt_type == "bandpass":
            return signal.sosfilt(self.bandpass_filter, self.data)

    def compute_psd(self, window_size=2):
        data_filtered = self.filter_data("bandpass")
        samples = int(window_size * self.frequency)
        recent_data = data_filtered[-samples:]
        axis_freq, data_psd = signal.welch(recent_data, fs=self.frequency, nperseg=250)
        alpha_band = (axis_freq >= self.freq_low) & (axis_freq <= self.freq_high)
        print(np.mean(data_psd[alpha_band]))
        return axis_freq, data_psd

    def compute_alpha_thresh(self):
        data_filtered = self.filter_data("bandpass")
        psd = np.zeros(11)
        for i in range(0,11):
            axis_freq, psd_window = signal.welch(data_filtered[(i+3)*self.frequency:(i+5)*self.frequency], fs=self.frequency, nperseg = 250)
            alpha_mask = (axis_freq >= self.freq_low) & (axis_freq <= self.freq_high)
            psd[i] = np.mean(psd_window[alpha_mask])
        return psd

    def set_threshold(self, psd_low, psd_high):

        psd_median_low = np.median(psd_low)
        psd_median_high = np.median(psd_high)

        psd_mad_low = np.median(np.absolute(psd_low - psd_median_low))

        geometric_threshold = np.sqrt(psd_median_high*psd_median_low)
        statistical_threshold = np.mean(psd_low) + 3*np.std(psd_low)

        Coefficient_median = psd_mad_low / psd_median_low

        C_low = 0.1
        C_high = 0.4

        alpha_raw = (Coefficient_median - C_low) / (C_high - C_low)
        alpha = max(0.0, min(1.0, alpha_raw))
        print(alpha_raw, Coefficient_median)
        if psd_median_high <= psd_median_low:
            raise ValueError("High PSD should not be lower than low PSD")
        if psd_median_high < 1.5 * psd_median_low:
            print("High PSD is very weak, check electrodes!")
        print(alpha*statistical_threshold + (1-alpha)*geometric_threshold)
        return (alpha)*statistical_threshold + (1-alpha)*geometric_threshold

    def calibration_phase(self):

        print("Calibration Phase: Please keep your eyes open for 15 seconds.")
        state = 'O'
        self.serial_port.write(state.encode())
        self.serial_port.reset_input_buffer()
        for i in range(0,self.duration*self.frequency,1):
            self.update_data()

        psd_low = self.compute_alpha_thresh()


        time.sleep(1)
        
        print("Calibration Phase: Please keep your eyes closed for 15 seconds.")
        state = 'C'
        self.serial_port.write(state.encode())
        self.serial_port.reset_input_buffer()
        for i in range(0,self.duration*self.frequency,1):
            self.update_data()

        psd_high = self.compute_alpha_thresh()

        self.calibration_threshold = self.set_threshold(psd_low, psd_high)

        time.sleep(1)
        print("Calibration completed!")
        self.serial_port.write(b'F')
        return self.calibration_threshold

    def drain_eye_events(self):
        """
        To be called periodically from the main thread (e.g. inside
        update_interface): drains the discrete-event queue populated by
        EyeStateThread and accumulates it for the final log. The
        ("closure_start", ts) tuples only mark the beginning of a closure
        still in progress and are ignored here; only complete EyeEvent
        tuples (start, end, duration) are counted.
        """
        if self._event_queue is None:
            return
        while True:
            try:
                item = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, EyeEvent):
                self.eye_events.append(item)

    def save_recording(self):
        print("\nSaving 5-minute recording in the background...")
        filename = 'New_Recording.csv'
        n = min(self.rec_idx, self.rec_len)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp_s", "eeg_value", "eye_state_closed"])
            for t, value, eye in zip(self.rec_time[:n], self.rec[:n], self.rec_eye[:n]):
                writer.writerow([f"{t:.4f}", value, int(eye)])
        print(f"Successfully saved {n} samples to {filename}.")

        if self.eye_events:
            events_filename = 'eye_closure_events.csv'
            with open(events_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["start_s", "end_s", "duration_s"])
                for ev in self.eye_events:
                    writer.writerow([f"{ev.start_ts:.4f}", f"{ev.end_ts:.4f}", f"{ev.duration:.4f}"])
            print(f"Saved {len(self.eye_events)} prolonged-closure events to {events_filename}.")

        if self.servo_events:
            servo_filename = 'servo_activation_log.csv'
            with open(servo_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp_s", "state_sent"])
                for ts, state in self.servo_events:
                    writer.writerow([f"{ts:.4f}", state])
            print(f"Saved {len(self.servo_events)} servo activation events to {servo_filename}.")

    def update_psd_history(self, state, psd_new):
        
        match state:
            case 'D':
                self.psd_hist_open[self.psd_open_idx] = psd_new
                if self.psd_open_idx <=13: 
                    self.psd_open_idx +=1 

                else: 
                    self.psd_open_idx = 0
                    self.flag_open_full = True

            case 'A':
                self.psd_hist_closed[self.psd_closed_idx] = psd_new
                if self.psd_closed_idx <=13: 
                    self.psd_closed_idx +=1 

                else: 
                    self.psd_closed_idx = 0
                    self.flag_closed_full = True

        if self.flag_closed_full and self.flag_open_full:
            self.calibration_threshold = self.set_threshold(self.psd_hist_open, self.psd_hist_closed)
            print(f"Threshold successfully updated to: {self.calibration_threshold} at second {self.main_timer/self.frequency}")
            self.flag_open_full = False
            self.flag_closed_full = False

    def update_interface(self, curve):
        self.main_timer += 1

        self.update_data()
        curve.setData(self.data)
        self.drain_eye_events()

        if self.main_timer % 250 == 0:
            axis_freq, psd = self.compute_psd()
            psd_alpha = np.mean(psd[(axis_freq >= self.freq_low) & (axis_freq <= self.freq_high)])
            raw_new_state = check_eyes_state(psd_alpha, self.calibration_threshold)

            current_time = time.perf_counter()

        
            if (current_time - self.state_refractory_time) < self.state_refractory_period:
                new_state = self.state 
            else:
                new_state = raw_new_state

            state_confirmed = False

            if self.serial_port and self.serial_port.is_open and new_state == self.state:
                if new_state != self.past_state:
                    self.state_refractory_time = current_time  # Log the exact actuation time
                    try:
                        self.serial_port.write(new_state.encode())
                        servo_ts = current_time - self.t0
                        self.servo_events.append((servo_ts, new_state))
                        
                        if new_state == 'A':
                            curve.setPen('r')
                        else:
                            curve.setPen('y')
                    except serial.SerialException:
                        pass

                state_confirmed = True    

            elif new_state == self.past_state:
                self.state = self.past_state

            if state_confirmed:
                self.update_psd_history(new_state, psd_alpha)

            self.past_state = self.state
            self.state = new_state

        if self.main_timer == self.rec_len:
            import threading as _threading
            _threading.Thread(target=self.save_recording).start()
            self.record = False


def check_eyes_state(psd, threshold):
    if psd < threshold:
        return 'D'
    else:
        return 'A'


def load_eye_calibration(path="eye_calibration.json"):
    """Load EAR thresholds saved by calibration_ear.py, if present."""
    try:
        with open(path) as f:
            cfg = json.load(f)
        print(f"Loaded eye calibration from {path}: {cfg}")
        return cfg
    except FileNotFoundError:
        print(f"No {path} found, using default EAR thresholds. "
              f"Run calibration_ear.py first for better accuracy.")
        return {}


def main():
    # Common time base for synchronizing the EEG loop and the video thread
    temp_t0 = time.perf_counter()

    eye_cfg = load_eye_calibration()
    shared_eye_state = SharedEyeState()
    eye_stop_event = threading.Event()
    eye_event_queue = queue.Queue()


    eeg_data = EEGData(frequency=250, duration=15, shared_eye_state=shared_eye_state, t0=temp_t0)
    eeg_data._event_queue = eye_event_queue

    print("\nConnecting to Arduino...")
    print("Connected to Arduino on port:", eeg_data.serial_port)
    print("\nLoading...")

    time.sleep(3)

    eeg_data.serial_port.write(b'R')
    print("Handshake sent to Arduino. Starting calibration phase...")

    eeg_data.calibration_threshold = eeg_data.calibration_phase()
    eeg_data.main_timer = 0

    print("Loading...")
    for i in range(5 * eeg_data.frequency):
        eeg_data.update_data()

    print("\nReady!")

    true_t0 = time.perf_counter()
    eeg_data.t0 = true_t0



    eye_thread = EyeStateThread(
        shared_state=shared_eye_state,
        t0=true_t0,
        event_queue=eye_event_queue,
        stop_event=eye_stop_event,
        ear_close_threshold=eye_cfg.get("ear_close_threshold", 0.21),
        ear_open_threshold=eye_cfg.get("ear_open_threshold", 0.24),
        closed_duration_threshold=eye_cfg.get("closed_duration_threshold", 1.5),
        target_fps=15,
    )
    
    eye_thread.start()


    eeg_data.record = True
    app = pg.mkQApp("EEG Data Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="EEG Data Plot")
    plot = win.addPlot(title="EEG Signal")
    curve = plot.plot(pen='y')

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(lambda: eeg_data.update_interface(curve))
    timer.start(1000 // eeg_data.frequency)

    try:
        pg.exec()
    finally:
        eye_stop_event.set()
        eye_thread.join(timeout=2.0)
        # If the window was closed before the 5-minute recording finished,
        # update_interface never triggered save_recording(): save whatever
        # was captured so far (EEG, eye-state and servo activation logs)
        # instead of discarding it.
        if eeg_data.record:
            eeg_data.record = False
            eeg_data.save_recording()


if __name__ == '__main__':
    main()
