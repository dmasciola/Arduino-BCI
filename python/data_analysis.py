import numpy as np
import pyqtgraph as pg
import serial
import time
import scipy.signal as signal
import csv


port_name = "/dev/ttyACM0"

class EEGData:
    def __init__(self, frequency=250, duration=5):

        self.frequency = frequency  # Sampling frequency in Hz
        self.rec_len = 5*60*self.frequency
        self.rec = np.zeros(self.rec_len)   # Recording
        self.rec_idx = 0
        self.record = False   # File set to recording
        self.duration = duration  # Duration of data collection in seconds
        self.data = np.zeros(self.duration * self.frequency)  # Initialize an empty array to store EEG data
        self.data_index = 0  # Index to keep track of the current position in the data array
        self.main_timer = 0  # Timer to keep track of elapsed time in samples
        self.state = 'D'    # State (hypotesis: open)
        self.past_state = 'D'   # Past state


        self.freq_low = 8  # Lower frequency bound for the Alpha band
        self.freq_high = 12  # Upper frequency bound for the Alpha band
        self.bandpass_filter = signal.butter(4, [self.freq_low, self.freq_high], btype='bandpass', fs=self.frequency, output='sos')
        #self.notch_filter = signal.butter(4, [49, 51], btype='bandstop', fs=self.frequency, output='sos')
         
        self.serial_port = serial.Serial(port_name, baudrate=115200, timeout=1/self.frequency)
        self.calibration_threshold = 0  

    def read_eeg_data(self):
        """
        Reads EEG data from the specified serial port.
        
        Args:
            serial_port (str): The serial port to read from (e.g., 'COM3' or '/dev/ttyUSB0').
        """
        sensor_value = 0 
        if self.serial_port.is_open:
            while self.serial_port.in_waiting < 4:
                pass
            try:
                if self.serial_port.read(1) == b'\xA5':  # Check starting byte
                    packet = self.serial_port.read(3)
                    high_byte, low_byte, stop_byte = packet

                    if stop_byte == 0x5A:  # Validate and stop byte
                        sensor_value = (high_byte << 8) | low_byte  

            except serial.SerialException:
                print("Error reading from serial port. Please check the connection with Arduino.")
                self.serial_port.close()  
                self.serial_port = None   
               
        #print(sensor_value)
        return sensor_value
    
    def update_data(self):
        """
        Cleans old data from the EEG data array to maintain a maximum length.
        
        Args:
            data (np.ndarray): The EEG data array.
            serial_port (serial.Serial): The serial port object.
        """

        new_value = self.read_eeg_data()

        if self.record and self.rec_idx < self.rec_len: 
            self.rec[self.rec_idx] = new_value
            self.rec_idx += 1
        

        if self.data_index >= len(self.data):
            self.data[:-1] = self.data[1:]
            self.data[-1] = new_value  # Shift data to the left and add new EEG data
        else:
            self.data[self.data_index] = new_value  # Add new EEG data to the array
            self.data_index += 1

        return self.data 

    def filter_data(self, filt_type):
        """
        Applies filter to the EEG data to isolate the Alpha band (8-12 Hz).
        
        Args:
            data (np.ndarray): The EEG data array.
        """
        if filt_type == "bandpass":
            return signal.sosfilt(self.bandpass_filter, self.data)
        elif filt_type == "notch":
            return signal.sosfilt(self.notch_filter, self.data)

    def compute_psd(self,window_size = 2):
        """
        Computes the Power Spectral Density for the given EEG data in the Alpha band.
        
        Args:
            data (np.ndarray): The EEG data array.
        """
        data_filtered = self.filter_data("bandpass")
        samples = int(window_size * self.frequency)
        recent_data = data_filtered[-samples:]
        axis_freq, data_psd = signal.welch(recent_data, fs=self.frequency, nperseg = 250)
        #------------------------------------------------
        alpha_band = (axis_freq >= self.freq_low) & (axis_freq <= self.freq_high)
        print(np.mean(data_psd[alpha_band]))
        #------------------------------------------------
        return axis_freq, data_psd 

    def set_treshold(self, frequency, psd_low, psd_high):
        """
        Sets the threshold for the Power Spectral Density in the Alpha band. The treshold is used to
        determine whether the user has their eyes open or closed based on the EEG data.

        
        Args:
            frequency (np.ndarray): The frequency array.
            psd_low (float): The lower threshold for PSD.
            psd_high (float): The upper threshold for PSD.
        """
        alpha_band = (frequency >= self.freq_low) & (frequency <= self.freq_high)  # Identify the Alpha band mask to have a more precise psd
        psd_mean_low = np.mean(psd_low[alpha_band])
        psd_mean_high = np.mean(psd_high[alpha_band])
        psd_std_low = np.std(psd_low[alpha_band])
        #------------------------------------------------
        if psd_mean_high <= psd_mean_low:
            raise ValueError("High PSD should not be lower than low PSD")
        
        if psd_mean_high < 1.5*psd_mean_low:
            print("High PSD is very weak, check electrodes!")
            pass
        #------------------------------------------------
        print(psd_mean_low + 3*psd_std_low)
        return psd_mean_low + 3*psd_std_low # Treshold is defined to minimize errors in output establishment

    def calibration_phase(self):
        """
        Performs the calibration phase to set the threshold for the Power Spectral Density in the Alpha band.
        The user is prompted to keep their eyes open and closed for a few seconds each to collect EEG data.
        """

        print("Calibration Phase: Please keep your eyes open for 5 seconds.")
        timer = 0
        state = 'O'
        self.serial_port.write(state.encode())
        while timer < 5:

            self.update_data()
            timer += 1/self.frequency  # Increment the timer based on the sampling frequency
        axis_freq, psd_low = self.compute_psd(window_size = 4) # Throws away the first second which can contain artifacts

        time.sleep(1)  # Wait for 1 second before starting the next phase

        print("Calibration Phase: Please keep your eyes closed for 5 seconds.")
        timer = 0    
        state = 'C'
        self.serial_port.write(state.encode())
        while timer < 5:

            self.update_data()  
            timer += 1/self.frequency  # Increment the timer based on the sampling frequency
        psd_high = self.compute_psd(window_size = 4)[1]  # Throws away the first second which can contain artifacts

        self.calibration_threshold = self.set_treshold(axis_freq, psd_low, psd_high)  # Set the threshold based on collected data

        time.sleep(1)  # Wait for 1 second before starting the main loop

        print("Calibration completed!")
        self.serial_port.write(b'F')

        return self.calibration_threshold  # Return the computed threshold        

    def save_recording(self):
        print("\nSaving 5-minute recording in the background...")
        filename = 'New_Recording.csv'
        with open(filename, 'w', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            for value in self.rec:
                writer.writerow([value])
        print(f"Successfully saved {self.rec_len} samples to {filename}.")

    def update_interface(self, curve):
            """
            Updates the user interface with the current time.
            
            Args:
                data (np.ndarray): The EEG data array.
                time (float): The current time in seconds.
            """
    
            self.main_timer += 1 
    
            self.update_data() 
            curve.setData(self.data)  
    
    
            if self.main_timer % 250 == 0:  # Check the eyes state every 250 samples (1 second)

                axis_freq, psd = self.compute_psd()  
                psd_alpha = np.mean(psd[(axis_freq >= self.freq_low) & (axis_freq <= self.freq_high)])   
                new_state = check_eyes_state(psd_alpha, self.calibration_threshold)  # Check the state


                if self.serial_port and self.serial_port.is_open and new_state == self.state:

                    if new_state != self.past_state:
                        try:
                            self.serial_port.write(new_state.encode())  # Send the state to the Arduino

                            # GUI Color change
                            if new_state == 'A': curve.setPen('r')
                            else: curve.setPen('y')
                            
                        except serial.SerialException:
                            pass
                self.past_state = self.state
                self.state = new_state
                #self.main_timer = 0  # Reset time

            if self.main_timer == self.rec_len: 

                import threading

                threading.Thread(target=self.save_recording).start()

                self.record = False


def check_eyes_state(psd, threshold):
    """
    Checks the state of the user's eyes based on the Power Spectral Density and the set threshold.
    
    Args:
        psd (float): The computed Power Spectral Density.
        threshold (float): The threshold for determining eyes state.
    """
    if psd < threshold:
        #print("open")
        return 'D'
    else:
        #print("closed")
        return 'A'

def main():

    eeg_data = EEGData(frequency=250, duration=5)  # Create an instance of the EEGData class

    print("\nConnecting to Arduino...")

    print("Connected to Arduino on port:", eeg_data.serial_port)
    print("\nLoading...")

    time.sleep(3)  # Wait for 3 seconds to allow the serial connection to stabilize

    eeg_data.serial_port.write(b'R')  # Send the handshake byte 'R' to the Arduino to initiate data transmission

    print("Handshake sent to Arduino. Starting calibration phase...")
    
    eeg_data.calibration_threshold = eeg_data.calibration_phase() 
    eeg_data.main_timer = 0  # Initialize the main timer

    print("Loading...")

    for i in range(5*eeg_data.frequency):

       eeg_data.update_data()  # Update the EEG data array for the initial 5 seconds

    print("\nReady!")

    eeg_data.record = True
    app = pg.mkQApp("EEG Data Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="EEG Data Plot")
    plot = win.addPlot(title="EEG Signal")
    curve = plot.plot(pen='y')

    

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(lambda: eeg_data.update_interface(curve))  # Connect the timer to the update_interface function
    timer.start(1000 // eeg_data.frequency)

    pg.exec()  # Start the PyQtGraph event loop
    
if __name__ == '__main__':
    main()