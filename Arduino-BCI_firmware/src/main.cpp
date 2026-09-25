#include <Arduino.h>
#include <Servo.h>

Servo myServo;

// put function declarations here:
void MoveServo(bool activateServo);
int readEEGData();

struct EEGData {
  int sensorValue; // Placeholder for EEG data value
  byte packet[4]; // Packets the data into 16 bits with start and stop bits for serial transmission
};



const int servoPin = 5; // Pin to which the servo is connected
const int analogPin = A0; // Analog pin to read the biomedical data from the SHIELD
const int correctLed = 7; // Pin for the correct LED (built-in LED on Arduino Uno)
const int piezoPin = 13;

void setup() {
  myServo.attach(servoPin); // Attach servo to the specified pin
  myServo.write(0); // Initialize the servo position to 0 degrees
  Serial.begin(115200); // Start serial communication at 115200 baud
  pinMode(analogPin, INPUT); // Set the analog pin as an input
  pinMode(correctLed, OUTPUT); // Set the correct LED pin as an output
  pinMode(piezoPin, OUTPUT);

  while (true) {
    if (Serial.available() > 0) { // Check if data is available on the serial port
      char handshake = Serial.read();
      if (handshake == 'R') {    // Wait for the correct handshake byte 'R' from the Python script
        break;        // Exit the loop if the correct handshake is received
      }
    }

  }
}


const int calibrationTime = 5; // Duration of the calibration phase in seconds
const int frequency = 250; // Frequency of the calibration readings in Hz

unsigned long currentMicros = 0; // Variable to store the current time in microseconds
unsigned long previousMicros = 0; // Variable to store the previous time in microseconds

void loop() {


  if (Serial.available() > 0) {
    char command = Serial.read(); // Read the command from the serial input
    switch (command) {
      case 'A': // Command to activate the servo
        MoveServo(true); // Move the servo to 90 degrees
        break;
      case 'D': // Command to deactivate the servo
        MoveServo(false); // Move the servo back to 0 degrees
        break;
      case 'O': // Eyes Open (Single high beep)
        tone(piezoPin, 1000, 500); 
        break;
      case 'C': // Eyes Closed (Double low beep)
        tone(piezoPin, 800, 300); 
        delay(400); // Quick pause
        tone(piezoPin, 800, 300);
        break;
      case 'F': // Calibration Finished (Long high beep)
        tone(piezoPin, 1200, 1000);
        digitalWrite(correctLed, HIGH); // Turn on the correct LED 
        break;
    }
  }
  
  currentMicros = micros(); // Get the current time in microseconds
  

  
  if(currentMicros - previousMicros >= (1000000 / frequency)) { // Check if it's time for the next reading
    previousMicros = currentMicros; // Update the previous time
    EEGData data;
    data.sensorValue = readEEGData(); // Read EEG data from the SHIELD
    
    data.packet[0] = 0xA5; // Start byte
    data.packet[1] = (data.sensorValue >> 8) & 0xFF; // High byte of the sensor value
    data.packet[2] = data.sensorValue & 0xFF; // Low byte
    data.packet[3] = 0x5A; // Stop byte

    Serial.write(data.packet, sizeof(data.packet)); // Send the packet over serial
  }
  
}



void MoveServo(bool activateServo) {
  // Function to control the servo movement
  if (activateServo) {
    myServo.write(90); // Move the servo to 90 degrees
  }
  else {
    myServo.write(0); // Move the servo back to 0 degrees
  }
}


int readEEGData() {
  // Placeholder function to read EEG data from the SHIELD
  // This function should be implemented to return actual EEG data
  int sensorValue = analogRead(analogPin); // Read the analog value from the SHIELD
  return sensorValue; // Return the read value
}