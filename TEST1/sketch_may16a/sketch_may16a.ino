/*
   AD8232 ECG Monitor
   Optimized for Arduino Serial Plotter
*/

const int ECG_PIN = A0;
const int LO_PLUS = 10;
const int LO_MINUS = 11;

void setup() {

  // Faster baud rate for smoother waveform
  Serial.begin(115200);

  // Lead-off detection pins
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
}

void loop() {

  // Check whether electrodes are disconnected
  if ((digitalRead(LO_PLUS) == 1) ||
      (digitalRead(LO_MINUS) == 1)) {

    // Flat middle value if leads disconnected
    Serial.println(512);
  }
  else {

    // Read ECG signal
    int ecgValue = analogRead(ECG_PIN);

    // Send to Serial Plotter
    Serial.println(ecgValue);
  }

  // Very small delay for stable plotting
  delayMicroseconds(500);
}