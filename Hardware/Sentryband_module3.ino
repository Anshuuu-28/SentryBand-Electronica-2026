/*
 * SentryBand -- Module 3 embedded firmware (Wokwi simulation).
 *
 * Ports src/fusion.py's fuseState() and src/alerts.py's alert behavior
 * into real embedded C++, running on a simulated MPU6050 accelerometer
 * (fall detection) + potentiometer (heart-rate proxy) in Wokwi. This
 * demonstrates the DECISION LOGIC validated in Python (see Modules 1-2:
 * real SisFall/PPG-DaLiA data, real int8 .tflite models) actually
 * running on a microcontroller -- closing the deck's Slide 14 claim
 * ("Working Prototype ... running fully offline on a dev board").
 *
 * SCOPE NOTE (documented honestly, matching the rest of this project):
 * this ports the FUSION/decision state machine, not the full ML feature
 * extraction pipeline. Replicating the exact spectral/statistical
 * features from src/features.py in embedded C is out of scope for this
 * hardware demo stage -- and Wokwi's simulated sensors don't produce
 * physically realistic fall/arrhythmia waveforms anyway, so it wouldn't
 * add real validation value. Fall detection here uses a simple
 * magnitude-spike threshold (a real, common embedded heuristic); heart
 * alert uses a simple bpm range check matching the tachycardia/
 * bradycardia ranges documented in src/real_data_loader.py.
 *
 * Libraries needed (Wokwi should auto-detect from these #includes; if
 * not, add a libraries.txt alongside this file listing:
 *   Adafruit MPU6050
 *   Adafruit Unified Sensor
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

const int POT_PIN = A0;
const int LED_GREEN = 8;
const int LED_YELLOW = 9;
const int LED_RED = 10;
const int BUZZER_PIN = 11;

// Fall detection thresholds (simple magnitude-spike heuristic)
const float FALL_MAGNITUDE_THRESHOLD_G = 2.2;   // sudden spike above this = possible fall
const float FREE_FALL_THRESHOLD_G = 0.4;        // sudden dip near zero-g also flags a fall

// Heart alert thresholds -- matches the tachycardia (140-180 bpm) /
// bradycardia (30-45 bpm) ranges used when building the real heart-alert
// training data in src/real_data_loader.py
const int HR_LOW_BPM = 45;
const int HR_HIGH_BPM = 140;

// Exact state names from src/config.py, so Serial output matches the
// Python pipeline's vocabulary
const char* STATE_NORMAL = "Normal";
const char* STATE_POSSIBLE_FALL = "Possible Fall";
const char* STATE_HEART_ALERT = "Heart Alert";
const char* STATE_COMBINED_EMERGENCY = "Combined Emergency";

unsigned long lastBeepToggle = 0;
bool beepOn = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found -- check wiring.");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("SentryBand firmware ready -- fusion logic running on-device.");
  Serial.println("Drag the potentiometer to simulate heart rate; shake/flip");
  Serial.println("the MPU6050 in the Wokwi 3D view to simulate a fall.");
}

// Direct C++ port of src/fusion.py's fuse(fall_flag, heart_flag).
const char* fuseState(bool fallFlag, bool heartFlag) {
  if (fallFlag && heartFlag) return STATE_COMBINED_EMERGENCY;
  if (fallFlag) return STATE_POSSIBLE_FALL;
  if (heartFlag) return STATE_HEART_ALERT;
  return STATE_NORMAL;
}

// Mirrors src/alerts.py's simulate_buzzer_led(): silent for Normal,
// intermittent beep+flash for single-signal alerts, continuous
// beep+fast-flash for the most urgent Combined Emergency state.
void driveAlerts(const char* state) {
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, LOW);

  if (strcmp(state, STATE_NORMAL) == 0) {
    digitalWrite(LED_GREEN, HIGH);
    noTone(BUZZER_PIN);
    return;
  }

  unsigned long now = millis();
  unsigned int interval = (strcmp(state, STATE_COMBINED_EMERGENCY) == 0) ? 150 : 400;
  if (now - lastBeepToggle >= interval) {
    lastBeepToggle = now;
    beepOn = !beepOn;
  }

  if (strcmp(state, STATE_COMBINED_EMERGENCY) == 0) {
    digitalWrite(LED_RED, beepOn ? HIGH : LOW);
    if (beepOn) tone(BUZZER_PIN, 1500); else noTone(BUZZER_PIN);
  } else if (strcmp(state, STATE_POSSIBLE_FALL) == 0) {
    digitalWrite(LED_YELLOW, beepOn ? HIGH : LOW);
    if (beepOn) tone(BUZZER_PIN, 1000); else noTone(BUZZER_PIN);
  } else {  // Heart Alert
    digitalWrite(LED_YELLOW, beepOn ? HIGH : LOW);
    digitalWrite(LED_RED, beepOn ? HIGH : LOW);
    if (beepOn) tone(BUZZER_PIN, 1200); else noTone(BUZZER_PIN);
  }
}

// Mirrors src/alerts.py's simulate_ble_alert(): prints the JSON payload
// that would be sent over BLE to a paired phone/caregiver app -- state
// and derived signals only, never raw sensor data, matching the deck's
// "raw health data never leaves the wrist" privacy claim.
void sendBleAlert(const char* state, float accelMagnitude, int heartRateBpm) {
  if (strcmp(state, STATE_NORMAL) == 0) return;
  Serial.print("[BLE] {\"device\":\"SentryBand\",\"state\":\"");
  Serial.print(state);
  Serial.print("\",\"accel_magnitude_g\":");
  Serial.print(accelMagnitude, 2);
  Serial.print(",\"heart_rate_bpm\":");
  Serial.print(heartRateBpm);
  Serial.println("}");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // MPU6050 library reports m/s^2; convert to g for the threshold checks
  float ax = a.acceleration.x / 9.81;
  float ay = a.acceleration.y / 9.81;
  float az = a.acceleration.z / 9.81;
  float magnitude = sqrt(ax * ax + ay * ay + az * az);

  bool fallFlag = (magnitude > FALL_MAGNITUDE_THRESHOLD_G) ||
                  (magnitude < FREE_FALL_THRESHOLD_G);

  int potValue = analogRead(POT_PIN);
  Serial.print("[DEBUG] raw potValue=");   
  Serial.println(potValue); 
  int heartRateBpm = map(potValue, 0, 1023, 40, 180);
  bool heartFlag = (heartRateBpm < HR_LOW_BPM) || (heartRateBpm > HR_HIGH_BPM);

  const char* state = fuseState(fallFlag, heartFlag);

  driveAlerts(state);
  sendBleAlert(state, magnitude, heartRateBpm);

  Serial.print("[STATE] ");
  Serial.print(state);
  Serial.print("  |  accel_mag=");
  Serial.print(magnitude, 2);
  Serial.print("g  heart_rate=");
  Serial.print(heartRateBpm);
  Serial.println(" bpm");

  delay(100);
}
