/*
 * AI-Powered Smart Hydration Tracker
 * ESP32-CAM Firmware — Production Code
 *
 * Hardware:
 *   - AI Thinker ESP32-CAM
 *   - MPU-6050 (I2C: SDA=GPIO2, SCL=GPIO14)
 *   - HX711 Load Cell Amplifier (DT=GPIO13, SCK=GPIO12)
 *   - Buzzer (GPIO15)
 * ============================================================
 */

#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "HX711.h"

// ============================================================
// CONFIGURATION — update before flashing
// ============================================================
#define WIFI_SSID        "YOUR_WIFI_SSID"
#define WIFI_PASSWORD    "YOUR_WIFI_PASSWORD"

// Firebase Realtime Database URL
#define FIREBASE_URL     "https://aquatrack-b00f5-default-rtdb.firebaseio.com"
#define FIREBASE_SECRET  "YOUR_DATABASE_SECRET"

// ============================================================
// PIN DEFINITIONS
// ============================================================
#define SDA_PIN          2       // MPU-6050 I2C Data
#define SCL_PIN          14      // MPU-6050 I2C Clock
#define HX711_DT         13      // HX711 Data
#define HX711_SCK        12      // HX711 Clock
#define BUZZER_PIN       15      // Active buzzer / vibration motor

// ============================================================
// SENSOR THRESHOLDS
// ============================================================
const float TILT_DRINK_THRESHOLD  = 35.0;   // degrees — drinking posture
const float TILT_REST_THRESHOLD   = 20.0;   // degrees — bottle at rest
const float MIN_SIP_ML            = 15.0;   // minimum valid sip (ml)
const float MAX_SIP_ML            = 600.0;  // maximum valid sip (ml)
const int   SETTLE_READINGS       = 4;      // consecutive stable readings
const float CALIBRATION_FACTOR    = -1075.0; // HX711 calibration — update after cal

// ============================================================
// MPU-6050
// ============================================================
const int   MPU_ADDR = 0x68;
int16_t     rawAx, rawAy, rawAz;

// ============================================================
// HX711
// ============================================================
HX711 scale;

// Median filter for weight noise rejection
const int  MED_SIZE  = 7;
float      medBuf[MED_SIZE];
int        medIdx    = 0;
bool       medFilled = false;

float getMedianWeight() {
  float sorted[MED_SIZE];
  int   count = medFilled ? MED_SIZE : medIdx;
  for (int i = 0; i < count; i++) sorted[i] = medBuf[i];
  // Insertion sort
  for (int i = 1; i < count; i++) {
    float key = sorted[i];
    int   j   = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j]; j--;
    }
    sorted[j + 1] = key;
  }
  return sorted[count / 2];
}

// ============================================================
// STATE MACHINE
// ============================================================
enum DrinkState { IDLE, DRINKING, RETURNING };
DrinkState  currentState  = IDLE;
int         settleCount   = 0;
float       startWeight   = 0.0;
float       currentWeight = 0.0;
float       currentTilt   = 0.0;

// ============================================================
// ALERT TRACKING
// ============================================================
unsigned long lastDrinkTime      = 0;
const long    ALERT_INTERVAL_MS  = 2700000; // 45 minutes
bool          alertSent          = false;

// ============================================================
// WIFI & FIREBASE
// ============================================================
bool wifiConnected = false;

void connectWiFi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
  } else {
    wifiConnected = false;
    Serial.println("\nWiFi failed — running in offline mode");
  }
}

bool pushToFirebase(String path, String jsonPayload) {
  if (!wifiConnected || WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String url = String(FIREBASE_URL) + path + ".json?auth=" + FIREBASE_SECRET;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(jsonPayload);
  http.end();
  return (code == 200 || code == 201);
}

bool patchFirebase(String path, String jsonPayload) {
  if (!wifiConnected || WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String url = String(FIREBASE_URL) + path + ".json?auth=" + FIREBASE_SECRET;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.PATCH(jsonPayload);
  http.end();
  return (code == 200);
}

// ============================================================
// SENSOR READS
// ============================================================
float readTiltAngle() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  rawAx = Wire.read() << 8 | Wire.read();
  rawAy = Wire.read() << 8 | Wire.read();
  rawAz = Wire.read() << 8 | Wire.read();

  float gx = rawAx / 16384.0;
  float gy = rawAy / 16384.0;
  float gz = rawAz / 16384.0;

  // True tilt angle from vertical — axis-independent
  return atan2(sqrt(gx * gx + gy * gy), abs(gz)) * 180.0 / PI;
}

float readWeight() {
  if (!scale.is_ready()) return currentWeight;
  float raw = scale.get_units(3);
  if (raw < 0) raw = 0;

  // Add to median filter buffer
  medBuf[medIdx % MED_SIZE] = raw;
  medIdx++;
  if (medIdx >= MED_SIZE) medFilled = true;

  return getMedianWeight();
}

// ============================================================
// ALERT FUNCTIONS
// ============================================================
void buzzAlert(int beeps, int duration_ms) {
  for (int i = 0; i < beeps; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < beeps - 1) delay(150);
  }
}

void sendDehydrationAlert() {
  // 3 short beeps = dehydration warning
  buzzAlert(3, 300);

  // Push alert to Firebase
  String payload = "{\"type\":\"DEHYDRATION_ALERT\","
                   "\"timestamp\":" + String(millis()) + ","
                   "\"mins_since_drink\":" +
                   String((millis() - lastDrinkTime) / 60000.0, 1) + "}";
  pushToFirebase("/users/tasneem/alerts", payload);
}

// ============================================================
// DRINK EVENT LOGGING
// ============================================================
void logDrinkEvent(float sip_ml, float start_w, float end_w, float tilt) {
  // 1 short beep = drink confirmed
  buzzAlert(1, 200);

  Serial.println("DRINK_LOGGED," + String(sip_ml, 1));

  // Push drink record to Firebase
  String drinkPayload =
    "{\"sip_ml\":"      + String(sip_ml, 1)   + ","
    "\"start_weight\":" + String(start_w, 1)  + ","
    "\"end_weight\":"   + String(end_w, 1)    + ","
    "\"tilt_angle\":"   + String(tilt, 1)     + ","
    "\"timestamp\":"    + String(millis())    + "}";
  pushToFirebase("/users/tasneem/drinks", drinkPayload);

  lastDrinkTime = millis();
  alertSent     = false;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // ── MPU-6050 init ──────────────────────────────────────────
  Wire.begin(SDA_PIN, SCL_PIN);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);  // Wake up — disable sleep mode
  Wire.endTransmission(true);

  // Set accelerometer to ±2G range (best sensitivity)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Set gyroscope to ±250°/s range
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Serial.println("MPU-6050 initialized");

  // ── HX711 init ─────────────────────────────────────────────
  scale.begin(HX711_DT, HX711_SCK);
  scale.set_scale(CALIBRATION_FACTOR);

  Serial.println("Taring load cell — keep bottle still...");
  delay(2000);
  scale.tare(20);   // Average 20 readings for clean zero
  Serial.println("Load cell tared");

  // Pre-fill median buffer
  for (int i = 0; i < MED_SIZE; i++) {
    float r    = scale.get_units(3);
    medBuf[i]  = (r < 0) ? 0 : r;
  }
  medIdx    = MED_SIZE;
  medFilled = true;

  // ── WiFi & Firebase ────────────────────────────────────────
  connectWiFi();

  // ── Ready ─────────────────────────────────────────────────
  lastDrinkTime = millis();
  buzzAlert(2, 100);  // 2 quick beeps = system ready

  Serial.println("=== AquaTrack Ready ===");
  Serial.println("Weight_g,Tilt_deg,State");
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {

  // ── 1. Listen for commands from Python dashboard ───────────
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'A') {
      // 'A' = dehydration alert command from Python
      sendDehydrationAlert();
    }
    if (cmd == 'T') {
      // 'T' = re-tare command
      scale.tare(10);
      Serial.println("INFO:TARE_COMPLETE");
    }
  }

  // ── 2. Read sensors ────────────────────────────────────────
  currentTilt   = readTiltAngle();
  currentWeight = readWeight();

  // ── 3. State machine ───────────────────────────────────────
  switch (currentState) {

    case IDLE:
      // Continuously update baseline weight while bottle is resting
      if (currentTilt < TILT_REST_THRESHOLD) {
        startWeight = currentWeight;
      }

      // Transition to DRINKING when tilt threshold exceeded
      if (currentTilt > TILT_DRINK_THRESHOLD) {
        currentState = DRINKING;
        settleCount  = 0;
        Serial.println("EVENT:DRINK_START");
      }

      // Check dehydration alert timer
      if (!alertSent &&
          (millis() - lastDrinkTime) > ALERT_INTERVAL_MS) {
        sendDehydrationAlert();
        alertSent = true;
      }
      break;

    case DRINKING:
      // Wait for bottle to return to rest position
      if (currentTilt < TILT_REST_THRESHOLD) {
        currentState = RETURNING;
        settleCount  = 0;
        Serial.println("EVENT:DRINK_END_DETECTED");
      }
      break;

    case RETURNING:
      // Count consecutive stable flat readings
      if (currentTilt < TILT_REST_THRESHOLD) {
        settleCount++;
      } else {
        // Still moving — reset settle counter
        settleCount = 0;
      }

      if (settleCount >= SETTLE_READINGS) {
        // Bottle has settled — calculate sip volume
        float sipML = startWeight - currentWeight;

        if (sipML > MIN_SIP_ML && sipML < MAX_SIP_ML) {
          logDrinkEvent(sipML, startWeight, currentWeight, currentTilt);
        } else {
          // Too small or too large — likely false detection
          Serial.println("EVENT:FALSE_DETECTION," + String(sipML, 1));
        }

        currentState = IDLE;
        settleCount  = 0;
      }
      break;
  }

  // ── 4. Send data to Python dashboard via Serial ───────────
  // Format: Weight,Tilt,State
  Serial.print(currentWeight, 1);
  Serial.print(",");
  Serial.print(currentTilt, 1);
  Serial.print(",");
  Serial.println(currentState == IDLE      ? "IDLE" :
                 currentState == DRINKING  ? "DRINKING" :
                                             "RETURNING");

  // 10 Hz sampling rate
  delay(100);
}
