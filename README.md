# IOT_Irrigation-Prediction-Model



Smart Irrigation System using Iot…
 
Introduction

➤ India's population crossing 1.3 billion in 2016

So balance between the optimum population growth and a healthy of nation is far to be achieved.

➤ The rising population need for increased agricultural production

➤ Irrigated agriculture has been important source increased agricultural production

"IOT based smart irrigation system" is for to create an IOT base automated irrigation mechanism which turns the pumping motor ON and OFF pass command through IOT platform.




➤ The Internet of Things (IoT) is the inter-networking of "physical devices" also referred to as "connected devices and "smart devices".

➤ Sometimes referred to as the Internet of Everything (IOE) Machine to Machine (M2M) communicating.

➤ IOT is expected to offer advanced connectivity of devices, systems, and services that covers a variety of protocols, domains, and applications



 


System Component: Hardware and Software

1. Arduino: It is an open-source platform based on easy-to-use hardware

and software:
➤Hardware: Arduino

Arduino board designs use a variety of microprocessors and controllers in system

1) To read inputs - light on a sensor

2) To twitter message and turn it into an output-activating a motor
3) Turning on an LED









Software :-Arduino (IDE= Integrated Development Environment)

Use: Check Conditions:

Writing Sketches: Programs written using Arduino Software (IDE) are called sketches

Upload: Compiles your code and uploads it to the configured board

New: Creates a new sketch

Save: Saves your sketch

Serial Monitor: File, Edit, Sketch, Tools, Help
 




Window Application (Motor Control)

This Window Application install in with: Smartphone, tablets or Computers

➤Give command: ON/OFF from this type of application
 
GSM shield:

➤ GSM stands for Global System for Mobile Communications

➤ GSM supports outgoing and incoming voice calls, Simple Message System (SMS or text messaging), and data communication
 

 
Soil Moisture sensor

Use: To measure the moisture content of the soil.

Copper electrodes are used to sense the moisture content of soil.
 


Wireless water level detector sensor

Use:

The water level sensor mechanism to detect and indicate the water level in an water source.
 
Relay Switch

Relays are switches that open and close Motors Based on Command of Arduino

In a basic relay there are three contactors:

normally open (ND), normally closed (NC) and common (COM), At ON input state, the COM is connected to NC.
 


Submersible Motor Pump

Q Search

A submersible pump is for water lifting. Motor is connected with Raspberry PI 3 via Arduino.
 
How the system works?

➤ Step 1:

Login: (Enter Username/Password) and

Give Command (ON/OFF) to your application
:

IOT base platform: Collect and send all Analog data to GSM Shield

GSM Shield connected in RP3 (Raspberry Pi 3) (Microcomputer)

Now command(ON/OFF) command pass to RP3
  
 
 
 
 
 
 


✅ Basic Arduino code
cpp
CopyEdit
int soilMoistureSensorPin = A0;  // Analog pin
int pumpPin = 7;  // Digital pin to relay

void setup() {
  Serial.begin(9600);
  pinMode(pumpPin, OUTPUT);
  digitalWrite(pumpPin, LOW); // Pump off by default
}

void loop() {
  int moistureValue = analogRead(soilMoistureSensorPin);
  Serial.print("Soil Moisture: ");
  Serial.println(moistureValue);

  // Assuming: dry < 500 (adjust based on sensor)
  if (moistureValue < 500) {
    Serial.println("Soil is dry, turning on pump");
    digitalWrite(pumpPin, HIGH);
    delay(5000);  // Water for 5 seconds
    digitalWrite(pumpPin, LOW);
  } else {
    Serial.println("Soil is wet, pump OFF");
    digitalWrite(pumpPin, LOW);
  }

  delay(10000); // Wait 10 seconds
}

✅ Model training using python 
نستخدم مكتبة scikit-learn لتدريب نموذج بسيط.
📌 Model training Python example code 
python
CopyEdit
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# تحميل البيانات
df = pd.read_csv("moisture_data.csv")  # ملف CSV فيه البيانات

# تجهيز الميزات Features والهدف Target
X = df[['Moisture']]  # يمكنك إضافة خصائص أخرى
y = df['Irrigation']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# تدريب نموذج Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, 'irrigation_model.pkl')
________________________________________
✅ المرحلة 3: عمل API باستخدام Flask لتوصيل Arduino مع الذكاء الاصطناعي
python
CopyEdit
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('irrigation_model.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        moisture = float(request.args.get('moisture'))
        prediction = model.predict([[moisture]])
        return jsonify({'irrigate': int(prediction[0])})
    except:
        return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
________________________________________
✅ المرحلة 4: جعل Arduino يتصل بـ Flask API عبر WiFi (باستخدام ESP8266)
📌 مثال على كود ESP8266 داخل Arduino:
cpp
CopyEdit
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

const char* ssid = "اسم_الشبكة";
const char* password = "كلمة_المرور";

const String server = "http://192.168.1.100:5000/predict"; // IP للـ Python API

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
}

void loop() {
  int moisture = analogRead(A0);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = server + "?moisture=" + String(moisture);
    http.begin(url);
    int httpCode = http.GET();

    if (httpCode == 200) {
      String payload = http.getString();
      Serial.println(payload);
      if (payload.indexOf("\"irrigate\":1") >= 0) {
        digitalWrite(D7, HIGH);  // شغل الري
        delay(5000);
        digitalWrite(D7, LOW);
      }
    }

    http.end();
  }

  delay(10000);
}

>UseCase Digram
 














>System Work Flow
 
 



 









Case study:

➤ Manufacturers' Association for Information Technology (MAIT)

➤ MAIT Industries head office is located in Melbourne, Australia.

➤ MAIT Industries provides innovative monitoring and irrigation control solution
 
Software developed by MAIT:

INTELLITROL-Radio telemetry irrigation control

Network Control Program

Remote network configuration from PC

Remote re-routing of radio transmission paths directly from the PC

Remote re-programming of field loggers

Alarm notification settings
  
The system provides real-time information :EX.:soil moisture, soil temperature and rainfall

MAIT's system also integrates a "rain switch "stopping irrigation when it rains

Advantages

➤ Water Conservation

➤ Real-Time Data give

➤ Lowered Operation Costs

➤ Efficient and Saves Time

➤ Increase in productivity

➤ Reduce soil erosion and nutrient leaching




Challenges

➤ Complexity: The IOT is a diverse and complex network

➤ Privacy/Security:

➤ Lesser Employment of Manual Staff or unskilled workers:

➤ Equipment is costlier.

➤ Awareness of Indian farmer for this technology

Conclusion

➤ I conclude that this system is easy to implement and time, money and manpower saving solution for irrigating fields.

➤ A farmer should visualize his agricultural land's moisture content from time to time and water level of source is sufficient or not. IOT based smart irrigation system displays the values of the sensors continuously in smart phone or on computer's web page and farmer can operate them anytime from and anywhere.
