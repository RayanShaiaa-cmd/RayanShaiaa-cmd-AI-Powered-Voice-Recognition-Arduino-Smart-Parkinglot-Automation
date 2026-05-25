#include <Servo.h>
#include <LiquidCrystal_I2C.h>
int portLED = 2;

int portBuzzer = 4;

Servo myServo;
LiquidCrystal_I2C lcd(0x27,16,2);

int portTrig = 3;
int portEcho = 5;
float distance;
float travelTime;
int counter = 0;

String msg;

bool objectDetected = false;
unsigned long lastUltrasonicRead = 0;
const unsigned long ultrasonicInterval = 50; // كل 100 ms

void setup() {
  pinMode(portLED,OUTPUT);
  pinMode(portBuzzer,OUTPUT);
  pinMode(portTrig,OUTPUT);
  pinMode(portEcho,INPUT);

  digitalWrite(portLED,HIGH);

  myServo.attach(6);
  myServo.write(0);

  lcd.init();
  lcd.backlight();

  Serial.begin(115200);
}

void loop() {

  lcd.setCursor(0,0);
  lcd.print("--Parking lots--");
  lcd.setCursor(0,1);
  lcd.print("Count : ");
  lcd.setCursor(8, 1);
  lcd.print(counter);

  digitalWrite(portTrig,LOW);
  delayMicroseconds(2);
  digitalWrite(portTrig,HIGH);
  delayMicroseconds(10);
  digitalWrite(portTrig,LOW);

  if (millis() - lastUltrasonicRead >= ultrasonicInterval)
{
  lastUltrasonicRead = millis();

  digitalWrite(portTrig, LOW);
  delayMicroseconds(2);
  digitalWrite(portTrig, HIGH);
  delayMicroseconds(10);
  digitalWrite(portTrig, LOW);

  travelTime = pulseIn(portEcho, HIGH, 25000); // timeout 25ms
  distance = 0.034 * travelTime / 2;

  if (distance <= 7 && !objectDetected)
  {
    objectDetected = true;   // تم اكتشاف جسم
    counter++;               // عدّ مرة واحدة فقط
  }

  if (distance > 5)
  {
    objectDetected = false;  // الجسم ابتعد
  }
}

  Serial.println(distance);

  lcd.scrollDisplayLeft();
  delay(300);
  lcd.scrollDisplayLeft();
  delay(300);

  Serial.println("Enter turn on , turn off : ");

  while(Serial.available() != 0)
  {
  msg = Serial.readString();
  msg.trim();
  Serial.println(msg);
  if (msg == "on")
  {
    digitalWrite(portLED,HIGH);
    Serial.println("LED os turned on");
  }
  if (msg == "off")
  {
    digitalWrite(portLED,LOW);
    Serial.println("LED os turned off");
  }
  if (msg == "unlock")
  {
    myServo.write(80);
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("--- WELCOME ---");
    counter += 1;

    for (int x = 0; x <= 5; x++)
    {
      digitalWrite(portBuzzer,HIGH);
      delay(500);
      digitalWrite(portBuzzer,LOW);
    }
    
    myServo.write(0);

    lcd.setCursor(0,0);
    lcd.print("--Parking lots--");
  }
  if (msg == "alarm")
  {
    for (int x = 0; x <= 5; x++)
    {
      digitalWrite(portBuzzer,HIGH);
      delay(500);
      digitalWrite(portBuzzer,LOW);
	    delay(500);
      
    }
  }
  if (msg == "number")
  {
    lcd.clear();
    lcd.setCursor(0 , 0);
    lcd.print("The parking lot");
    lcd.setCursor(0 , 1);
    lcd.print("has : ");
    lcd.setCursor(3 , 1);
    lcd.print(counter);
    lcd.setCursor(5 , 1);
    lcd.print(" cars");
    delay(5000);
  }
  }
}
