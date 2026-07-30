// ============================================================//
//  16x2 LCD Wiring:
//    RS  → pin 12
//    EN  → pin 11
//    D4  → pin 5
//    D5  → pin 4
//    D6  → pin 3
//    D7  → pin 2
//    RW  → GND
//    VSS → GND
//    VCC → 5V
//    LED+ → 5V (through 220Ω resistor)
//    LED- → GND
//    V0  → potentiometer wiper
// for potentiometer V0 connects to the middle leg, other legs connect to ground and 5V
// ============================================================

#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
void setup() 
{
  Serial.begin(9600);
  lcd.begin(16, 2);

  // Startup messages to test if screen is wired properly
  lcd.print("Resistor");
  lcd.setCursor(0, 1);
  lcd.print("Detector Ready");
  delay(2000);

  lcd.clear();
  lcd.print("Waiting...");

  Serial.println("READY");
}

void loop() 
{
  if (Serial.available() > 0)
  {
    String incoming = Serial.readStringUntil('\n');
    incoming.trim();
    lcd.clear();

    if (incoming == "NONE" || incoming == "Unknown")
    {
      lcd.setCursor(0,0);
      lcd.print("No resistor");
      lcd.setCursor(0,1);
      lcd.print("detected");
    } else
    {
      lcd.setCursor(0,0);
      if (incoming.length() > 16)
      {
        lcd.print(incoming.substring(0,16));
      } else
      {
        lcd.print(incoming);
      }

      lcd.setCursor(0,1);
      lcd.print("Detected");
    }
  }
}
