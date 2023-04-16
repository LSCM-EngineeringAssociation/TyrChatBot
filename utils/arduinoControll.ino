#include <Servo.h>

#define numOfValsRec 12
#define digitsPerValRec 3

Servo thumb, index, middle, ring, pinky, wrist, shoulder, bicep, eyeUpDown, eyeLeftRight, neck, jaw;

const int servoPins[numOfValsRec] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

int ValsRec[numOfValsRec];
int stringLength = numOfValsRec * digitsPerValRec + 1; //$000000000000
int counter = 0;
bool counterStart = false;
String receivedString;

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < numOfValsRec; i++) {
    Servo servo;
    servo.attach(servoPins[i]);
    switch (i) {
      case 0: thumb = servo; break;
      case 1: index = servo; break;
      case 2: middle = servo; break;
      case 3: ring = servo; break;
      case 4: pinky = servo; break;
      case 5: wrist = servo; break;
      case 6: shoulder = servo; break;
      case 7: bicep = servo; break;
      case 8: eyeUpDown = servo; break;
      case 9: eyeLeftRight = servo; break;
      case 10: neck = servo; break;
      case 11: jaw = servo; break;
    }
  }
}

void receiveData() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '$') {
      counterStart = true;
    }
    if (counterStart) {
      if (counter < stringLength) {
        receivedString = String(receivedString + c);
        counter++;
      }
      if (counter >= stringLength) {
        for (int i = 0; i < numOfValsRec; i++) {
          int num = (i * digitsPerValRec) + 1;
          ValsRec[i] = receivedString.substring(num, num + digitsPerValRec).toInt();
        }
        receivedString = "";
        counter = 0;
        counterStart = false;
      }
    }
  }
}

void loop() {
  receiveData();
  thumb.write(ValsRec[0]);
  index.write(ValsRec[1]);
  middle.write(ValsRec[2]);
  ring.write(ValsRec[3]);
  pinky.write(ValsRec[4]);
  wrist.write(ValsRec[5]);
  shoulder.write(ValsRec[6]);
  bicep.write(ValsRec[7]);
  eyeUpDown.write(ValsRec[8]);
  eyeLeftRight.write(ValsRec[9]);
  neck.write(ValsRec[10]);
  jaw.write(ValsRec[11]);
}