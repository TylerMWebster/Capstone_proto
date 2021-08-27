/********************************************************************/
// First we include the libraries
#include <OneWire.h> 
#include <DallasTemperature.h>
/********************************************************************/
// Data wire is plugged into pin 2 on the Arduino 
#define ONE_WIRE_BUS 2 
/********************************************************************/
// Setup a oneWire instance to communicate with any OneWire devices  
// (not just Maxim/Dallas temperature ICs) 
OneWire oneWire(ONE_WIRE_BUS); 
/********************************************************************/
// Pass our oneWire reference to Dallas Temperature. 
DallasTemperature sensors(&oneWire);
/********************************************************************/ 
 float strt_time;
 float curr_time;
 int device_count;
 DeviceAddress addys = { };
 

void setup(void) 
{ 

 DeviceAddress this_add;
 // start serial port 
 Serial.begin(9600); 
 Serial.println("Dallas Temperature IC Control Library Demo"); 
 // Start up the library 
 sensors.begin();
 Serial.print("Number of sensors found: "); 
 device_count = sensors.getDeviceCount();
 Serial.println(device_count);
 delay(50);
 Serial.println("Getting Device Adresses...");
 for(int i = 0; i < device_count; i++){
  sensors.getAddress(this_add, i);
  addys[i] = this_add;
  printAddress(this_add);
  Serial.println();
 }
 Serial.println("All Adresses Found");
} 
void loop(void) 
{ 
 // call sensors.requestTemperatures() to issue a global temperature 
 // request to all devices on the bus 
/********************************************************************/
 //Serial.print(" Requesting temperatures..."); 
 strt_time = millis();
 sensors.requestTemperatures();// Send the command to get temperature readings 
 curr_time = millis();
 //Serial.print(curr_time - strt_time);
 //Serial.print(", ");
 
 for(int d = 0; d < device_count; d++){
 //sensors.requestTemperaturesByAddress(addys[d]);
 Serial.print(sensors.getTempF(addys[d]));
 Serial.print(", ");
 }
 Serial.println();
} 


// function to print a device address
void printAddress(DeviceAddress deviceAddress)
{
  for (uint8_t i = 0; i < 8; i++)
  {
    // zero pad the address if necessary
    if (deviceAddress[i] < 16) Serial.print("0");
    Serial.print(deviceAddress[i], HEX);
  }
}
