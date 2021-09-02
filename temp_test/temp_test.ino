/********************************************************************/
// First we include the libraries
#include <OneWire.h>
#include <DallasTemperature.h>
#include <stdlib.h>
 /********************************************************************/
// Data wire is plugged into pin 2 on the Arduino
#define ONE_WIRE_BUS 2

// The max number of sensors
#define MAX_SENSORS 10
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

DeviceAddress addrs[MAX_SENSORS];

void setup(void)
{
  // start serial port
  Serial.begin(9600);
  // Start up the library
  sensors.begin();
  // Discover the network
  discoverNetwork();
    
}


void loop(void)
{
   
  strt_time = millis();
  sensors.requestTemperatures();// Send the command to get temperature readings
  sensors.setWaitForConversion(true);
  curr_time = millis();
  
  String json = "";
  json+="{";

  // Print JSON format to serial
  for (int i = 0; i < device_count; i++) {
    json += "\"";
      for (uint8_t j = 0; j < 8; j++)
      {
        // zero pad the address if necessary
        if (addrs[i][j] < 16) json+= "0" ;
        json+= addrs[i][j];
      }
    json += "\"";
    /*
    Serial.print("Time");
    Serial.print(" : ");
    Serial.print(millis());
    Serial.print(", ");
    */
    //printAddress(addrs[i]);
    json+= " : ";
    json += "\"";
    json += sensors.getTempF(addrs[i]);
    json += "\"";
    if( i < device_count -1){
      json += ", ";
    }
  }
  json+="}";

  //Serial.print(sensors.getTempF(a));

  Serial.println(json);
  delay(50);
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

// Function to discover network size and sensor adresses
void discoverNetwork(){

  DeviceAddress this_add;

  // Find network size
  Serial.print("Number of sensors found: ");
  device_count = sensors.getDeviceCount();
  Serial.println(device_count);
  delay(50);

  // Find adresses for all sensoprs in the network and add them to a gloabal list
  Serial.println("Getting Device Adresses...");

  for (int i = 0; i < device_count; i++) {
    sensors.getAddress(this_add, i);

    // Copy array of 8 bytes representing address to sensor address array
    memcpy(addrs[i], this_add, sizeof(addrs[i]));
    
    printAddress(this_add);
    Serial.println();
  }

  Serial.println("All Adresses Found");

}
