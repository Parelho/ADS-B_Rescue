# Smart LoRaWAN

# Overview
This is a repo for a generic LoRaWAN application. The code contains the function to create a JSON payload and send to Smart LoRaWAN module using AT command. It can be use to apply in any application involving IoT. As others applications made by BSC, the hardware used was a *ESP32* using platform.io on VS Code.

## Folder Contents
```
├── SmartLoraWAN
│   ├── .vscode            
|   ├── docs
│   │   └── SMART_LoRa_AT_Command_v1.0_en_v2.14.pdf
│   ├── include
|   |   ├── shared_data.h
|   |   ├── chaves_wan.h
│   │   └── README.md      
│   ├── lib
│   │   └── README.md      
│   ├── src
│   │   ├── chaves_wan.h   
│   │   └── main.cpp       
│   ├── test
│   │   └── README.md      
│   ├── .gitignore          
│   └── platformio.ini     
└── README.md              
```

## Hardware

The hardwares used to this project was:

- ESP32-WROOM-32
- SmartLora SMW-SX1262M0
- FTDI Module

## Software

The softwares used was:

- VSCode
- PlatformIO Extension

## PlatformIO
The PlatformIO is a VS Code extension that allows to program the ESP32 on VS Code IDE. To run the code it is necessary to download the extension on VS Code and create or select a project that contains the *platformio.ini* file. If you are creating a project, the file will be created. It is import to set the right *_upload_port_* and *_monitor_port_* to upload the code to ESP32.

## Software Details

The Smart LoRaWAN module is programmed using AT commands, which means it is necessary to open a serial port for LoRa and send the command using the print function. In this case, a hardware serial port on port 1 named LORA was created, and every time an AT command is sent, it must be sent to this serial port using the command LORA.print(<command>).

> Note: To better understand these commands, refer to the [Smart LoRaWAN Documentation](docs/SMART_LoRa_AT_Command_v1.0_en_v2.14.pdf).

To ensure the module is efficient in battery usage, the deep sleep functionality of the ESP32 was used for GPIO inputs and timer wake-up. The deep sleep function allows the ESP32 to remain in standby mode until it is woken up by a wake-up reason, thereby avoiding unnecessary energy consumption.

> Tip: For more info about deep sleep, read this [article](https://randomnerdtutorials.com/esp32-deep-sleep-arduino-ide-wake-up-sources/).

Another point is the control of the Frame Counter in the code. The LoRaWAN mode used is __ABP__, primarily to reduce battery consumption. This mode requires that the Frame Counter be managed by the application software, as in our case. Therefore, 4 bytes of EEPROM memory were used to store the Frame Counter value, totaling approximately 4.3 billion possible values.

## How to Compile

To flash the ESP32, an FTDI to USB converter module was used.

Follow the circuit diagram below:

![FTDI Module](docs/circuit.jpeg)

Next, follow these steps to compile and flash the code:

1. Open the project folder in VS Code.
2. Install the PlatformIO extension in VS Code.
3. Go to PIO Home on the upper side of the screen.
4. Click on "New Project" or "Open Project".
5. Click the "Upload" button to compile and flash the code.
6. Once the upload process starts, press the flash and reset buttons on the circuit.
7. Wait for the code to compile and upload.
8. Open the serial monitor in VS Code.
9. Press the reset button on the circuit.

If everything is set up correctly, the program should run as expected.