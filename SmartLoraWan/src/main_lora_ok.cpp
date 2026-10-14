#include <iostream>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <wiringPi.h>
#include <pigpio.h>
#include <wiringSerial.h>
#include "../include/shared_data.h"
#include "../include/wan_keys.h"
//#include <json/json.h>
// #include <opencv2/opencv.hpp>
//#include "../include/PiHal.h"

#define IMAGE_SIZE_V 256
#define IMAGE_SIZE_H 320

//==========================================================================================
// FUNCOES
//==========================================================================================
#pragma region FUNCOES

int main();
int initializeTimer();

void sendATCommand(const char *cmd, int serial_port);
void processCommand(const std::string &command, int serial_port);
void initializeLoRa(int serial_port);
//std::string createPayload(bool keepAlive);
void initializeSystem(int serial_port);


uint32_t image[IMAGE_SIZE_V][IMAGE_SIZE_H];
const char* command_image = "raspistill -o image.bmp -w 320 -h 256";
char LED_STATUS_PIN = 32;
extern const unsigned char MagickImage[];

std::string deveui_var = "00:00:00:00:00:00:00:00";
std::string devadd_var = "00:00:00:00";

int serial_port; // indicador de qual serial

int initializeTimer(void)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}



/* Function: Send a AT Command to Lora Module
 * Parameters: Pointer to a char vector
 * Return: None
*/
void sendATCommand(const char *cmd, int serial_port){
    serialFlush(serial_port);
    int timerSend = millis();
    std::cout << "Sending command: " << cmd << std::endl;
    serialPuts(serial_port, cmd);
    serialPuts(serial_port, "\n");
    std::string response;
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    bool responseReceived = false;

    while ((timerSend + 1000) > millis()) 
    {
        while (serialDataAvail(serial_port)) 
        {
            char c = serialGetchar(serial_port);
            if (c == '\r' || c == '\n') 
            {
                if (!response.empty()) { // Se já capturou alguma coisa, encerra a leitura
                    responseReceived = true;
                    break;
                }
            } 
            else 
            {
                response += c;
            }
        }
        if (responseReceived) 
        {
            break;
        }
    }

    if (!response.empty()) 
    {
        std::cout << "\nResponse: " << response << std::endl;
    } else 
    {
        std::cout << "\nNo response received within the timeout period." << std::endl;
    }
    // std::cout << std::string("Command:") + cmd;

    if (cmd == "AT+RECVB=?") 
    {
        processCommand(response,serial_port);
    }
    if (cmd == "AT+DEUI=?") 
    {
        devadd_var = response.substr(12,11);
        std::cout << "\nResponse: " << devadd_var << std::endl;
    }
    
}

void processCommand(const std::string &command, int serial_port) {
    char buffer[200];
    char bufferPayload[200];
    size_t offset = 0;

    if (command.size() > 2) 
    {
        std::string substring = command.substr(2);
        std::cout << substring;
        if (substring == "0001") 
        {
            std::cout << ">>>>>>> aciona deplyer" << std::endl;
        }
        else if (substring == "0002") 
        {
            std::cout << ">>>>>>> desliga deplyer" << std::endl;
        }
        else if (substring == "0003") 
        {
            std::cout << ">>>>>>> msg3" << std::endl;
        }
        else if (substring == "0004") 
        {
            std::cout << ">>>>>>> msg4" << std::endl;
        }
        else if (substring == "0005") 
        {
            std::cout << ">>>>>>> msg5" << std::endl;
        }
    }
}

void initializeLoRa(int serial_port) 
{
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    char command_at[200];

    std::cout << "Initializing LoRa: " << std::endl;

    sendATCommand("AT", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("ATZ", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+NJM=0", serial_port); // Modo de Join ABP
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+CLASS=A", serial_port); // Classe A
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+DR=5", serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));    
    sendATCommand("AT+DEUI=?", serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sprintf(command_at, "AT+APPSKEY=%s", APPSKEY);
    sendATCommand(command_at, serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sprintf(command_at, "AT+NWKSKEY=%s", NWKSKEY);
    sendATCommand(command_at, serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sprintf(command_at, "AT+DADDR=%s", devadd_var);
    sendATCommand(command_at, serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+RX1DL=1000", serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+RX2DL=2000", serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT+RX2FQ=923300000", serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sprintf(command_at, "AT+SAVE");
    sendATCommand(command_at, serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::cout << "LoRa module configured and ready." << std::endl;
}

void initializeSystem(int serial_port)
{
   initializeLoRa(serial_port);
}

void handle_sigint(int signal ) 
{
    std::cout << "\nCaught Ctrl+C (signal " << signal << "). Cleaning up..." << std::endl;
    serialClose(serial_port);
    std::exit(0); // Exit gracefully
}

//==========================================================================================
// MAIN
//==========================================================================================
#pragma region MAIN
int main()
{
    static int contador = 0;

    gpioInitialise();
    wiringPiSetupGpio();
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN,0);
    
    const char compilation_date[] = "Compiled on " __DATE__ " at " __TIME__ "";
    std::cout << "Cubesat sw 0.0.001 " << compilation_date << std::endl;

    std::string command_1 = "sudo systemctl stop serial-getty@ttyAMA0";
    system(command_1.c_str()); 
    int baudRate = 9600; // Defina a taxa de transmissão desejada
    std::string device = "/dev/ttyAMA0"; // Verifique a porta correta
    std::string command = "sudo stty -F " + device + " " + std::to_string(baudRate);
    system(command.c_str());
    serial_port = serialOpen(device.c_str(), baudRate); 
    serialFlush(serial_port);   

    std::signal(SIGINT, handle_sigint); 
    std::cout << "Running... Press Ctrl+C to interrupt.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    if (serial_port < 0) 
    {
        std::cerr << "Error opening serial port" << std::endl;
        return 1;
    }
    else
    {
      std::cerr << "Sucess opening serial port!" << std::endl;
    }

    initializeSystem(serial_port);

    contador = 0;
    while(contador < 300)
    {
        contador++;
        sendATCommand("AT+SEND=100:Uplink Mauasat", serial_port);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        sendATCommand("AT+RECVB=?", serial_port);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        serialFlush(serial_port);
    }

    return 0;
}
