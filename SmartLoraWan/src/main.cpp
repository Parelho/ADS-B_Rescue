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
#include "../include/PiHal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/if.h>
#include <cstdint>
#include "shared_data.h"

// util pinagem cm3   https://elinux.org/RPi_BCM2835_GPIOs

typedef struct 
{
  long can_id;
  char data[8];
}Can_frame_mauasat ;

Can_frame_mauasat can_frame_mauasat;
Can_frame_mauasat new_frame;
Can_frame_mauasat frames[40];
Can_frame_mauasat frame_ADSB;

//==========================================================================================
// FUNCOES
//==========================================================================================
#pragma region FUNCOES

int main();
int initializeTimer();
void sendATCommand(const char *cmd, int serial_port);
void processCommand(const std::string &command, int serial_port);
void initializeLoRa(int serial_port);
void initializeSystem(int serial_port);
char send_can(long can_id, char *data);
char recieve_can(long *can_id, char *data);
void can_thread(int id);
void lora_thread(int id);
void ADSB_thread(int id);

char LED_STATUS_PIN = 32;

std::string deveui_var = "00:00:00:00:00:00:00:00";
std::string devadd_var = "00:00:00:00";

// Não descomentar, fica travando o rasp
// std::string command_photo = "sudo raspistill -cs 0 -o /home/pi/cubesat/image.jpg -h 1080 -w 1920 --colfx 128:128 -ex backlight -t 1";

long can_id_tx = 0;
char data_tx[8];

long can_id = 0;
char data[8];
char result;

int serial_port; // indicador de qual serial
int serial_port_ADSB; // indicador de qual serial
int contador = 0;
int ciclos = 0;
time_t time_last, time_now;
int frames_pos = 0;
bool status = 0;
char command_at[250];
char command_at_ADSB[250];

std::string response_ADSB = "000000000000000000000000";
std::string response_ADSB_high = "000000000000000000000000";
std::string response_ADSB_low = "000000000000000000000000";

char send_can(long can_id, char *data) 
{
    int s; 
    unsigned long aux_l;
	struct sockaddr_can addr;
	struct ifreq ifr;
	struct can_frame frame;

	if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
    {
		perror("Socket");
		return 1;
	}

	strcpy(ifr.ifr_name, "can0");
	ioctl(s, SIOCGIFINDEX, &ifr);

	memset(&addr, 0, sizeof(addr));
	addr.can_family = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("Bind");
		return 1;
	}

	frame.can_dlc = 8; // can payload size in bytes

	//aux_l = (frame.can_id & CAN_EFF_MASK) >> 18;
	//aux_l += ((frame.can_id & CAN_EFF_MASK) & 0x0003FFFF ) <<11;
	
    aux_l = ( can_id & 0x1FFFFFFF) >> 11;
    aux_l +=( can_id & 0x00000FFF) << 18; 
	frame.can_id = CAN_EFF_FLAG | aux_l;   /// DEFINE EXTENDED
	for (int i = 0; i < 8; i++) 
    {
		frame.data[i] = data[i];
	}

	if ( write(s, &frame, sizeof(frame)) != sizeof(frame) ) 
    {
		perror("Write");
		return 1;
	}

	if (close(s) < 0) 
    {
		perror("Close");
		return 1;
	}
	return 0;
}

char recieve_can(long *can_id, char *data) 
{
	int s; 
	int nbytes;
    unsigned long aux_l;
	struct sockaddr_can addr;
	struct ifreq ifr;
	struct can_frame frame;

	if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) 
    {
		perror("Socket");
		return 1;
	}

	strcpy(ifr.ifr_name, "can0");
	ioctl(s, SIOCGIFINDEX, &ifr);

	memset(&addr, 0, sizeof(addr));
	addr.can_family = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) 
    {
		perror("Bind");
		return 1;
	}

	nbytes = read(s, &frame, sizeof(frame));

	if (nbytes < 0) 
    {
		perror("Read");
		return 1;
	}

	aux_l = (frame.can_id & CAN_EFF_MASK) >> 18;
	aux_l += ((frame.can_id & CAN_EFF_MASK) & 0x0003FFFF ) <<11;
	*can_id = aux_l;

	for (int i = 0; i < 8; i++) 
    {
		data[i] = frame.data[i];
	}
           
    close(s);
	return 0;
}



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
    std::cout << "TX Lora>> " << cmd << std::endl;
    serialPuts(serial_port, cmd);
    serialPuts(serial_port, "\n");
    std::string response;
    
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
        std::cout << "Response: " << response << std::endl;
    } 
    else 
    {
        std::cout << "No response received within the timeout period." << std::endl;
    }

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
    signed int aux_int = 0;

    if (command.size() > 2) 
    {
        std::string substring = command.substr(2,4);
        std::cout << substring;
        if (substring == "0001") 
        {
            std::cout << "RX_lora>>>>>>> aciona deplyer" << std::endl;
            can_id = 0x00003006; //
            data[0] = 0x80; // l
            data[1] = 0x00; // 
            data[2] = 0x00; //
            data[3] = 0x00; //
            data[4] = 0x00; // 
            data[5] = 0x00; // 
            data[6] = 0x00; // 
            data[7] = 0x00; //         
            send_can(can_id, data);
        }
        else if (substring == "0002") 
        {
            std::cout << "RX_lora>>>>>>> desliga deplyer" << std::endl;
            can_id = 0x00003006; //
            data[0] = 0x00; // l
            data[1] = 0x00; // 
            data[2] = 0x00; //
            data[3] = 0x00; //
            data[4] = 0x00; // 
            data[5] = 0x00; // 
            data[6] = 0x00; // 
            data[7] = 0x00; //         
            send_can(can_id, data);
        }
        else if (substring == "0003") 
        {
            std::cout << "RX_lora>>>>>>> msg3" << std::endl;
        }
        else if (substring == "0004") 
        {
            std::cout << "RX_lora>>>>>>> desliga atitude" << std::endl;
            can_id = 0x00002004; //
            data[0] = 0x00; // 0
            data[1] = 0x00; // 
            data[2] = 0x00; //
            data[3] = 0x00; //
            data[4] = 0x00; // 
            data[5] = 0x00; // 
            data[6] = 0x00; // 
            data[7] = 0x00; //         
            send_can(can_id, data);
        }
        else if (substring == "0005") 
        {
            std::cout << "RX_lora>>>>>>> aciona atitude" << std::endl;
            can_id = 0x00002004; //
            data[0] = 0x80; // l
            substring = command.substr(6,4);
            strncpy(buffer, substring.c_str(), 4);
            sscanf(buffer, "%04X", &aux_int);
            buffer[4] = '\0';
            std::cout << "azimute:" << aux_int << std::endl;
            data[1] = (char)(aux_int>>8); // 
            data[2] = (char)aux_int; //
            data[3] = 0x00; //
            data[4] = 0x00; // 
            data[5] = 0x00; // 
            data[6] = 0x00; // 
            data[7] = 0x00; //         
            send_can(can_id, data);
        }
    }
}

void initializeLoRa(int serial_port) 
{
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    char command_at[200];

    std::cout << "Initializing LoRa: " << std::endl;
    sendATCommand("ATZ", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT", serial_port); // Resetando o módulo
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sendATCommand("AT", serial_port); // Resetando o módulo
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
    sendATCommand("AT+TXP=10", serial_port);
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
    gpioInitialise();
    wiringPiSetupGpio();
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN,status);

    system("sudo rm -rf /tmp/*");
    
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

    command_1 = "sudo systemctl stop serial-getty@ttyS0";
    system(command_1.c_str()); 
    int baudRate_ADSB = 115200;
    serial_port_ADSB = serialOpen("/dev/serial1", baudRate_ADSB);
    if (serial_port_ADSB < 0)
    {
        std::cerr << "Failed to open /dev/serial1" << std::endl;
        return 1;
    }
    std::cout << "Opened ADS-B serial port successfully."
            << std::endl;
    serialFlush(serial_port_ADSB);       

    std::signal(SIGINT, handle_sigint); 
    std::cout << "Running... Press Ctrl+C to interrupt.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

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
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    sprintf(command_at, "AT+SEND=251:Mauasat online %s", compilation_date);
    sendATCommand(command_at , serial_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    system("sudo ip link set can0 down"); 
    system("sudo ip link set can0 up type can bitrate 250000"); 

    time(&time_last);

        // Criar threads
    std::thread t1(can_thread,  1);
    std::thread t2(lora_thread, 2);
    std::thread t4(ADSB_thread, 4);


    std::cout << "Threads criadas, aguardando finalizacao." << std::endl;

    // Esperar as threads completarem
    t1.join();
    t2.join();
    t4.join();

    std::cout << "Todas as threads finalizaram. Programa principal encerrado." << std::endl;

    return 0;
}

void can_thread(int id) 
{
    static unsigned char contador_can = 0;
    std::cout << "can_thread " << id << " iniciada." << std::endl;
    while(1)
    {      
        contador_can = 0;

        while(contador_can < 100)
        {
            contador_can++;
            status = !status;
            digitalWrite(LED_STATUS_PIN,status);

            result = recieve_can(&can_id, data);
            if (!result) 
            {
                // Check if a frame for this can_id was already accounted for during the last second
                int frame_exists = 0;
                int pos_to_replace = -1;
                for (int i = 0; i < frames_pos; i++) 
                {
                    if (frames[i].can_id == can_id) 
                    {
                        frame_exists = 1;
                        pos_to_replace = i;
                        // printf("Replacing frame\n");
                    break;
                    }
                }

                new_frame.can_id = can_id;
                //printf(">%X \n",can_id);

                for (int i = 0; i < 8; i++) 
                {
                    new_frame.data[i] = data[i];
                }
                if (frame_exists) 
                {
                    frames[pos_to_replace] = new_frame;
                } 
                else 
                {
                    if (frames_pos < 40)
                    {
                        frames[frames_pos++] = new_frame;
                    }
                }
            }  
        }
        status = 1;
        digitalWrite(LED_STATUS_PIN,status);

        can_id_tx = 0x0000F800; //
        data_tx[0] = 0x00; // l
        data_tx[1] = 0x00; // 
        data_tx[2] = 0x00; //
        data_tx[3] = 0x00; //
        data_tx[4] = 0x00; // 
        data_tx[5] = 0x00; // 
        data_tx[6] = 0x00; // 
        data_tx[7] = ciclos; //         
        result = send_can(can_id_tx, data_tx);    

    }
    std::cout << "can_thread " << id << " finalizada." << std::endl;
}
void lora_thread(int id) 
{
    std::uint16_t area_spill = 0;
    std::uint16_t area_spill_pct = 0;
    std::uint32_t mmsi_ADSB = 0;
    int i = 0;

    size_t first_comma_pos = 0;
    size_t start_pos = 0;
    size_t end_pos = 0;
    std::string delimiter = ",";
    std::string partial_line_ADSB = "00000000000000000000000000";

    std::cout << "lora_thread " << id << " iniciada." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    while(1)
    {
        sprintf(command_at,"AT+SEND=250:[");
        for (i = 0; i < frames_pos; i++) 
        {
            sprintf(command_at,"%s\"%X:%02X%02X%02X%02X%02X%02X%02X%02X\"",command_at, frames[i].can_id, frames[i].data[0], frames[i].data[1], frames[i].data[2], frames[i].data[3], frames[i].data[4], frames[i].data[5], frames[i].data[6], frames[i].data[7]);
            if ((i == 4) || (i == 9) || (i == 14)|| (i == 19) || (i == 24)|| (i == 29))
            {
                sprintf(command_at,"%s]",command_at);
                sendATCommand(command_at , serial_port);
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                sprintf(command_at,"AT+SEND=250:[");
                sendATCommand("AT+RECVB=?", serial_port);
                std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            }
            else
            {
                if (i+1 < frames_pos)
                {
                    sprintf(command_at,"%s,",command_at); 
                }
                else
                {
                    sprintf(command_at,"%s]",command_at);
                    sendATCommand(command_at , serial_port);
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    sendATCommand("AT+RECVB=?", serial_port);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                }
            }
        }


        // tc can 0xf800
        sprintf(command_at,"AT+SEND=250:[\"%X:%02X%02X%02X%02X%02X%02X%02X%02X\"", can_id_tx, data_tx[0], data_tx[1], data_tx[2], data_tx[3], data_tx[4], data_tx[5], data_tx[6], data_tx[7]);


        // if(1) // se leu arquivo
        // {
        //     // lê arquivo
        //     std::ifstream inputFile("/home/pi/cubesat/output.txt");
        //     if (!inputFile.is_open()) 
        //     {
        //         std::cerr << "Error opening output.txt" << std::endl;
        //     }
        //     else
        //     {    
        //         std::string line_ADSB;
        //         std::getline(inputFile, line_ADSB);
        //         std::cout << line_ADSB << std::endl;
        //         inputFile.close();
        //         area_spill = 0;
        //         area_spill_pct = 0;
        //         mmsi_ADSB = 0;

        //         //std::cout << line_ADSB.length() << std::endl;
        //         start_pos = 1;
        //         end_pos = line_ADSB.find(delimiter);
        //         if (line_ADSB.length() > 9)
        //         {
        //             if (end_pos < std::string::npos)
        //             {
        //                 partial_line_ADSB = line_ADSB.substr(start_pos, end_pos - start_pos);
        //                 start_pos = end_pos + delimiter.length()+2;
        //                 end_pos = line_ADSB.find(delimiter, start_pos);
        //                 if (partial_line_ADSB.length() > 0)
        //                 {
        //                     area_spill = std::stoi(partial_line_ADSB) / 100;
        //                 }
        //                 else
        //                 {
        //                     area_spill = 0;
        //                 }
        //             // std::cout << area_spill << std::endl;
        //             }
        //             if (end_pos < std::string::npos)
        //             {
        //                 partial_line_ADSB = line_ADSB.substr(start_pos, end_pos - start_pos);
        //                 start_pos = end_pos + delimiter.length()+4;
        //                 end_pos = line_ADSB.find(delimiter, start_pos);
        //                 if (partial_line_ADSB.length() > 0)
        //                 {
        //                     area_spill_pct = std::stoi(partial_line_ADSB);
        //                 }
        //                 else
        //                 {
        //                     area_spill_pct = 0;
        //                 }
        //             // std::cout << area_spill_pct << std::endl;
        //             }
        //             if (end_pos <= std::string::npos)
        //             {
        //                 partial_line_ADSB = line_ADSB.substr(start_pos, end_pos - start_pos);
        //                 if (partial_line_ADSB.length() > 0)
        //                 {
        //                     mmsi_ADSB = (std::stoi(partial_line_ADSB));
        //                 }
        //                 else
        //                 {
        //                     mmsi_ADSB = 0;
        //                 }
        //             // std::cout << mmsi_ADSB << std::endl;
        //             }
        //         }
        //         else
        //         {
        //             std::cout << "Erro de detecção de mancha" << std::endl;
        //         }

        //         frame_ADSB.can_id  = 0x3808; //TTC_CAN ID
        //         frame_ADSB.data[0] = mmsi_ADSB >> 24;    // mmmsi high
        //         frame_ADSB.data[1] = mmsi_ADSB >> 16;  // mmmsi mid-high
        //         frame_ADSB.data[2] = mmsi_ADSB >> 8; // // mmmsi midlow
        //         frame_ADSB.data[3] = mmsi_ADSB;  // mmmsi low
        //         frame_ADSB.data[4] = (area_spill) >> 8; // area high div 100
        //         frame_ADSB.data[5] = (area_spill); // area low  div 100
        //         frame_ADSB.data[6] = area_spill_pct >> 8; // percentual high * 10
        //         frame_ADSB.data[7] = area_spill_pct; // percentual low * 10
                
        //         sprintf(command_at,"%s,\"%X:%02X%02X%02X%02X%02X%02X%02X%02X\"",command_at, frame_ADSB.can_id, frame_ADSB.data[0], frame_ADSB.data[1], frame_ADSB.data[2], frame_ADSB.data[3], frame_ADSB.data[4], frame_ADSB.data[5], frame_ADSB.data[6], frame_ADSB.data[7]);

        //     }
        // }
        // sprintf(command_at,"%s]",command_at);
        // sendATCommand(command_at , serial_port);
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        // sendATCommand("AT+RECVB=?", serial_port);
        // std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        // envia lora ADSB
        can_id_tx = 0x000600C9; //
        response_ADSB_high = response_ADSB.substr( 0, 14); std::cout <<response_ADSB_high << std::endl;
        response_ADSB_low  = response_ADSB.substr(14, 27); std::cout <<response_ADSB_low << std::endl;
        sprintf(command_at,"AT+SEND=250:[\"%X:%s\"", can_id_tx, response_ADSB_high.c_str());
        can_id_tx = 0x000600CA; //
        sprintf(command_at,"%s,\"%X:%s\"",command_at, can_id_tx, response_ADSB_low.c_str());

        sprintf(command_at,"%s]",command_at);
        sendATCommand(command_at , serial_port);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        sendATCommand("AT+RECVB=?", serial_port);
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        ciclos++;
    }
    std::cout << "lora_thread " << id << " finalizada." << std::endl;
}

void ADSB_thread(int id) 
{
    std::string response = "";
    bool responseReceived = false;
    static unsigned char contador = 0;
    
    std::cout << "ADSB_thread " << id << " iniciada." << std::endl;
    serialPuts(serial_port_ADSB, "Mauasat Receiving ADSB message...\n");
    response = "";

    while(1)
    {
        char buf[512] = {0};
        char payload[128] = {0};

        while(serialDataAvail(serial_port_ADSB)) 
        {
            char c = serialGetchar(serial_port_ADSB);
            
            if( (contador == 0) && (c != '*'))
            {
                response = "";
            }
            if (c == ';' || c == '\r' || c == '\n' || contador > 100) 
            {
                if (!response.empty()) 
                { 
                    //response += '\r';
                    //response += '\n';
                    responseReceived = true;
                    contador = 0;
                    serialFlush(serial_port_ADSB); 
                    break;
                }
            } 
            else 
            {
                
                if( c != '*')
                {
                    response += c;
                }
                contador++;
                
            }

        }  

        if (responseReceived) 
        {
            responseReceived = false;           
                         
            time_t now = time(NULL);

            // if (should_save_history()) {
            //     resave_and_send_data();
            // }
            
            // if (history_dirty) resave_and_send_data();

            response_ADSB = response;
            std::cout << "Received ADSB: " << response_ADSB << std::endl;
            serialPuts(serial_port_ADSB, "Mauasat Receiving ADSB : ");
            write(serial_port_ADSB, response.c_str(), response.length());
            response = "";
        } 

    }
    //aqui nao passa 

    std::cout << "ADSB_thread " << id << " finalizada." << std::endl;
}

//fim 
