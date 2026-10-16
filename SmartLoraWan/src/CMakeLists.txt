cmake_minimum_required(VERSION 3.10)

# Nome do projeto
project(cubesat_obc)

# Definir o padrão do compilador C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Encontrar a biblioteca OpenCV
#find_package(OpenCV REQUIRED)

# Encontrar a biblioteca JSONCpp
#find_package(jsoncpp REQUIRED)

# Encontra o pacote de threads do sistema
find_package(Threads REQUIRED)

# Incluir diretórios de cabeçalhos
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(/usr/include/jsoncpp/json)
include_directories(/usr/include/wiringPi) # Caso wiringPi esteja no diretório padrão
#include_directories(/home/pi/RadioLib-6.3.0/src) # Diretório do RadioLib
include_directories(/usr/local/include/pigpio)   # Diretório do pigpio
#include_directories(/usr/include/opencv4)

#link_directories(/home/pi/RadioLib-6.3.0/build) # Diretório onde as bibliotecas estão localizadas

# Adicionar o executável
add_executable(cubesat_obc main.cpp)

# Linka a biblioteca de threads ao seu executável
target_link_libraries(cubesat_obc Threads::Threads)

# Linkar bibliotecas
target_link_libraries(cubesat_obc wiringPi pigpio)

