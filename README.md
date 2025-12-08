# =============== Linux ===============

# MobileSim install:
sudo apt install ./mobilesim_0.9.8+ubuntu16_amd64.deb 

# C++:
sudo apt update && sudo apt install g++
C/C++ vscode extension (Microsoft)


# ARIA
sudo dpkg -i libaria_2.9.4+ubuntu16_amd64.deb 

# Build
make
chmod +x ./build/main

# Run
./build/main
  
# Detele the build (os aconselho a sempre rodar esse comando antes do make)
make clean

# =============== Windows ===============

Install on this order:

msys2-x86_64-20250830.exe

MobileSim-0.7.5.exe

ARIA-2.9.1-1-x64.exe


Open the MSYS terminal an run:

Installing the compiler:
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc

Testing the installation
export PATH="/c/msys64/ucrt64/bin:$PATH"
g++ --version

Adding to the PATH, permanently
nano ~/.bashrc
export PATH="/c/msys64/ucrt64/bin:$PATH"   //(Add this to the end of the file)
source ~/.bashrc
