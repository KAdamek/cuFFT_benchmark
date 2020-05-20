#!/bin/bash

#to run the script: sh call.sh memory_file.txt
#where the memory_file is a list of permited memory settings on the gpu

RED='\033[0;31m'
NC='\033[0m'

echo "Starting"
# setup your id of the card and the base of memory frequency
CARD=0
FREQ_MEM=1600
TYPE=C2C
ID=jetson
PREC=f
NUMBER=67108864
#NUMBER=33554432
#NUMBER=134217728

for LENGTH in 25 27 32 48 49 64 81 96 125 128 139 192 243 256 343 384 512 625 729 768 1024 1536 2048 2187 2401 3072 3125 4096 6144 6561 8192 12288 15625 16384 16807 19321 19683 24576 32768 49152 59049 65536 78125 98304 117649 131072 177147 196608 262144 390625 393216 524288 531441 786432 823543 1048576 1572864 1594323 1953125 2097152 2685619
do

printf "${RED}------ Running the ${LENGTH} ------${NC}\n"	

	NUMBER_A=$[ ${NUMBER}/${LENGTH} ]

	echo "../cuFFT_benchmark.exe ${LENGTH} 0 0 ${NUMBER_A} 10 ${PREC} ${TYPE} ${CARD}"
	../cuFFT_benchmark.exe ${LENGTH} 0 0 ${NUMBER_A} 10 ${PREC} ${TYPE} ${CARD}

done
# clean-up 

printf "${RED}----- That's all folks! -----${NC}\n"
