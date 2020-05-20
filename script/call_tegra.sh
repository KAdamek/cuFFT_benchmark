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

for LENGTH in 32 #768 1536 3072 6144 12288 24576 49152 98304 196608 393216 786432 1572864 #64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152
#for LENGTH in 49 82 139 232 243 343 625 729 1331 2187 2491 6561 14641 16807 19321 19683 59049 117649 161051 531441 823 543 17771561 2685619
do
rm timing-${ID}-${PREC}-${TYPE}-${LENGTH}-0-0.txt

#run the logger on the energy profiling
	printf "${RED}Tegrastats run ---${NC}"
	tegrastats --interval 10 | while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S.%3N')" "$line"; done > tegrastats-${ID}-${PREC}-${TYPE}-${LENGTH}-0-0.txt &
	printf "${RED}--- done.${NC}\n"
	sleep 5

printf "${RED}------ Running the ${LENGTH} ------${NC}\n"	
#run for each permitted memory a defined cuFFT
#need to change the length of cuFFT, now is setup to run on 2GB of data, i.e. number of FFT auto computed to fit 2GB with the set length

	NUMBER_A=$[ ${NUMBER}/${LENGTH} ]

#for CORE_MEM in 76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000
for CORE_MEM in 921600000 844800000 768000000 691200000 614400000 537600000 460800000 384000000 307200000 230400000 153600000 76800000
do
	echo ${CORE_MEM} | tee -a /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq
	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
	echo "nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} 0 0 ${NUMBER_A} 200 ${PREC} ${TYPE} ${CARD} 2>out.nvprof"
	nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} 0 0 ${NUMBER_A} 200 ${PREC} ${TYPE} ${CARD} 2>out.nvprof
	START_LINE=$[$(awk '/Start/{ print NR; exit }' out.nvprof) + 3]
	#in csv mode there is no line with Regs
	#	END_LINE=$[$(awk '/Regs:/{ print NR; exit }' out.nvprof) - 3]
	END_LINE=$[$(cat out.nvprof | wc -l) - 1]
	START_TIME=$(head -n $START_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_PLUS_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $2}')
	sleep 5
	echo ${TIMESTAMP}"," ${START_TIME}"," ${END_TIME}"," ${END_PLUS_TIME}"," ${CORE_MEM}>> timing-${ID}-${PREC}-${TYPE}-${LENGTH}-0-0.txt

#while IFS='' read -r line || [[ -n "$line" ]]; do
#        echo "Text read from file: $line"
#	CORE_MEM=$(echo $line | awk '{print $3}')
#	echo $CORE_MEM
#	nvidia-smi -i $CARD -ac $FREQ_MEM,$CORE_MEM
#	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
#	nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} 0 0 -2 200 ${PREC} ${TYPE} ${CARD} 2>out.nvprof
#	START_LINE=$[$(awk '/Start/{ print NR; exit }' out.nvprof) + 3]
#	#in csv mode there is no line with Regs
#	#	END_LINE=$[$(awk '/Regs:/{ print NR; exit }' out.nvprof) - 3]
#	END_LINE=$[$(cat out.nvprof | wc -l) - 1]
#	START_TIME=$(head -n $START_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
#	END_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
#	END_PLUS_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $2}')
#	sleep 5
#	echo ${TIMESTAMP}"," ${START_TIME}"," ${END_TIME}"," ${END_PLUS_TIME}"," ${CORE_MEM}>> timing-${ID}-${PREC}-${TYPE}-${LENGTH}-0-0.txt
#done < "$1"
done

# clean-up 
tegrastats --stop
echo 921600000 | tee -a /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq

printf "${RED}----- That's all folks! -----${NC}\n"
done
