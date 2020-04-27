#!/bin/bash

#to run the script: sh call.sh memory_file.txt
#where the memory_file is a list of permited memory settings on the gpu

RED='\033[0;31m'
NC='\033[0m'

echo "Starting"
# setup your id of the card and the base of memory frequency
CARD=0
FREQ_MEM=3003
TYPE=C2C
ID=jetson
PREC=f
CORE_MEM=900
NUMBER=67108864

for LENGTH in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152
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

	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
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

# clean-up 
tegrastats --stop

printf "${RED}----- That's all folks! -----${NC}\n"
done
