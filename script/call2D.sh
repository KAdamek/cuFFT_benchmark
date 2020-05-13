#!/bin/bash

#to run the script: sh call.sh memory_file.txt
#where the memory_file is a list of permited memory settings on the gpu

RED='\033[0;31m'
NC='\033[0m'

echo "Starting"
# setup your id of the card and the base of memory frequency
CARD=1
FREQ_MEM=5005
TYPE=C2C
ID=XP
PREC=h



for LENGTH in 32 64 128 256 512 1024 2048 4096 8192 16384 #32768 65536 131072 262144 524288 1048576 2097152
do
rm timing-${ID}-${PREC}-${TYPE}-${LENGTH}-${LENGTH}-0.txt

#run the logger on the energy profiling
	nvidia-smi --query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits -i $CARD -f nvidiasmi-${ID}-${PREC}-${TYPE}-${LENGTH}-${LENGTH}-0.txt -lms 10 &
	PID=$(echo $!)
	echo "Logging process id: $PID"
	sleep 5

printf "${RED}------ Running the ${LENGTH} ------${NC}\n"	
#run for each permitted memory a defined cuFFT
#need to change the length of cuFFT, now is setup to run on 2GB of data, i.e. number of FFT auto computed to fit 2GB with the set length
while IFS='' read -r line || [[ -n "$line" ]]; do
#        echo "Text read from file: $line"
	CORE_MEM=$(echo $line | awk '{print $3}')
	echo $CORE_MEM
	nvidia-smi -i $CARD -ac $FREQ_MEM,$CORE_MEM
	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
	nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} ${LENGTH} 0 -2 200 ${PREC} ${TYPE} $CARD 2>out.nvprof
	START_LINE=$[$(awk '/Start/{ print NR; exit }' out.nvprof) + 3]
	#in csv mode there is no line with Regs
	#	END_LINE=$[$(awk '/Regs:/{ print NR; exit }' out.nvprof) - 3))
	END_LINE=$[$(cat out.nvprof | wc -l) - 1]
	START_TIME=$(head -n $START_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_PLUS_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $2}')
	sleep 5
	echo ${TIMESTAMP}"," ${START_TIME}"," ${END_TIME}"," ${END_PLUS_TIME}"," ${CORE_MEM}>> timing-${ID}-${PREC}-${TYPE}-${LENGTH}-${LENGTH}-0.txt
done < "$1"

# clean-up 
nvidia-smi -i $CARD -rac
kill -2 $PID
done
