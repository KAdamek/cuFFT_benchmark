#!/bin/bash

#to run the script: sh call.sh memory_file.txt
#where the memory_file is a list of permited memory settings on the gpu

echo "Starting"
# setup your id of the card and the base of memory frequency
CARD=0
FREQ_MEM=3003
LENGTH=128

rm timing-P4-${LENGTH}-0-0.txt

#run the logger on the energy profiling
	nvidia-smi --query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits -i $CARD -f nvidiasmi-P4-${LENGTH}-0-0.txt -lms 10 &
	PID=$(echo $!)
	echo "Logging process id: $PID"
	sleep 5

echo "Running the ${LENGTH}"	
#run for each permitted memory a defined cuFFT
#need to change the length of cuFFT, now is setup to run on 2GB of data, i.e. number of FFT auto computed to fit 2GB with the set length
while IFS='' read -r line || [[ -n "$line" ]]; do
#        echo "Text read from file: $line"
	CORE_MEM=$(echo $line | awk '{print $3}')
	echo $CORE_MEM
	nvidia-smi -i $CARD -ac $FREQ_MEM,$CORE_MEM
	TIMESTAMP=$(date +"%Y/%m/%d %H:%M:%S.%3N")
	nvprof --print-gpu-trace -u ms --csv ../cuFFT_benchmark.exe ${LENGTH} 0 0 -2 200 f R2C $CARD 2>out.nvprof
	START_LINE=$[$(awk '/Start/{ print NR; exit }' out.nvprof) + 3]
	#in csv mode there is no line with Regs
	#	END_LINE=$[$(awk '/Regs:/{ print NR; exit }' out.nvprof) - 3]
	END_LINE=$[$(cat out.nvprof | wc -l) - 1]
	START_TIME=$(head -n $START_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $1}')
	END_PLUS_TIME=$(head -n $END_LINE out.nvprof | tail -n 1 | awk -F "," '{print $2}')
	sleep 5
	echo ${TIMESTAMP}"," ${START_TIME}"," ${END_TIME}"," ${END_PLUS_TIME}"," ${CORE_MEM}>> timing-P4-${LENGTH}-0-0.txt
done < "$1"

# clean-up 
nvidia-smi -i $CARD -rac
kill -2 $PID
