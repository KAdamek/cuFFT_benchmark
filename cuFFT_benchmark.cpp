#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "FFT_clases.h"
#include "debug.h"
#include "params.h"
#include "results.h"


//-----------------------------------------------
//---------- Signal generation
void Generate_signal_noise(float2 *h_input, int N, int nFFTs){
	for(int f=0; f<nFFTs; f++){
		for(int x=0; x<N; x++){
			h_input[N*f + x].y=rand() / (float)RAND_MAX;
			h_input[N*f + x].x=rand() / (float)RAND_MAX;
		}
	}
}
//------------------------------------------------<
int cuFFT_1D_C2C_half(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_R2C_half(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_C2R_half(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_C2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_R2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_C2R_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_C2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_R2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_1D_C2R_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);

int cuFFT_2D_C2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_2D_R2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_2D_C2R_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_2D_C2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_2D_R2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_2D_C2R_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);

int cuFFT_3D_C2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_3D_R2C_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_3D_C2R_float(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_3D_C2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_3D_R2C_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);
int cuFFT_3D_C2R_double(FFT_Lengths FFT_lengths, size_t nFFTs, int nRuns, int device, FFT_Configuration FFT_conf, FFT_Sizes FFT_size, double *execution_time, double *standard_deviation, double *transfer_time);


int main(int argc, char* argv[]) {
	
	if(argc!=9) {
		printf("Argument error!\n");
		printf("FFT sizes:\n");
		printf("    for 1D set Nx to non-zero, and Ny=Nz=0\n");
		printf("    for 2D set Nx,Ny to non-zero, and Nz=0\n");
		printf("    for 3D set all Nx,Ny,Nz\n");
		printf("1) FFT length in x [Nx]\n");
		printf("2) FFT length in y [Ny]\n");
		printf("3) FFT length in z [Nz]\n");
		printf("4) Number of FFTs [Nf] (if Nf<0 then Nf will be set to fill that much GB of memory. That is Nf=-1 means 1GB of memory)\n");
		printf("5) Number of runs of the kernel\n");
		printf("Select double \"d\" or single \"f\" precision\n");
		printf("6) precision\n");
		printf("Select FFT type complex-to-complex \"C2C\"; real-to-complex \"R2C\"; complex-to-real \"C2R\"\n");
		printf("7) FFT type\n");
		printf("8) Device id\n");
        return (1);
    }
	char * pEnd;
	
	int Nx = strtol(argv[1],&pEnd,10);
	int Ny = strtol(argv[2],&pEnd,10);
	int Nz = strtol(argv[3],&pEnd,10);
	int nFFTs = strtol(argv[4],&pEnd,10);
	int nRuns = strtol(argv[5],&pEnd,10);
	int device = strtol(argv[8],&pEnd,10);
	char input_precision = '0';
	if (strlen(argv[6])!=1) {
		printf("ERROR: Argument numerical precision is too long\n");
		return(1);
	}
	input_precision=*argv[6];
	if (strlen(argv[7])!=3){
		printf("ERROR: FFT type has wrong format\n");
		return(1);
	}
	char input_type[100];
	sprintf(input_type,"%s",argv[7]);
	int FFT_precision = 0;
	int FFT_type = 0;
	int FFT_dimension = 0;
	int FFT_inplace = 0;
	int FFT_host_to_device = 1;
	
	if(input_precision=='d') FFT_precision = FFT_PRECISION_DOUBLE;
	else if(input_precision=='f') FFT_precision = FFT_PRECISION_FLOAT;
	else if(input_precision=='h') FFT_precision = FFT_PRECISION_HALF;
	else {
		printf("ERROR: Unknown number precision. Select 'd' for double or 'f' for float.\n");
		return(1);
	}
	
	if(Nx>0 && Ny==0 && Nz==0) FFT_dimension = 1;
	else if(Nx>0 && Ny>0 && Nz==0) FFT_dimension = 2;
	else if(Nx>0 && Ny>0 && Nz>0) FFT_dimension = 3;
	else {
		printf("ERROR: Wrong dimension of the FFT.\n");
		return(1);
	}
	if(Nx==0) Nx=1;
	if(Ny==0) Ny=1;
	if(Nz==0) Nz=1;

	size_t input_bitprecision = 0;
	size_t output_bitprecision = 0;
	size_t one_input_FFT_size = 0;
	size_t one_output_FFT_size = 0;	
	
	if(strcmp(input_type,"C2C")==0) {
		FFT_type = FFT_TYPE_C2C;
		input_bitprecision = 2*FFT_precision;
		output_bitprecision = 2*FFT_precision;
		one_input_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) Nz)*input_bitprecision;
		one_output_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) Nz)*output_bitprecision;
	}
	else if(strcmp(input_type,"R2C")==0) {
		FFT_type = FFT_TYPE_R2C;
		input_bitprecision = FFT_precision;
		output_bitprecision = 2*FFT_precision;
		if(FFT_dimension==1) {
			one_input_FFT_size = ((size_t) Nx)*input_bitprecision;
			one_output_FFT_size = ((size_t) (Nx/2+1))*output_bitprecision;
		}
		else if(FFT_dimension==2) {
			one_input_FFT_size = ((size_t) Nx)*((size_t) Ny)*input_bitprecision;
			one_output_FFT_size = ((size_t) Nx)*((size_t) (Ny/2+1))*output_bitprecision;
		}
		else if(FFT_dimension==3) {
			one_input_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) Nz)*input_bitprecision;
			one_output_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) (Nz/2+1))*output_bitprecision;
		}
	}
	else if(strcmp(input_type,"C2R")==0) {
		FFT_type = FFT_TYPE_C2R;
		input_bitprecision = 2*FFT_precision;
		output_bitprecision = FFT_precision;
		if(FFT_dimension==1) {
			one_input_FFT_size = ((size_t) Nx)*input_bitprecision;
			one_output_FFT_size = ((size_t) Nx)*output_bitprecision;
		}
		else if(FFT_dimension==2) {
			one_input_FFT_size = ((size_t) Nx)*((size_t) Ny)*input_bitprecision;
			one_output_FFT_size = ((size_t) Nx)*((size_t) Ny)*output_bitprecision;
		}
		else if(FFT_dimension==3) {
			one_input_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) Nz)*input_bitprecision;
			one_output_FFT_size = ((size_t) Nx)*((size_t) Ny)*((size_t) Nz)*output_bitprecision;
		}
	}
	else {
		printf("ERROR: wrong FFT type.");
		return(1);
	}
	
	size_t mem_size = one_input_FFT_size*((size_t) nFFTs);
	size_t total_input_FFT_size = 0;
	size_t total_output_FFT_size = 0;
	double modGB = 1.0;
	if(nFFTs<0){
		nFFTs = abs(nFFTs);
		mem_size = ((size_t) nFFTs)*1024*1024*1024;
		nFFTs = (int) (mem_size/one_input_FFT_size);
		if(nFFTs<1) nFFTs=1;
		printf("mem_size=%zu; nFFTs=%d;\n", mem_size, nFFTs);
	}
	total_input_FFT_size = one_input_FFT_size*((size_t) nFFTs);
	total_output_FFT_size = one_output_FFT_size*((size_t) nFFTs);
	modGB = ((double) total_input_FFT_size)/((double) mem_size);
	
	if(DEBUG){
		printf("----------- DEBUG -----------\n");
		printf("Selected parameters:\n");
		printf("Performing %dD FFT;\n", FFT_dimension);
		printf("Nx = %d;\n", Nx);
		printf("Ny = %d;\n", Ny);
		printf("Nz = %d;\n", Nz);
		printf("Number of FFTs: %d;\n", nFFTs);
		printf("Number of runs: %d;\n", nRuns);
		printf("Precision: "); if(FFT_precision==FFT_PRECISION_DOUBLE) printf("double;\n"); else if(FFT_precision==FFT_PRECISION_FLOAT) printf("single;\n"); else  if(FFT_precision==FFT_PRECISION_HALF) printf("half;\n"); else printf("unknown;\n");
		printf("FFT type: "); if(FFT_type==FFT_TYPE_C2C) printf("complex-to-complex;\n"); else if(FFT_type==FFT_TYPE_R2C) printf("real-to-complex;\n"); else if(FFT_type==FFT_TYPE_C2R) printf("complex-to-real;\n");
		printf("Device: %d;\n", device);
		printf("----------\n");
		printf("Individual FFT sizes: I: %zu; O: %zu;\n", one_input_FFT_size/input_bitprecision, one_input_FFT_size/output_bitprecision);
		printf("Number of elements: I:%zu; O:%zu\n", total_input_FFT_size/input_bitprecision, total_output_FFT_size/output_bitprecision);
		printf("Bit precisions: I:%zu; O:%zu;\n", input_bitprecision, output_bitprecision);
		printf("Memory assigned for the FFTs: %f MB;\n", ((double) mem_size)/(1024.0*1024.0));
		printf("Memory taken by FFTs: I:%f MB; O:%f MB\n", ((double) total_input_FFT_size)/(1024.0*1024.0), ((double) total_output_FFT_size)/(1024.0*1024.0));
		printf("Memory modificator: %f;\n", modGB);
		printf("-----------------------------<\n");
	}
	
	// Creating classes
	FFT_Lengths FFT_lengths(Nx, Ny, Nz, nFFTs);
	FFT_Configuration FFT_conf(FFT_precision, FFT_type, FFT_dimension, FFT_inplace, FFT_host_to_device);
	FFT_Sizes FFT_size(total_input_FFT_size, total_output_FFT_size, input_bitprecision, output_bitprecision, total_input_FFT_size/input_bitprecision, total_output_FFT_size/output_bitprecision);
	
	// Performance measurements
	double FFT_execution_time = 0, FFT_standard_deviation = 0, FFT_transfer_time = 0;
	char str_FFT_type[20], str_FFT_precision[20];
	if(FFT_precision==FFT_PRECISION_DOUBLE) sprintf(str_FFT_precision,"double"); else sprintf(str_FFT_precision, "single");
	if(FFT_type==FFT_TYPE_C2C) sprintf(str_FFT_type, "C2C"); else if(FFT_type==FFT_TYPE_R2C) sprintf(str_FFT_type, "R2C"); else if(FFT_type==FFT_TYPE_C2R) sprintf(str_FFT_type, "C2R");
	Performance_results cuFFT_results;
	cuFFT_results.Assign(Nx, Ny, Nz, nFFTs, nRuns, modGB, FFT_dimension, str_FFT_type, str_FFT_precision, "SKA_cuFFT_results.dat");
	
	// Rozdelit na podle FFT dimensions. Uvnitr nich to rozdelit na float nebo double a pak na C2C, C2R and R2C.
	if(FFT_dimension==1) {
		if(FFT_precision==FFT_PRECISION_FLOAT){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_1D_C2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_1D_R2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_1D_C2R_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");
		}
		else if(FFT_precision==FFT_PRECISION_DOUBLE){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_1D_C2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_1D_R2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_1D_C2R_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");				
		}
		else if(FFT_precision==FFT_PRECISION_HALF){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_1D_C2C_half(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_1D_R2C_half(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_1D_C2R_half(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");				
		}
		else printf("ERROR: wrong FFT precision.\n");
	}
	else if(FFT_dimension==2) {
		if(FFT_precision==FFT_PRECISION_FLOAT){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_2D_C2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_2D_R2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_2D_C2R_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");
		}
		else if(FFT_precision==FFT_PRECISION_DOUBLE){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_2D_C2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_2D_R2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_2D_C2R_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");				
		}
		else printf("ERROR: wrong FFT precision.\n");
	}
	else if(FFT_dimension==3) {
		if(FFT_precision==FFT_PRECISION_FLOAT){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_3D_C2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_3D_R2C_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_3D_C2R_float(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");
		}
		else if(FFT_precision==FFT_PRECISION_DOUBLE){
			if(FFT_type==FFT_TYPE_C2C)      cuFFT_3D_C2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_R2C) cuFFT_3D_R2C_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else if(FFT_type==FFT_TYPE_C2R) cuFFT_3D_C2R_double(FFT_lengths, nFFTs, nRuns, device, FFT_conf, FFT_size, &FFT_execution_time, &FFT_standard_deviation, &FFT_transfer_time);
			else printf("ERROR: wrong FFT_type.\n");				
		}
		else printf("ERROR: wrong FFT precision.\n");
	}
	
	cuFFT_results.GPU_time = FFT_execution_time;
	cuFFT_results.GPU_stdev = FFT_standard_deviation;
	cuFFT_results.TR_time  = FFT_transfer_time;
	if(VERBOSE) printf("     cuFFT Execution time:\033[32m%0.3f\033[0mms\n", cuFFT_results.GPU_time);
	if(VERBOSE) printf("     cuFFT Standard deviation:\033[32m%0.3f\033[0mms\n", cuFFT_results.GPU_stdev);
	if(VERBOSE) printf("     cuFFT Transfer time:\033[32m%0.3f\033[0mms\n", cuFFT_results.TR_time);
	cuFFT_results.Save();

	return (0);
}
