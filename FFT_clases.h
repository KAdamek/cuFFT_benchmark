#ifndef GPU_FFT_CLASES
#define GPU_FFT_CLASES


class FFT_Lengths {
public:
	size_t Nx;
	size_t Ny;
	size_t Nz;
	size_t nFFTs;
	
	FFT_Lengths(size_t t_Nx, size_t t_Ny, size_t t_Nz, size_t t_nFFTs){
		Nx    = t_Nx;
		Ny    = t_Ny;
		Nz    = t_Nz;
		nFFTs = t_nFFTs;
	}
};

class FFT_Configuration {
public:
	int FFT_precision;
	int FFT_type;
	int FFT_dimension;
	int FFT_inplace;
	int FFT_host_to_device;
	
	FFT_Configuration(int t_FFT_precision, int t_FFT_type, int t_FFT_dimension, int t_FFT_inplace, int t_FFT_host_to_device){
		FFT_precision      = t_FFT_precision;
		FFT_type           = t_FFT_type;
		FFT_dimension      = t_FFT_dimension;
		FFT_inplace        = t_FFT_inplace;
		FFT_host_to_device = t_FFT_host_to_device;
	}
};

class FFT_Sizes {
public:
	size_t total_input_FFT_size;
	size_t total_output_FFT_size;
	size_t input_bitprecision;
	size_t output_bitprecision;
	size_t input_nElements;
	size_t output_nElements;
	
	FFT_Sizes(size_t t_total_input_FFT_size, size_t t_total_output_FFT_size, int t_input_bitprecision, int t_output_bitprecision, size_t t_input_nElements, size_t t_output_nElements){
		total_input_FFT_size  = t_total_input_FFT_size;
		total_output_FFT_size = t_total_output_FFT_size;
		input_bitprecision    = t_input_bitprecision;
		output_bitprecision   = t_output_bitprecision;
		input_nElements       = t_input_nElements;
		output_nElements      = t_output_nElements;
	}
};

#endif