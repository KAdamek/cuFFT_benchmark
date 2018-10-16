#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

class Performance_results{
public:
	char filename[200];
	double GPU_time;
	double TR_time;
	int Nx;
	int Ny;
	int Nz;
	int nFFTs;
	int nRuns;
	double modGB;
	int FFT_dimension;
	char FFT_type[10];
	char FFT_precision[10];
	
	Performance_results() {
		GPU_time = 0;
	}
	
	void Save(){
		ofstream FILEOUT;
		FILEOUT.open (filename, std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << Nx << " " << Ny << " " << Nz << " " << nFFTs << " " << nRuns << " " << modGB << " " << FFT_dimension << " " << GPU_time << " " << TR_time << " " << FFT_type << " " << FFT_precision << endl;
		FILEOUT.close();
	}
	
	void Print(){
		cout << std::fixed << std::setprecision(8) << Nx << " " << Ny << " " << Nz << " " << nFFTs << " " << nRuns << " " << modGB << " " << FFT_dimension << " " << GPU_time << " " << TR_time << " " << FFT_type << " " << FFT_precision << endl;
	}
	
	void Assign(int t_Nx, int t_Ny, int t_Nz, int t_nFFTs, int t_nRuns, double t_modGB, int t_FFT_dimension, const char *t_FFT_type, const char *t_FFT_precision, const char *t_filename){
		Nx    = t_Nx;
		Ny    = t_Ny;
		Nz    = t_Nz;
		nFFTs = t_nFFTs;
		nRuns = t_nRuns;
		modGB = t_modGB;
		FFT_dimension = t_FFT_dimension;

		sprintf(FFT_type,"%s",t_FFT_type);
		sprintf(FFT_precision,"%s",t_FFT_precision);
		sprintf(filename,"%s", t_filename);
	}
	
};
