#ifndef DATA_FUNCTIONS_H
#define DATA_FUNCTIONS_H

#include "neural_network.h"
#define MGHMF 0x15134
#define SPACE_SHUTTLE 0x15135
#define UNIVARIATE 0x15136

typedef struct{
	DTYPE upper;
	DTYPE lower;
}Bound;


extern std::vector<std::vector<DTYPE>*>* read_csv(const char* filename,WORD type,BOOLEAN header);
extern void normalize_data(std::vector<std::vector<DTYPE>*>* data);
extern void soft_normalize_data(std::vector<std::vector<DTYPE>*>* data,DTYPE lower,DTYPE upper);
extern NeuralNetwork* read_neural_network(const char* filename);
extern void write_neural_network(const char* filename,NeuralNetwork *network);
extern std::vector<std::vector<std::vector<DTYPE>*>*>* read_MGH_data();
extern void fill_inputs_outputs(std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,std::vector<std::vector<DTYPE>*>* data);
extern std::vector<std::vector<DTYPE>*>* filter_data(std::vector<std::vector<DTYPE>*>* data,DTYPE lower_bound,DTYPE upper_bound);
extern void fill_inputs_outputs_multiple(std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,std::vector<std::vector<DTYPE>*>* data,DWORD n_output);
extern std::vector<std::vector<DTYPE>*>* sax_encoding(std::vector<std::vector<DTYPE>*>* data,DWORD intervals);
extern std::vector<std::vector<std::vector<DTYPE>*>*>* read_MGH_data(DWORD start,DWORD end);
extern void clean_2d_data(std::vector<std::vector<DTYPE>*>* data);
extern void clean_3d_data(std::vector<std::vector<std::vector<DTYPE>*>*> *data);
extern void read_features(const char* filename,std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,DWORD input_size,DWORD output_size);
extern void read_feature_meta(const char* filename,std::vector<char*> *types,std::vector<std::vector<char*>*>* stock_codes);
extern void get_feature_dimensions(const char* filename,DWORD* input_size,DWORD* output_size);
extern char* make_stock_filename(char* type,char* code);
extern std::vector<char*>* read_stock_list(const char* filename);
#endif
