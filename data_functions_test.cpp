#include "data_functions.h"

int main(){
	//DWORD input_nodes=5,output_nodes=5,hidden_nodes=3;
	//ACTIVATION_TYPE hidden_type=SIGMOID,output_type=TANH;
	DWORD seed=75432;
	DTYPE learning_rate=.1;
	init_genrand(seed);
	//NeuralNetwork *result=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	//randomize_weight(result);
	//print_neural_network(result);

	//write_neural_network("network.bin",result);

	//destroy_neural_network(result);


	//std::vector<std::vector<DTYPE>*>* data=read_csv("mgh002.csv",MGHMF,TRUE);
	std::vector<std::vector<DTYPE>*>* data=new std::vector<std::vector<DTYPE>*>(1);
	data[0][0]=new std::vector<DTYPE>(10);
	DWORD i;
	for(i=0;i<data[0][0]->size();i++){
		data[0][0][0][i]=5;
	}


	std::vector<std::vector<DTYPE>*>* result=sax_encoding(data,10);
	for(i=0;i<data[0][0]->size();i++){
		printf("%lf\n",result[0][0][0][i]);
	}

/*
	normalize_data(data);

	init_genrand(seed);
	std::vector<std::vector<DTYPE>*>* inputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);
	std::vector<std::vector<DTYPE>*>* outputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);
	fill_inputs_outputs(inputs,outputs,data);

	NeuralNetwork *result=read_neural_network("space0.bin");
	//gradient_descent_network(result,inputs,outputs,10,learning_rate);
	compute_gradient(result,inputs,outputs);


	DTYPE mse;
	mse=evaluate_network(result,inputs,outputs);

	printf("mse=%lf\n",mse);
	print_neural_network(result);
*/
}
