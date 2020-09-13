#include "neural_network.h"

int main(){
	std::vector<std::vector<DTYPE> *> *inputs=new std::vector<std::vector<DTYPE> *>;
	inputs->resize(1);
	inputs[0][0]=new std::vector<DTYPE>;
	inputs[0][0]->resize(2);
	inputs[0][0][0][0]=1;
	inputs[0][0][0][1]=2;
	std::vector<std::vector<DTYPE> *> *outputs=new std::vector<std::vector<DTYPE> *>;
	outputs->resize(1);
	outputs[0][0]=new std::vector<DTYPE>;
	outputs[0][0]->resize(1);
	outputs[0][0][0][0]=.3;
	DWORD input_nodes=2,hidden_nodes=2,output_nodes=1;
	NeuralNetwork *network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,TANH,TANH);
	network->input_nodes[0][0].weights[0][0]=.1;
	network->input_nodes[0][0].weights[0][1]=.2;
	network->input_nodes[0][1].weights[0][0]=.3;
	network->input_nodes[0][1].weights[0][1]=.4;
	network->hidden_nodes[0][0].weights[0][0]=.5;
	network->hidden_nodes[0][1].weights[0][0]=.6;
	compute_network(network,inputs[0][0]);
	print_neural_network(network);
	printf("before=%lf\n",network->output_nodes[0][0].result);
	clean_network_states(network);
	gradient_descent_network(network,inputs,outputs,100,1);
	compute_network(network,inputs[0][0]);
	print_neural_network(network);
	printf("after=%lf\n",network->output_nodes[0][0].result);
}
