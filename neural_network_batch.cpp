#include <math.h>
#include "neural_network.h"
#include "heap.h"

DWORD average_gradient_score(NeuralNetwork* network,DTYPE bound,std::vector<std::vector<std::vector<DTYPE> *>*> *inputs){
	DWORD i;
	DWORD score=0;
	for(i=0;i<inputs->size();i++){
		score+=gradient_score(network,0,inputs[0][i]);
	}
	return score/inputs->size();

}

void update_network_gradient_by_copy_gradient(NeuralNetwork *network,NeuralNetwork *copy){
	DWORD j,k;
	//gradients between input and hidden layer
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]+=copy->input_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient+=copy->hidden_nodes[0][j].bias_gradient;
		network->hidden_nodes[0][j].alpha_gradient+=copy->hidden_nodes[0][j].alpha_gradient;
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]+=copy->hidden_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient+=copy->output_nodes[0][j].bias_gradient;
	}

}

void average_gradient(NeuralNetwork *network,DWORD sample_size){
	DWORD j,k;
	//gradients between input and hidden layer
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]/=sample_size;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient/=sample_size;
		network->hidden_nodes[0][j].alpha_gradient/=sample_size;
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]/=sample_size;
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient/=sample_size;
	}
}


void gradient_descent_batch(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs,DWORD epochs,DTYPE learning_rate){
	NeuralNetwork *copy=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
	copy_neural_network_to_neural_network(network,copy);
	DWORD i,j;
	for(i=0;i<epochs;i++){
		for(j=0;j<inputs->size();j++){
			compute_gradient(copy,inputs[0][j],outputs[0][j]);
			update_network_gradient_by_copy_gradient(network,copy);
		}
		average_gradient(network,inputs->size());
		update_network_by_gradient(network,learning_rate);
		clean_network_states(copy);
		copy_neural_network_to_neural_network(network,copy);
	}
}
