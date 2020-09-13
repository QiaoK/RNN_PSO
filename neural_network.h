#ifndef NN_H
#define NN_H
#include <vector>
#include <map>
#include <set>
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "random.h"
#define DWORD unsigned long long
#define INTEGER long long
#define WORD int
#define DTYPE double
#define ACTIVATION_TYPE unsigned int
#define TRUE 1
#define FALSE 0
#define BOOLEAN char
#define Calloc(a,b) ((b*)malloc(sizeof(b)*a))
#define Free free
#define MAX_DWORD std::numeric_limits<DWORD>::infinity()
#define MAX_DTYPE std::numeric_limits<DTYPE>::infinity()
#define SIGMOID 0x1999
#define TANH 0x2000

typedef struct{
	std::vector<DTYPE> *weights;
	std::vector<DTYPE> *weights_gradient;
}InputNode;

typedef struct{
	std::vector<DTYPE> *weights;
	std::vector<DTYPE> *hidden_weights;
	DTYPE bias;
	DTYPE context;
	DTYPE result;
	DTYPE bias_gradient;
	DTYPE alpha_gradient;
	std::vector<DTYPE> *weights_gradient;
	std::vector<DTYPE> *hidden_weights_gradient;
}HiddenNode;

typedef struct{
	DTYPE bias;
	DTYPE bias_gradient;
	DTYPE result;
}OutputNode;

typedef struct{
	std::vector<InputNode> *input_nodes;
	std::vector<HiddenNode> *hidden_nodes;
	std::vector<OutputNode> *output_nodes;
	ACTIVATION_TYPE hidden_type;
	ACTIVATION_TYPE output_type;
}NeuralNetwork;

typedef struct{
	std::vector<std::vector<DTYPE>*> *input_nodes;
	std::vector<std::vector<DTYPE>*> *hidden_nodes;
	std::vector<std::vector<DTYPE>*> *hidden_nodes_self;
	std::vector<DTYPE> *hidden_bias;
	std::vector<DTYPE> *output_bias;
}NeuralNetworkVector;

typedef struct{
	NeuralNetwork* current;
	NeuralNetworkVector* velocity;
	NeuralNetworkVector* local_best;
	DTYPE mse;
	DTYPE current_mse;
	DWORD gradient_score;
}Particle;

typedef struct{
	DTYPE input_value;
	DTYPE output_value;
}NodeCache;

typedef struct{
	std::vector<DTYPE> *hidden_nodes;
	std::vector<DTYPE> *output_nodes;
}GradientError;

extern NeuralNetwork* initialize_neural_network(DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type);
extern void randomize_weight(NeuralNetwork* network);
extern void randomize_weight2(NeuralNetwork* network, std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs);
extern void copy_neural_network_to_neural_network(NeuralNetwork* network1,NeuralNetwork* network2);
extern int compute_network(NeuralNetwork* network,std::vector<DTYPE>* input);
extern void print_neural_network(NeuralNetwork* network);
extern void destroy_neural_network(NeuralNetwork* network);
extern DTYPE evaluate_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs);
extern NeuralNetwork *particle_swarm_optimization(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha);
extern void clean_network_states(NeuralNetwork* network);
extern void compute_gradient(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs);
extern NeuralNetwork *gradient_descent(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE learning_rate,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type);
extern void gradient_descent_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE learning_rate);
extern std::vector<NeuralNetwork*>* particle_swarm_optimization_by_gradient(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha,DWORD gradient_list_size,DWORD mse_list_size,DTYPE bound);
extern DWORD gradient_score(NeuralNetwork* network,DTYPE bound,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DTYPE* mse);
extern void adam_gradient_descent_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE alpha,DTYPE beta1,DTYPE beta2,DTYPE epsilon);
extern void adam_gradient_descent_network_one_spoch(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DTYPE alpha,DTYPE beta1,DTYPE beta2,DTYPE epsilon,NeuralNetwork *v,NeuralNetwork *m,NeuralNetwork *temp,DWORD t);

extern void gradient_descent_batch(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs,DWORD epochs,DTYPE learning_rate);
extern std::vector<NeuralNetwork*>* particle_swarm_optimization_by_gradient_batch(std::vector<std::vector<std::vector<DTYPE> *>*> *inputs,std::vector<std::vector<std::vector<DTYPE> *>*> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha,DWORD gradient_list_size,DWORD mse_list_size,DTYPE bound);
extern DTYPE evaluate_network_multiple(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs);
extern void gradient_descent_batch_parallel(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs,DWORD epochs,DTYPE learning_rate);
extern DWORD average_gradient_score(NeuralNetwork* network,DTYPE bound, std::vector<std::vector<std::vector<DTYPE> *>*> *inputs,std::vector<std::vector<std::vector<DTYPE> *>*> *outputs,DTYPE* mse);

extern DTYPE evaluate_network_sign(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs);

extern DTYPE evaluate_network_last_sign(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs);
#endif
