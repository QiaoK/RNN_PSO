#include <math.h>
#include <omp.h>
#include "neural_network.h"
#include "heap.h"
#define PROCS 6
#define LAMBDA 0.01

bool mse_compare(Particle* p1,Particle* p2){
	return p1->current_mse<p2->current_mse;
}

bool gradient_compare(Particle* p1,Particle* p2){
	return p1->gradient_score>p2->gradient_score;
}

NeuralNetwork* initialize_neural_network(DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type){
	NeuralNetwork* result=Calloc(1,NeuralNetwork);
	result->hidden_type=hidden_type;
	result->output_type=output_type;
	result->input_nodes=new std::vector<InputNode>;
	result->hidden_nodes=new std::vector<HiddenNode>;
	result->output_nodes=new std::vector<OutputNode>;
	result->input_nodes->resize(input_nodes);
	result->hidden_nodes->resize(hidden_nodes);
	result->output_nodes->resize(output_nodes);
	DWORD i;
	for(i=0;i<result->input_nodes->size();i++){
		result->input_nodes[0][i].weights=new std::vector<DTYPE>(hidden_nodes);
		result->input_nodes[0][i].weights_gradient=new std::vector<DTYPE>(hidden_nodes);
		//printf("i=%lld,size=%ld\n",i,result->input_nodes[0][i].weights->size());
		//result->hidden_nodes[0][i].context=0;
	}
	for(i=0;i<result->hidden_nodes->size();i++){
		result->hidden_nodes[0][i].weights=new std::vector<DTYPE>(output_nodes);
		result->hidden_nodes[0][i].weights_gradient=new std::vector<DTYPE>(output_nodes);
		result->hidden_nodes[0][i].hidden_weights=new std::vector<DTYPE>(hidden_nodes);
		result->hidden_nodes[0][i].hidden_weights_gradient=new std::vector<DTYPE>(hidden_nodes);
		result->hidden_nodes[0][i].context=0;
	}
	return result;
}

NeuralNetworkVector *initialize_neural_network_vector(DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes){
	NeuralNetworkVector* result=Calloc(1,NeuralNetworkVector);
	result->input_nodes=new std::vector<std::vector<DTYPE>*>(input_nodes);
	result->hidden_nodes=new std::vector<std::vector<DTYPE>*>(hidden_nodes);
	result->hidden_nodes_self=new std::vector<std::vector<DTYPE>*>(hidden_nodes);

	result->hidden_bias=new std::vector<DTYPE>;
	result->output_bias=new std::vector<DTYPE>;

	result->output_bias->resize(output_nodes,0);
	result->hidden_bias->resize(hidden_nodes,0);
	DWORD i;
	for(i=0;i<input_nodes;i++){
		result->input_nodes[0][i]=new std::vector<DTYPE>(hidden_nodes,0);
	}
	for(i=0;i<hidden_nodes;i++){
		result->hidden_nodes[0][i]=new std::vector<DTYPE>(output_nodes,0);
		result->hidden_nodes_self[0][i]=new std::vector<DTYPE>(hidden_nodes,0);
	}
	return result;
}

void destroy_neural_network(NeuralNetwork* network){
	DWORD i;
	for(i=0;i<network->input_nodes->size();i++){
		delete network->input_nodes[0][i].weights;
		delete network->input_nodes[0][i].weights_gradient;
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		delete network->hidden_nodes[0][i].weights;
		delete network->hidden_nodes[0][i].hidden_weights;
		delete network->hidden_nodes[0][i].weights_gradient;
		delete network->hidden_nodes[0][i].hidden_weights_gradient;
	}
	delete network->input_nodes;
	delete network->hidden_nodes;
	delete network->output_nodes;
	Free(network);
}

void destroy_neural_network_vector(NeuralNetworkVector* network){
	DWORD i;
	for(i=0;i<network->input_nodes->size();i++){
		delete network->input_nodes[0][i];
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		delete network->hidden_nodes[0][i];
	}
	for(i=0;i<network->hidden_nodes_self->size();i++){
		delete network->hidden_nodes_self[0][i];
	}
	delete network->input_nodes;
	delete network->hidden_nodes;
	delete network->hidden_nodes_self;
	delete network->hidden_bias;
	delete network->output_bias;
	Free(network);
}

void copy_neural_network_to_neural_network(NeuralNetwork* network1,NeuralNetwork* network2){
	DWORD i,j;
	for(i=0;i<network1->input_nodes->size();i++){
		for(j=0;j<network1->input_nodes[0][i].weights->size();j++){
			network2->input_nodes[0][i].weights[0][j]=network1->input_nodes[0][i].weights[0][j];
		}
	}
	for(i=0;i<network1->hidden_nodes->size();i++){
		for(j=0;j<network1->hidden_nodes[0][i].weights->size();j++){
			network2->hidden_nodes[0][i].weights[0][j]=network1->hidden_nodes[0][i].weights[0][j];
		}
		for(j=0;j<network1->hidden_nodes[0][i].hidden_weights->size();j++){
			network2->hidden_nodes[0][i].hidden_weights[0][j]=network1->hidden_nodes[0][i].hidden_weights[0][j];
		}
		network2->hidden_nodes[0][i].bias=network1->hidden_nodes[0][i].bias;
	}
	for(i=0;i<network1->output_nodes->size();i++){
		network2->output_nodes[0][i].bias=network1->output_nodes[0][i].bias;
	}
}


void copy_neural_network_to_vector(NeuralNetwork* network,NeuralNetworkVector* vector){
	DWORD i,j;
	for(i=0;i<network->input_nodes->size();i++){
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			vector->input_nodes[0][i][0][j]=network->input_nodes[0][i].weights[0][j];
		}
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			vector->hidden_nodes[0][i][0][j]=network->hidden_nodes[0][i].weights[0][j];
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			vector->hidden_nodes_self[0][i][0][j]=network->hidden_nodes[0][i].hidden_weights[0][j];
		}
		vector->hidden_bias[0][i]=network->hidden_nodes[0][i].bias;
	}
	for(i=0;i<network->output_nodes->size();i++){
		vector->output_bias[0][i]=network->output_nodes[0][i].bias;
	}
}

void copy_vector_to_neural_network(NeuralNetworkVector* vector,NeuralNetwork* network){
	DWORD i,j;
	for(i=0;i<network->input_nodes->size();i++){
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			network->input_nodes[0][i].weights[0][j]=vector->input_nodes[0][i][0][j];
		}
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			network->hidden_nodes[0][i].weights[0][j]=vector->hidden_nodes[0][i][0][j];
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			network->hidden_nodes[0][i].hidden_weights[0][j]=vector->hidden_nodes_self[0][i][0][j];
		}
		network->hidden_nodes[0][i].bias=vector->hidden_bias[0][i];
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].bias=vector->output_bias[0][i];
	}
}

void copy_vector_to_vector(NeuralNetworkVector* v1,NeuralNetworkVector* v2){
	DWORD i,j;
	for(i=0;i<v1->input_nodes->size();i++){
		for(j=0;j<v1->input_nodes[0][i]->size();j++){
			v2->input_nodes[0][i][0][j]=v1->input_nodes[0][i][0][j];
		}
	}
	for(i=0;i<v1->hidden_nodes->size();i++){
		for(j=0;j<v1->hidden_nodes[0][i]->size();j++){
			v2->hidden_nodes[0][i][0][j]=v1->hidden_nodes[0][i][0][j];
		}
		for(j=0;j<v1->hidden_nodes_self[0][i]->size();j++){
			v2->hidden_nodes_self[0][i][0][j]=v1->hidden_nodes_self[0][i][0][j];
		}
		v2->hidden_bias[0][i]=v1->hidden_bias[0][i];
	}
	for(i=0;i<v1->output_bias->size();i++){
		v2->output_bias[0][i]=v1->output_bias[0][i];
	}
}

void randomize_weight(NeuralNetwork* network){
	DWORD i,j;
	for(i=0;i<network->input_nodes->size();i++){
		//printf("i=%lld,size=%ld\n",i,network->input_nodes[0][i].weights->size());
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){

			network->input_nodes[0][i].weights[0][j]=(2*genrand_real2()-1);
		}
	}

	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			network->hidden_nodes[0][i].weights[0][j]=(2*genrand_real2()-1);
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			network->hidden_nodes[0][i].hidden_weights[0][j]=(2*genrand_real2()-1);
		}
		network->hidden_nodes[0][i].bias=(2*genrand_real2()-1);
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].bias=(2*genrand_real2()-1);
	}
}

DTYPE evalute_node(DTYPE value,ACTIVATION_TYPE type){
	DTYPE temp;
	switch(type){
		case SIGMOID:
			return 1/(1+exp(-value));
		case TANH:
			temp=exp(2*value);
			return (temp-1)/(temp+1);
		default :
			break;
	}
	return 0;
}

int compute_network(NeuralNetwork* network,std::vector<DTYPE>* input){
	if(input->size()!=network->input_nodes->size()){
		printf("Input dimension not matching %ld!=%ld\n",input->size(),network->input_nodes->size());
		return -1;
	}
	DWORD i,j;
	for(i=0;i<network->hidden_nodes->size();i++){
		network->hidden_nodes[0][i].result=0;
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].result=0;
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->input_nodes->size();j++){
			network->hidden_nodes[0][i].result+=input[0][j]*network->input_nodes[0][j].weights[0][i];
		}
		for(j=0;j<network->hidden_nodes->size();j++){
			network->hidden_nodes[0][i].result+=network->hidden_nodes[0][j].context*network->hidden_nodes[0][j].hidden_weights[0][i];
		}
		network->hidden_nodes[0][i].result+=network->hidden_nodes[0][i].bias;
		network->hidden_nodes[0][i].result=evalute_node(network->hidden_nodes[0][i].result,network->hidden_type);
		network->hidden_nodes[0][i].context=network->hidden_nodes[0][i].result;
	}
	for(i=0;i<network->output_nodes->size();i++){
		for(j=0;j<network->hidden_nodes->size();j++){
			network->output_nodes[0][i].result+=network->hidden_nodes[0][j].result*network->hidden_nodes[0][j].weights[0][i];
		}
		network->output_nodes[0][i].result+=network->output_nodes[0][i].bias;
		network->output_nodes[0][i].result=evalute_node(network->output_nodes[0][i].result,network->output_type);
	}
	return 0;
}

void clean_network_states(NeuralNetwork* network){
	DWORD i;
	for(i=0;i<network->hidden_nodes->size();i++){
		network->hidden_nodes[0][i].context=0;
	}
}

DTYPE evaluate_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs){
	DWORD i,j;
	clean_network_states(network);
	DTYPE mse=0,temp;
	for(i=0;i<inputs->size();i++){
		//printf("i=%lld, compute start\n",i);
		//printf("size=%ld\n",outputs[0][i]->size());
		compute_network(network,inputs[0][i]);
		//printf("i=%lld, compute end\n",i);
		for(j=0;j<outputs[0][i]->size();j++){
			temp=outputs[0][i][0][j]-network->output_nodes[0][j].result;
			mse+=temp*temp;
		}
	}
	return mse;
}

void update_particle(Particle *particle,NeuralNetworkVector *global_best,DTYPE c1,DTYPE c2,DTYPE alpha,DTYPE phi1,DTYPE phi2){
	DWORD i,j;
	DTYPE l_best,g_best,loc;
	for(i=0;i<particle->velocity->input_nodes->size();i++){
		for(j=0;j<particle->velocity->input_nodes[0][i]->size();j++){
			l_best=particle->local_best->input_nodes[0][i][0][j];
			g_best=global_best->input_nodes[0][i][0][j];
			loc=particle->current->input_nodes[0][i].weights[0][j];
			particle->velocity->input_nodes[0][i][0][j]=loc*alpha+phi1*c1*(l_best-loc)+phi2*c2*(g_best-loc);
			particle->current->input_nodes[0][i].weights[0][j]+=particle->velocity->input_nodes[0][i][0][j];
		}
	}
	for(i=0;i<particle->velocity->hidden_nodes->size();i++){
		for(j=0;j<particle->velocity->hidden_nodes[0][i]->size();j++){
			l_best=particle->local_best->hidden_nodes[0][i][0][j];
			g_best=global_best->hidden_nodes[0][i][0][j];
			loc=particle->current->hidden_nodes[0][i].weights[0][j];
			particle->velocity->hidden_nodes[0][i][0][j]=loc*alpha+phi1*c1*(l_best-loc)+phi2*c2*(g_best-loc);
			particle->current->hidden_nodes[0][i].weights[0][j]+=particle->velocity->hidden_nodes[0][i][0][j];
		}
		for(j=0;j<particle->velocity->hidden_nodes_self[0][i]->size();j++){
			l_best=particle->local_best->hidden_nodes_self[0][i][0][j];
			g_best=global_best->hidden_nodes_self[0][i][0][j];
			loc=particle->current->hidden_nodes[0][i].hidden_weights[0][j];
			particle->velocity->hidden_nodes_self[0][i][0][j]=loc*alpha+phi1*c1*(l_best-loc)+phi2*c2*(g_best-loc);
			particle->current->hidden_nodes[0][i].hidden_weights[0][j]+=particle->velocity->hidden_nodes_self[0][i][0][j];
			/*
			if(particle->current->hidden_nodes[0][i].hidden_weights[0][j]>1){
				particle->current->hidden_nodes[0][i].hidden_weights[0][j]=1;
			}else{
				if(particle->current->hidden_nodes[0][i].hidden_weights[0][j]<-1){
					particle->current->hidden_nodes[0][i].hidden_weights[0][j]=-1;
				}
			}
			*/
		}


		l_best=particle->local_best->hidden_bias[0][i];
		g_best=global_best->hidden_bias[0][i];
		loc=particle->current->hidden_nodes[0][i].bias;
		particle->velocity->hidden_bias[0][i]=loc*alpha+phi1*c1*(l_best-loc)+phi2*c2*(g_best-loc);
		particle->current->hidden_nodes[0][i].bias+=particle->velocity->hidden_bias[0][i];
	}
	for(i=0;i<particle->velocity->output_bias->size();i++){
		l_best=particle->local_best->output_bias[0][i];
		g_best=global_best->output_bias[0][i];
		loc=particle->current->output_nodes[0][i].bias;
		particle->velocity->output_bias[0][i]=loc*alpha+phi1*c1*(l_best-loc)+phi2*c2*(g_best-loc);
		particle->current->output_nodes[0][i].bias+=particle->velocity->output_bias[0][i];
	}
}

NeuralNetwork *particle_swarm_optimization(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha){
	//local variables
	DWORD i,j;
	std::vector<Particle> *particles=new std::vector<Particle>;
	std::vector<DTYPE> *phi1=new std::vector<DTYPE>(particle_num);
	std::vector<DTYPE> *phi2=new std::vector<DTYPE>(particle_num);
	Particle *temp_best=NULL;
	DTYPE temp,best;
	NeuralNetworkVector *global_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
	//initialize all particles
	particles->resize(particle_num);
	for(i=0;i<particle_num;i++){
		particles[0][i].current=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		particles[0][i].local_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		particles[0][i].velocity=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		randomize_weight(particles[0][i].current);
		copy_neural_network_to_vector(particles[0][i].current,particles[0][i].local_best);
	}
	//Compute g_best before execution
	best=MAX_DTYPE;
	for(i=0;i<particle_num;i++){
		//printf("i=%lld\n",i);
		temp=evaluate_network(particles[0][i].current,inputs,outputs);
		//printf("initial %lld=%lf\n",i,temp);
		particles[0][i].mse=temp;
		if(temp<best){
			best=temp;
			temp_best=&particles[0][i];
		}
	}
	copy_neural_network_to_vector(temp_best->current,global_best);
	//PSO algorithm
	for(i=0;i<epochs;i++){
		printf("epoch i, g_best=%lf\n",best);
		for(j=0;j<particle_num;j++){
			phi1[0][j]=genrand_real2();
			phi2[0][j]=genrand_real2();
		}
		#pragma omp parallel private(j,temp) num_threads(PROCS)
		{
			#pragma omp for schedule(static)
			for(j=0;j<particle_num;j++){
				printf("    processing particle %lld\n",j);
				update_particle(&particles[0][j],global_best,C1,C2,alpha,phi1[0][j],phi2[0][j]);
				temp=evaluate_network(particles[0][j].current,inputs,outputs);
				particles[0][j].current_mse=temp;
				if(temp<=particles[0][j].mse){
					particles[0][j].mse=temp;
					copy_neural_network_to_vector(particles[0][j].current,particles[0][j].local_best);
				}
			}
		}
		temp_best=NULL;
		for(j=0;j<particle_num;j++){
			if(particles[0][j].mse<best){
				best=particles[0][j].mse;
				temp_best=&particles[0][j];
			}
		}
		if(temp_best!=NULL){
			copy_vector_to_vector(temp_best->local_best,global_best);
		}
	}
	//clean up
	delete phi1;
	delete phi2;
	for(i=0;i<particle_num;i++){
		destroy_neural_network_vector(particles[0][i].local_best);
		destroy_neural_network_vector(particles[0][i].velocity);
		destroy_neural_network(particles[0][i].current);
	}
	printf("best=%lf\n",best);
	NeuralNetwork *result=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	copy_vector_to_neural_network(global_best,result);
	return result;
}
/*
DWORD gradient_score(NeuralNetwork* network,DTYPE bound,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs, DTYPE* mse){
	DWORD i,j,result=0;
	DTYPE temp;
	//Compute weight bound for hidden to output weights
	std::vector<DTYPE> *hidden_node_scores=new std::vector<DTYPE>;
	std::vector<DTYPE> *output_node_scores=new std::vector<DTYPE>;
	hidden_node_scores->resize(network->hidden_nodes->size(),0);
	output_node_scores->resize(network->output_nodes->size(),0);
	clean_network_states(network);
	mse[0]=0;
	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
		for(j=0;j<network->hidden_nodes->size();j++){
			hidden_node_scores[0][j]+=network->hidden_nodes[0][j].context;
		}
		for(j=0;j<network->output_nodes->size();j++){
			temp=outputs[0][i][0][j]-network->output_nodes[0][j].result;
			mse[0]+=temp*temp;
			output_node_scores[0][j]+=network->output_nodes[0][j].result;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		hidden_node_scores[0][j]/=inputs->size();
		if(hidden_node_scores[0][j]<bound&&hidden_node_scores[0][j]>(1-bound)){
			result++;
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		output_node_scores[0][j]/=inputs->size();
		if(output_node_scores[0][j]<bound&&output_node_scores[0][j]>(1-bound)){
			result++;
		}
	}
	return result;
}
*/

DWORD gradient_score(NeuralNetwork* network,DTYPE bound,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DTYPE* mse){
	DTYPE D_in=0;
	DWORD i,j,result=0;
	DTYPE temp,min,max,w_max,v_max;
	//Compute weight bound for hidden to output weights
	std::vector<DTYPE> *vector_max=new std::vector<DTYPE>;
	std::vector<DTYPE> *vector_min=new std::vector<DTYPE>;
	vector_max->resize(network->hidden_nodes->size(),-1);
	vector_min->resize(network->hidden_nodes->size(),1);
	clean_network_states(network);
	mse[0]=0;
	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
		for(j=0;j<network->hidden_nodes->size();j++){
			if(network->hidden_nodes[0][j].context>vector_max[0][j]){
				vector_max[0][j]=network->hidden_nodes[0][j].context;
			}
			if(network->hidden_nodes[0][j].context<vector_min[0][j]){
				vector_min[0][j]=network->hidden_nodes[0][j].context;
			}
		}
		for(j=0;j<network->output_nodes->size();j++){
			temp=outputs[0][i][0][j]-network->output_nodes[0][j].result;
			mse[0]+=temp*temp;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		temp=vector_max[0][j]-vector_min[0][j];
		D_in+=temp*temp;
	}
	v_max=sqrt(3)*8.72/(D_in*sqrt(network->hidden_nodes->size()));

	for(i=0;i<network->output_nodes->size();i++){
		for(j=0;j<network->hidden_nodes->size();j++){
			if(fabs(network->hidden_nodes[0][j].weights[0][i])<v_max){
				result++;
			}
		}

	}
	//Compute weight bound for input to hidden weights
	for(j=0;j<inputs[0][0]->size();j++){
		min=inputs[0][0][0][j];
		max=inputs[0][0][0][j];
		for(i=1;i<inputs->size();i++){
			if(inputs[0][i][0][j]<min){
				min=inputs[0][i][0][j];
			}
			if(inputs[0][i][0][j]>max){
				max=inputs[0][i][0][j];
			}
		}
		temp=max-min;
		D_in+=temp*temp;
	}

	w_max=sqrt(3)*8.72/(D_in*sqrt(inputs[0][0]->size()+network->hidden_nodes->size()));

	for(i=0;i<network->hidden_nodes->size();i++){

		for(j=0;j<network->input_nodes->size();j++){
			if(fabs(network->input_nodes[0][j].weights[0][i])<w_max){
				result++;
			}
		}
		
		for(j=0;j<network->hidden_nodes->size();j++){
			if(fabs(network->hidden_nodes[0][j].hidden_weights[0][i])<w_max){
				result++;
			}
		}
		
	}
	delete vector_max;
	delete vector_min;
	return result;
}


void randomize_weight2(NeuralNetwork* network, std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs){
	DTYPE D_in=0;
	DWORD i,j;
	DTYPE temp,min,max,w_max,v_max;
	//Compute weight bound for hidden to output weights
	std::vector<DTYPE> *vector_max=new std::vector<DTYPE>;
	std::vector<DTYPE> *vector_min=new std::vector<DTYPE>;
	vector_max->resize(network->hidden_nodes->size(),-1);
	vector_min->resize(network->hidden_nodes->size(),1);
	clean_network_states(network);

	//Compute weight bound for input to hidden weights
	for(j=0;j<inputs[0][0]->size();j++){
		min=inputs[0][0][0][j];
		max=inputs[0][0][0][j];
		for(i=1;i<inputs->size();i++){
			if(inputs[0][i][0][j]<min){
				min=inputs[0][i][0][j];
			}
			if(inputs[0][i][0][j]>max){
				max=inputs[0][i][0][j];
			}
		}
		temp=max-min;
		D_in+=temp*temp;
	}

	w_max=sqrt(3)*8.72/(D_in*sqrt(inputs[0][0]->size()+network->hidden_nodes->size()));

	for(i=0;i<network->input_nodes->size();i++){
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			network->input_nodes[0][i].weights[0][j]=(2*w_max*genrand_real2()-1);
		}
	}

	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
		for(j=0;j<network->hidden_nodes->size();j++){
			if(network->hidden_nodes[0][j].context>vector_max[0][j]){
				vector_max[0][j]=network->hidden_nodes[0][j].context;
			}
			if(network->hidden_nodes[0][j].context<vector_min[0][j]){
				vector_min[0][j]=network->hidden_nodes[0][j].context;
			}
		}
		for(j=0;j<network->output_nodes->size();j++){
			temp=outputs[0][i][0][j]-network->output_nodes[0][j].result;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		temp=vector_max[0][j]-vector_min[0][j];
		D_in+=temp*temp;
	}
	v_max=sqrt(3)*8.72/(D_in*sqrt(network->hidden_nodes->size()));

	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			network->hidden_nodes[0][i].weights[0][j]=(2*v_max*genrand_real2()-1);
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			network->hidden_nodes[0][i].hidden_weights[0][j]=(2*v_max*genrand_real2()-1);
		}
		network->hidden_nodes[0][i].bias=(2*genrand_real2()-1);
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].bias=(2*genrand_real2()-1);
	}

	delete vector_max;
	delete vector_min;
}


Particle* copy_current_particle(Particle* particle,WORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type){
	Particle *p=Calloc(1,Particle);
	p->current=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	copy_neural_network_to_neural_network(particle->current,p->current);
	p->current_mse=particle->current_mse;
	p->gradient_score=particle->gradient_score;
	return p;
}

std::vector<NeuralNetwork*>* particle_swarm_optimization_by_gradient(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha,DWORD gradient_list_size,DWORD mse_list_size,DTYPE bound){
	//local variables
	DWORD i,j,temp_score;
	std::vector<Particle*> *particles=new std::vector<Particle*>;
	std::vector<DTYPE> *phi1=new std::vector<DTYPE>(particle_num);
	std::vector<DTYPE> *phi2=new std::vector<DTYPE>(particle_num);
	Particle *temp_best=NULL;
	DTYPE temp,best;
	NeuralNetworkVector *global_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
	//initialize all particles
	particles->resize(particle_num);
	for(i=0;i<particle_num;i++){
		//printf("i=%lld\n",i);
		particles[0][i]=Calloc(1,Particle);
		particles[0][i]->current=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		particles[0][i]->local_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		particles[0][i]->velocity=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		randomize_weight(particles[0][i]->current);
		copy_neural_network_to_vector(particles[0][i]->current,particles[0][i]->local_best);
	}
	//Compute g_best before execution
	best=MAX_DTYPE;
	#pragma omp parallel private(i) num_threads(PROCS)
	{
		#pragma omp for schedule(guided)
		for(i=0;i<particle_num;i++){
			//printf("i=%lld\n",i);
			particles[0][i]->mse=evaluate_network(particles[0][i]->current,inputs,outputs);
			//printf("initial %lld=%lf\n",i,particles[0][i]->mse);
		}
	}
	for(i=0;i<particle_num;i++){
		if(particles[0][i]->mse<best){
			best=particles[0][i]->mse;
			temp_best=particles[0][i];
		}
	}


	copy_neural_network_to_vector(temp_best->current,global_best);
	//PSO algorithm
	PriorityQueue* gradient_list=create_priority_queue();			//If gradient_list is not full.
	std::vector<Particle*> *mse_list=new std::vector<Particle*>;
	Particle* p;

	for(i=0;i<epochs;i++){
		//printf("epoch %lld, g_best=%lf\n",i,best);
		for(j=0;j<particle_num;j++){
			phi1[0][j]=genrand_real2();
			phi2[0][j]=genrand_real2();
		}
		#pragma omp parallel private(j,temp) num_threads(PROCS)
		{
			#pragma omp for schedule(static)
			for(j=0;j<particle_num;j++){
				update_particle(particles[0][j],global_best,C1,C2,alpha,phi1[0][j],phi2[0][j]);
				//gradient_descent_network(particles[0][j]->current,inputs,outputs,5,.1);
				particles[0][j]->gradient_score=gradient_score(particles[0][j]->current,bound,inputs,outputs,&temp);
				//temp=evaluate_network(particles[0][j]->current,inputs,outputs);
				//printf("    processing particle %lld,mse=%lf\n",j,temp);
				particles[0][j]->current_mse=temp;
				if(temp<=particles[0][j]->mse){
					particles[0][j]->mse=temp;
					copy_neural_network_to_vector(particles[0][j]->current,particles[0][j]->local_best);
				}

				/*
				if(particles[0][j]->mse<best){
					best=particles[0][j]->mse;
					copy_vector_to_vector(particles[0][j]->local_best,global_best);
				}
				*/
			}
		}
		temp_best=NULL;
		for(j=0;j<particle_num;j++){
			if(particles[0][j]->mse<best){
				best=particles[0][j]->mse;
				temp_best=particles[0][j];
			}
		}
		if(temp_best!=NULL){
			copy_vector_to_vector(temp_best->local_best,global_best);
		}
		//Sort particles by mse
		std::sort(particles->begin(), particles->end(), mse_compare);
		//copy best particles
		mse_list->insert(mse_list->end(), particles->begin(), particles->begin()+mse_list_size);
		//Sort best particles by gradient
		std::sort(mse_list->begin(), mse_list->end(), gradient_compare);
		for(j=0;j<mse_list_size;j++){
			if(PQ_LENGTH(gradient_list)<gradient_list_size){
				//If gradient_list is not full, insert best particles in order of gradient score
				p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
				insert_to_priority_queue(gradient_list,p);
			}else{
				//If gradient list is full, only insert if the worst particle in the list has gradient score worst than best gradient score 
				p=top_of_priority_queue(gradient_list);
				temp_score=p->gradient_score;
				if(p->gradient_score<mse_list[0][j]->gradient_score||(temp_score==mse_list[0][j]->gradient_score&&j==0)){
					//printf("replacement triggered at i=%lld,j=%lld,new score=%lld,old score=%lld\n",i,j,mse_list[0][j]->gradient_score,p->gradient_score);
					p=pop_from_priority_queue(gradient_list);
					destroy_neural_network(p->current);
					Free(p);
					p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
					insert_to_priority_queue(gradient_list,p);
					if(temp_score==mse_list[0][j]->gradient_score){
						break;
					}
				}else{
					//Mutation factor, force ejection of worst mse particle and replace it with current best gradient.
					if(j==0&&p->current_mse>mse_list[0][j]->current_mse){
						//printf("mutation triggered at i=%lld,j=%lld,new score=%lld,old score=%lld\n",i,j,mse_list[0][j]->gradient_score,p->gradient_score);
						p=pop_from_priority_queue(gradient_list);
						destroy_neural_network(p->current);
						Free(p);
						p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
						insert_to_priority_queue(gradient_list,p);
					}
					break;
				
				}
			}
		}
		mse_list->clear(); 
	}
	//clean up
	delete phi1;
	delete phi2;
	for(i=0;i<particle_num;i++){
		destroy_neural_network_vector(particles[0][i]->local_best);
		destroy_neural_network_vector(particles[0][i]->velocity);
		destroy_neural_network(particles[0][i]->current);
		Free(particles[0][i]);
	}
	delete particles;
	//printf("best=%lf\n",best);
	// Keep global best and all particles in gradient list.
	std::vector<NeuralNetwork*> *result=new std::vector<NeuralNetwork*>;
	NeuralNetwork *network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	copy_vector_to_neural_network(global_best,network);
	result->push_back(network);
	while(!is_priority_queue_empty(gradient_list)){
		p=pop_from_priority_queue(gradient_list);
		result->push_back(p->current);
		Free(p);
	}
	destroy_neural_network_vector(global_best);
	destroy_priority_queue(gradient_list);
	delete mse_list;
	return result;
}

DTYPE evalute_gradient(DTYPE value,ACTIVATION_TYPE type){
	DTYPE temp;
	switch(type){
		case SIGMOID:
			temp=exp(value);
			return temp/((1+temp)*(1+temp));
		case TANH:
			temp=evalute_node(value,TANH);
			return 1-temp*temp;
		default :
			break;
	}
	return 0;
}

int compute_network_by_context(NeuralNetwork* network,std::vector<DTYPE>* input,std::vector<std::vector<NodeCache>*> *hidden_caches,std::vector<std::vector<NodeCache>*> *output_caches,DWORD current_index){
	//printf("check context\n");
	if(input->size()!=network->input_nodes->size()){
		printf("Input dimension not matching %ld!=%ld\n",input->size(),network->input_nodes->size());
		return -1;
	}
	DWORD i,j;
	for(i=0;i<network->hidden_nodes->size();i++){
		network->hidden_nodes[0][i].result=0;
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].result=0;
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->input_nodes->size();j++){
			network->hidden_nodes[0][i].result+=input[0][j]*network->input_nodes[0][j].weights[0][i];
		}
		for(j=0;j<network->hidden_nodes->size();j++){
			network->hidden_nodes[0][i].result+=network->hidden_nodes[0][j].context*network->hidden_nodes[0][j].hidden_weights[0][i];
		}

		network->hidden_nodes[0][i].result+=network->hidden_nodes[0][i].bias;
		hidden_caches[0][current_index][0][i].input_value=network->hidden_nodes[0][i].result;
		network->hidden_nodes[0][i].result=evalute_node(network->hidden_nodes[0][i].result,network->hidden_type);
		network->hidden_nodes[0][i].context=network->hidden_nodes[0][i].result;
		hidden_caches[0][current_index][0][i].output_value=network->hidden_nodes[0][i].result;
	}
	for(i=0;i<network->output_nodes->size();i++){
		for(j=0;j<network->hidden_nodes->size();j++){
			network->output_nodes[0][i].result+=network->hidden_nodes[0][j].result*network->hidden_nodes[0][j].weights[0][i];
		}
		network->output_nodes[0][i].result+=network->output_nodes[0][i].bias;
		output_caches[0][current_index][0][i].input_value=network->output_nodes[0][i].result;
		network->output_nodes[0][i].result=evalute_node(network->output_nodes[0][i].result,network->output_type);
		output_caches[0][current_index][0][i].output_value=network->output_nodes[0][i].result;
	}
	return 0;
}

void compute_gradient(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs){
	std::vector<std::vector<NodeCache>*> *hidden_caches=new std::vector<std::vector<NodeCache>*>;
	std::vector<std::vector<NodeCache>*> *output_caches=new std::vector<std::vector<NodeCache>*>;
	std::vector<GradientError> *errors=new std::vector<GradientError>;
	DWORD i,j,k;
	DTYPE gradient,error,temp,sum_error;
	//DTYPE denominator,scale;
	errors->resize(inputs->size());
	for(i=0;i<inputs->size();i++){
		errors[0][i].hidden_nodes=new std::vector<DTYPE>;
		errors[0][i].output_nodes=new std::vector<DTYPE>;
		errors[0][i].hidden_nodes->resize(network->hidden_nodes->size());
		errors[0][i].output_nodes->resize(network->output_nodes->size());
	}
	hidden_caches->resize(inputs->size());
	output_caches->resize(inputs->size());
	for(i=0;i<hidden_caches->size();i++){
		hidden_caches[0][i]=new std::vector<NodeCache>;
		hidden_caches[0][i]->resize(network->hidden_nodes->size());
	}
	for(i=0;i<output_caches->size();i++){
		output_caches[0][i]=new std::vector<NodeCache>;
		output_caches[0][i]->resize(network->output_nodes->size());
	}

	//feed forward through time series
	clean_network_states(network);
	for(i=0;i<inputs->size();i++){
		//printf("check i=%lld\n",i);
		compute_network_by_context(network,inputs[0][i],hidden_caches,output_caches,i);
	}

	std::vector<DTYPE> *recurrent_error=new std::vector<DTYPE>;
	recurrent_error->resize(network->hidden_nodes->size(),0);
	//back propagate error
	for(i=inputs->size()-1;i!=MAX_DWORD;i--){
		//printf("time stamp=%lld,output->size()=%lu\n",i,network->output_nodes->size());
		//output error
		for(j=0;j<network->output_nodes->size();j++){
			gradient=evalute_gradient(output_caches[0][i][0][j].input_value,network->output_type);
			error=output_caches[0][i][0][j].output_value-outputs[0][i][0][j];
			errors[0][i].output_nodes[0][j]=error*gradient;
			//printf("time stamp %lld output error for node %lld is %lf*%lf=%lf,input=%lf,predict=%lf,target=%lf\n",i,j,gradient,error,error*gradient,output_caches[0][i][0][j].input_value,output_caches[0][i][0][j].output_value,outputs[0][i][0][j]);
		}
		//hidden error
		for(j=0;j<network->hidden_nodes->size();j++){
			gradient=evalute_gradient(hidden_caches[0][i][0][j].input_value,network->output_type);
			error=0;
			for(k=0;k<network->output_nodes->size();k++){
				error+=network->hidden_nodes[0][j].weights[0][k]*errors[0][i].output_nodes[0][k];
				//printf("time stamp %lld error[%lld]+=%lf*%lf=%lf, error[%lld]=%lf\n",i,j,network->hidden_nodes[0][j].weights[0][k],errors[0][i].output_nodes[0][k],network->hidden_nodes[0][j].weights[0][k]*errors[0][i].output_nodes[0][k],j,error);
			}
			temp=error;

			if(i<inputs->size()-1){
				sum_error=0;
				for(k=0;k<network->hidden_nodes->size();k++){
					sum_error+=recurrent_error[0][k]*network->hidden_nodes[0][j].hidden_weights[0][k];
				}
				temp+=sum_error/network->hidden_nodes->size();
				error+=sum_error;//(inputs->size()-1-i);
				//printf("error+=%lf,error=%lf\n",sum_error,error);
			}
			recurrent_error[0][j]=temp*gradient;

			errors[0][i].hidden_nodes[0][j]=error*gradient;
		}
	}
	delete recurrent_error;


	//printf("clear gradients\n");
	//compute gradient for parameters.
	for(i=0;i<network->input_nodes->size();i++){
		//printf("i=%lld,size=%ld\n",i,network->input_nodes[0][i].weights_gradient->size());
		std::fill(network->input_nodes[0][i].weights_gradient->begin(),network->input_nodes[0][i].weights_gradient->end(),0);
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		std::fill(network->hidden_nodes[0][i].weights_gradient->begin(),network->hidden_nodes[0][i].weights_gradient->end(),0);
		std::fill(network->hidden_nodes[0][i].hidden_weights_gradient->begin(),network->hidden_nodes[0][i].hidden_weights_gradient->end(),0);
		network->hidden_nodes[0][i].bias_gradient=0;
	}
	for(i=0;i<network->output_nodes->size();i++){
		network->output_nodes[0][i].bias_gradient=0;
	}
	//printf("compute gradient\n");

	for(i=0;i<errors->size();i++){
		//printf("time stamp=%lld\n",i);
		//gradients between input and hidden layer
		for(j=0;j<network->input_nodes->size();j++){
			for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
				network->input_nodes[0][j].weights_gradient[0][k]+=inputs[0][i][0][j]*errors[0][i].hidden_nodes[0][k];
	
					//printf("input to hidden gradient w(%lld,%lld)=%lf=%lf*%lf\n",j,k,inputs[0][i][0][j]*errors[0][i].hidden_nodes[0][k],inputs[0][i][0][j],errors[0][i].hidden_nodes[0][k]);
			}
		}
		for(j=0;j<network->hidden_nodes->size();j++){
			network->hidden_nodes[0][j].bias_gradient+=errors[0][i].hidden_nodes[0][j];
			if(i>0){
				for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
					network->hidden_nodes[0][j].hidden_weights_gradient[0][k]+=hidden_caches[0][i-1][0][j].output_value*errors[0][i].hidden_nodes[0][k];
				}
			}
			//printf("bias_gradient=%lf\n",errors[0][i].hidden_nodes[0][j]);
		}
		//printf("hidden to output\n");
		//gradients between hidden and output layer
		for(j=0;j<network->hidden_nodes->size();j++){
			for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
				network->hidden_nodes[0][j].weights_gradient[0][k]+=hidden_caches[0][i][0][j].output_value*errors[0][i].output_nodes[0][k];
				//printf("hidden to output gradient w(%lld,%lld)=%lf=%lf*%lf\n",j,k,hidden_caches[0][i][0][j].output_value*errors[0][i].output_nodes[0][k],hidden_caches[0][i][0][j].output_value,errors[0][i].output_nodes[0][k]);
			}
		}
		for(j=0;j<network->output_nodes->size();j++){
			network->output_nodes[0][j].bias_gradient+=errors[0][i].output_nodes[0][j];
			//printf("bias_gradient=%lf\n",errors[0][i].output_nodes[0][j]);
		}
	}
	//Average gradients
	//gradients between input and hidden layer

	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]/=errors->size();
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient/=errors->size();
		if(errors->size()>1){
			for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
				network->hidden_nodes[0][j].hidden_weights_gradient[0][k]/=(errors->size()-1);
			}
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]/=errors->size();
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient/=errors->size();
	}

	/*
	denominator=0,scale=1600;
	for(i=0;i<errors->size();i++){
		denominator+=scale/(i+scale);
	}
	//printf("denominator=%lf\n",denominator);
	for(i=0;i<errors->size();i++){
		//printf("time stamp=%lld\n",i);
		//gradients between input and hidden layer
		for(j=0;j<network->input_nodes->size();j++){
			for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
				network->input_nodes[0][j].weights_gradient[0][k]+=inputs[0][i][0][j]*errors[0][i].hidden_nodes[0][k]*scale/(i+scale);
	
					//printf("input to hidden gradient w(%lld,%lld)=%lf=%lf*%lf\n",j,k,inputs[0][i][0][j]*errors[0][i].hidden_nodes[0][k],inputs[0][i][0][j],errors[0][i].hidden_nodes[0][k]);
			}
		}
		for(j=0;j<network->hidden_nodes->size();j++){
			network->hidden_nodes[0][j].bias_gradient+=errors[0][i].hidden_nodes[0][j]*scale/(i+scale);
			if(i>0){
				for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
					network->hidden_nodes[0][j].hidden_weights_gradient[0][k]+=hidden_caches[0][i-1][0][j].output_value*errors[0][i].hidden_nodes[0][k]*scale/(i+scale);
				}
			}
			//printf("bias_gradient=%lf\n",errors[0][i].hidden_nodes[0][j]);
		}
		//printf("hidden to output\n");
		//gradients between hidden and output layer
		for(j=0;j<network->hidden_nodes->size();j++){
			for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
				network->hidden_nodes[0][j].weights_gradient[0][k]+=hidden_caches[0][i][0][j].output_value*errors[0][i].output_nodes[0][k]*scale/(i+scale);
				//printf("hidden to output gradient w(%lld,%lld)=%lf=%lf*%lf\n",j,k,hidden_caches[0][i][0][j].output_value*errors[0][i].output_nodes[0][k],hidden_caches[0][i][0][j].output_value,errors[0][i].output_nodes[0][k]);
			}
		}
		for(j=0;j<network->output_nodes->size();j++){
			network->output_nodes[0][j].bias_gradient+=errors[0][i].output_nodes[0][j]*scale/(i+scale);
			//printf("bias_gradient=%lf\n",errors[0][i].output_nodes[0][j]);
		}
	}
	//gradients between input and hidden layer
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]/=denominator;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient/=denominator;
		if(errors->size()>1){
			for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
				network->hidden_nodes[0][j].hidden_weights_gradient[0][k]/=(denominator-1);
			}
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]/=denominator;
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient/=denominator;
	}
	*/
	for(i=0;i<inputs->size();i++){
		delete errors[0][i].hidden_nodes;
		delete errors[0][i].output_nodes;
	}
	for(i=0;i<hidden_caches->size();i++){
		delete hidden_caches[0][i];
	}
	for(i=0;i<output_caches->size();i++){
		delete output_caches[0][i];
	}

	delete errors;
	delete hidden_caches;
	delete output_caches;
}
/*
void update_network_by_gradient(NeuralNetwork *network,DTYPE learning_rate){
	DWORD j,k;
	//Update weights by gradients.
	//gradients between input and hidden layer
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights[0][k]-=learning_rate*network->input_nodes[0][j].weights_gradient[0][k];
			//printf("gradient=%lf\n",network->input_nodes[0][j].weights[0][k]);
		}
	}
	//gradient within hidden node
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias-=learning_rate*network->hidden_nodes[0][j].bias_gradient;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			network->hidden_nodes[0][j].hidden_weights[0][k]-=learning_rate*network->hidden_nodes[0][j].hidden_weights_gradient[0][k];
			//printf("gradient=%lf\n",network->input_nodes[0][j].weights[0][k]);
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights[0][k]-=learning_rate*network->hidden_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias-=learning_rate*network->output_nodes[0][j].bias_gradient;
	}

}
*/
void update_network_by_gradient(NeuralNetwork *network,DTYPE learning_rate,DWORD n){
	DWORD j,k;
	//Update weights by gradients.
	//gradients between input and hidden layer
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights[0][k]-=learning_rate*(network->input_nodes[0][j].weights_gradient[0][k]+LAMBDA*network->input_nodes[0][j].weights[0][k]/n);
			//printf("gradient=%lf\n",network->input_nodes[0][j].weights[0][k]);
		}
	}
	//gradient within hidden node
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias-=learning_rate*network->hidden_nodes[0][j].bias_gradient;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			network->hidden_nodes[0][j].hidden_weights[0][k]-=learning_rate*(network->hidden_nodes[0][j].hidden_weights_gradient[0][k]+LAMBDA*network->hidden_nodes[0][j].hidden_weights[0][k]/n);
			/*
			if(network->hidden_nodes[0][j].hidden_weights[0][k]>1){
				network->hidden_nodes[0][j].hidden_weights[0][k]=1;
			}else{
				if(network->hidden_nodes[0][j].hidden_weights[0][k]<-1){
					network->hidden_nodes[0][j].hidden_weights[0][k]=-1;
				}
			}
			*/
			//printf("gradient=%lf\n",network->input_nodes[0][j].weights[0][k]);
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights[0][k]-=learning_rate*(network->hidden_nodes[0][j].weights_gradient[0][k]+LAMBDA*network->hidden_nodes[0][j].weights[0][k]/n);
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias-=learning_rate*network->output_nodes[0][j].bias_gradient;
	}

}


NeuralNetwork *gradient_descent(std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE learning_rate,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type){
	NeuralNetwork *result=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	randomize_weight(result);
	printf("MSE=%lf\n",evaluate_network(result,inputs,outputs));
	DWORD i;
	for(i=0;i<epochs;i++){
		//printf("epochs %lld\n",i);
		compute_gradient(result,inputs,outputs);
		//printf("check\n");
		update_network_by_gradient(result,learning_rate,inputs->size());
		//printf("check\n");
		clean_network_states(result);
		//printf("check\n");
	}
	return result;
}

void gradient_descent_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE learning_rate){
	clean_network_states(network);
	DWORD i;
	for(i=0;i<epochs;i++){
		compute_gradient(network,inputs,outputs);
		update_network_by_gradient(network,learning_rate,inputs->size());
		clean_network_states(network);
	}
}

void add_gradients_by_scalar(NeuralNetwork* network,NeuralNetwork* result,DTYPE alpha){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			result->input_nodes[0][j].weights_gradient[0][k]=network->input_nodes[0][j].weights_gradient[0][k]+alpha;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		result->hidden_nodes[0][j].bias_gradient=network->hidden_nodes[0][j].bias_gradient+alpha;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			result->hidden_nodes[0][j].hidden_weights_gradient[0][k]=network->hidden_nodes[0][j].hidden_weights_gradient[0][k]+alpha;
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			result->hidden_nodes[0][j].weights_gradient[0][k]=network->hidden_nodes[0][j].weights_gradient[0][k]+alpha;
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		result->output_nodes[0][j].bias_gradient=network->output_nodes[0][j].bias_gradient+alpha;
	}
}

void multiple_gradients_by_scalar(NeuralNetwork* network,NeuralNetwork* result,DTYPE alpha){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			result->input_nodes[0][j].weights_gradient[0][k]=network->input_nodes[0][j].weights_gradient[0][k]*alpha;
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		result->hidden_nodes[0][j].bias_gradient=network->hidden_nodes[0][j].bias_gradient*alpha;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			result->hidden_nodes[0][j].hidden_weights_gradient[0][k]=network->hidden_nodes[0][j].hidden_weights_gradient[0][k]*alpha;
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			result->hidden_nodes[0][j].weights_gradient[0][k]=network->hidden_nodes[0][j].weights_gradient[0][k]*alpha;
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		result->output_nodes[0][j].bias_gradient=network->output_nodes[0][j].bias_gradient*alpha;
	}
}

void square_root_gradients(NeuralNetwork* network){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]=sqrt(network->input_nodes[0][j].weights_gradient[0][k]);
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient=sqrt(network->hidden_nodes[0][j].bias_gradient);
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			network->hidden_nodes[0][j].hidden_weights_gradient[0][k]=sqrt(network->hidden_nodes[0][j].hidden_weights_gradient[0][k]);
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]=sqrt(network->hidden_nodes[0][j].weights_gradient[0][k]);
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient=sqrt(network->output_nodes[0][j].bias_gradient);
	}
}

void add_gradients_by_gradients(NeuralNetwork* network,NeuralNetwork* network2,NeuralNetwork* result){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			result->input_nodes[0][j].weights_gradient[0][k]=network->input_nodes[0][j].weights_gradient[0][k]+network2->input_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		result->hidden_nodes[0][j].bias_gradient=network->hidden_nodes[0][j].bias_gradient+network2->hidden_nodes[0][j].bias_gradient;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			result->hidden_nodes[0][j].hidden_weights_gradient[0][k]=network->hidden_nodes[0][j].hidden_weights_gradient[0][k]+network2->hidden_nodes[0][j].hidden_weights_gradient[0][k];
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			result->hidden_nodes[0][j].weights_gradient[0][k]=network->hidden_nodes[0][j].weights_gradient[0][k]+network2->hidden_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		result->output_nodes[0][j].bias_gradient=network->output_nodes[0][j].bias_gradient+network2->output_nodes[0][j].bias_gradient;
	}
}

void divide_gradients_by_gradients(NeuralNetwork* network,NeuralNetwork* network2,NeuralNetwork* result){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			result->input_nodes[0][j].weights_gradient[0][k]=network->input_nodes[0][j].weights_gradient[0][k]/network2->input_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		result->hidden_nodes[0][j].bias_gradient=network->hidden_nodes[0][j].bias_gradient/network2->hidden_nodes[0][j].bias_gradient;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			result->hidden_nodes[0][j].hidden_weights_gradient[0][k]=network->hidden_nodes[0][j].hidden_weights_gradient[0][k]/network2->hidden_nodes[0][j].hidden_weights_gradient[0][k];
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			result->hidden_nodes[0][j].weights_gradient[0][k]=network->hidden_nodes[0][j].weights_gradient[0][k]/network2->hidden_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		result->output_nodes[0][j].bias_gradient=network->output_nodes[0][j].bias_gradient/network2->output_nodes[0][j].bias_gradient;
	}
}

void square_gradients(NeuralNetwork* network){
	DWORD j,k;
	for(j=0;j<network->input_nodes->size();j++){
		for(k=0;k<network->input_nodes[0][j].weights_gradient->size();k++){
			network->input_nodes[0][j].weights_gradient[0][k]*=network->input_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->hidden_nodes->size();j++){
		network->hidden_nodes[0][j].bias_gradient*=network->hidden_nodes[0][j].bias_gradient;
		for(k=0;k<network->hidden_nodes[0][j].hidden_weights_gradient->size();k++){
			network->hidden_nodes[0][j].hidden_weights_gradient[0][k]*=network->hidden_nodes[0][j].hidden_weights_gradient[0][k];
		}
	}
	//gradients between hidden and output layer
	for(j=0;j<network->hidden_nodes->size();j++){
		for(k=0;k<network->hidden_nodes[0][j].weights_gradient->size();k++){
			network->hidden_nodes[0][j].weights_gradient[0][k]*=network->hidden_nodes[0][j].weights_gradient[0][k];
		}
	}
	for(j=0;j<network->output_nodes->size();j++){
		network->output_nodes[0][j].bias_gradient*=network->output_nodes[0][j].bias_gradient;
	}
}



void adam_gradient_descent_network(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DWORD epochs,DTYPE alpha,DTYPE beta1,DTYPE beta2,DTYPE epsilon){
	clean_network_states(network);
	DWORD i;
	DTYPE t=0;
	NeuralNetwork *v=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
	NeuralNetwork *m=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
	NeuralNetwork *temp=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
	for(i=0;i<epochs;i++){
		t+=1;
		compute_gradient(network,inputs,outputs);
		//Compute vector m
		multiple_gradients_by_scalar(network,temp,1-beta1);
		multiple_gradients_by_scalar(m,m,beta1);
		add_gradients_by_gradients(m,temp,m);
		//compute vector v
		square_gradients(network);
		multiple_gradients_by_scalar(network,network,1-beta2);
		multiple_gradients_by_scalar(v,v,beta2);
		add_gradients_by_gradients(v,network,v);
		//biased-corrected moments
		multiple_gradients_by_scalar(v,v,1/(1-pow(beta2,t)));
		square_root_gradients(v);
		add_gradients_by_scalar(v,v,epsilon);
		multiple_gradients_by_scalar(m,m,1/(1-pow(beta1,t)));
		//compute adjusted gradient
		divide_gradients_by_gradients(m,v,network);
		//update parameters
		update_network_by_gradient(network,alpha,inputs->size());
		clean_network_states(network);
	}

	destroy_neural_network(v);
	destroy_neural_network(m);
}

void adam_gradient_descent_network_one_spoch(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs,DTYPE alpha,DTYPE beta1,DTYPE beta2,DTYPE epsilon,NeuralNetwork *v,NeuralNetwork *m,NeuralNetwork *temp,DWORD t){
	clean_network_states(network);
	compute_gradient(network,inputs,outputs);
	//Compute vector m
	multiple_gradients_by_scalar(network,temp,1-beta1);
	multiple_gradients_by_scalar(m,m,beta1);
	add_gradients_by_gradients(m,temp,m);
	//compute vector v
	square_gradients(network);
	multiple_gradients_by_scalar(network,network,1-beta2);
	multiple_gradients_by_scalar(v,v,beta2);
	add_gradients_by_gradients(v,network,v);
	//biased-corrected moments
	multiple_gradients_by_scalar(v,v,1/(1-pow(beta2,t)));
	square_root_gradients(v);
	add_gradients_by_scalar(v,v,epsilon);
	multiple_gradients_by_scalar(m,m,1/(1-pow(beta1,t)));
	//compute adjusted gradient
	divide_gradients_by_gradients(m,v,network);
	//update parameters
	update_network_by_gradient(network,alpha,inputs->size());
}



DWORD average_gradient_score(NeuralNetwork* network,DTYPE bound,std::vector<std::vector<std::vector<DTYPE> *>*> *inputs,std::vector<std::vector<std::vector<DTYPE> *>*> *outputs,DTYPE *mse){
	DWORD i;
	DWORD score=0;
	mse[0]=0;
	DTYPE temp=0;
	for(i=0;i<inputs->size();i++){
		score+=gradient_score(network,bound,inputs[0][i],outputs[0][i],&temp);
		mse[0]+=temp;
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


void gradient_descent_batch_parallel(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs,DWORD epochs,DTYPE learning_rate){
	DWORD i,j;
	std::vector<NeuralNetwork*> *copy=new std::vector<NeuralNetwork*>(inputs->size());
	for(i=0;i<inputs->size();i++){
		copy[0][i]=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
		copy_neural_network_to_neural_network(network,copy[0][i]);
	}
	for(i=0;i<epochs;i++){
		#pragma omp parallel private(j) num_threads(PROCS)
		{
			#pragma omp for schedule(static)
			for(j=0;j<inputs->size();j++){
				clean_network_states(copy[0][j]);
				compute_gradient(copy[0][j],inputs[0][j],outputs[0][j]);
			}
		}
		for(j=0;j<inputs->size();j++){
			update_network_gradient_by_copy_gradient(network,copy[0][j]);
		}
		average_gradient(network,inputs->size());
		update_network_by_gradient(network,learning_rate,inputs[0][0]->size());
		for(j=0;j<inputs->size();j++){
			copy_neural_network_to_neural_network(network,copy[0][j]);
		}
	}
	for(i=0;i<inputs->size();i++){
		destroy_neural_network(copy[0][i]);
	}
	delete copy;
}

void gradient_descent_batch(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs,DWORD epochs,DTYPE learning_rate){
	DWORD i,j;
	std::vector<NeuralNetwork*> *copy=new std::vector<NeuralNetwork*>(inputs->size());
	for(i=0;i<inputs->size();i++){
		copy[0][i]=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
		copy_neural_network_to_neural_network(network,copy[0][i]);
	}
	for(i=0;i<epochs;i++){
		for(j=0;j<inputs->size();j++){
			clean_network_states(copy[0][j]);
			compute_gradient(copy[0][j],inputs[0][j],outputs[0][j]);
		}
		for(j=0;j<inputs->size();j++){
			update_network_gradient_by_copy_gradient(network,copy[0][j]);
		}
		average_gradient(network,inputs->size());
		update_network_by_gradient(network,learning_rate,inputs[0][0]->size());
		for(j=0;j<inputs->size();j++){
			copy_neural_network_to_neural_network(network,copy[0][j]);
		}
	}
	for(i=0;i<inputs->size();i++){
		destroy_neural_network(copy[0][i]);
	}
	delete copy;
}

DTYPE evaluate_network_sign(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs,std::vector<std::vector<DTYPE> *> *outputs){
	DWORD i,j;
	clean_network_states(network);
	DTYPE result=0,temp;
	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
		for(j=0;j<outputs[0][i]->size();j++){
			temp=(outputs[0][i][0][j]-.5)*(network->output_nodes[0][j].result-.5);
			if(temp>0||(outputs[0][i][0][j]==0.5)){
				result+=1;
			}
		}
	}
	return result/inputs->size();
}

DTYPE evaluate_network_last_sign(NeuralNetwork* network,std::vector<std::vector<DTYPE> *> *inputs){
	DWORD i;
	clean_network_states(network);
	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
	}
	return network->output_nodes[0][0].result;
}


DTYPE evaluate_network_multiple(NeuralNetwork* network,std::vector<std::vector<std::vector<DTYPE> *> *> *inputs,std::vector<std::vector<std::vector<DTYPE> *> *> *outputs){
	DWORD i;
	DTYPE mse=0;
	for(i=0;i<inputs->size();i++){
		mse+=evaluate_network(network,inputs[0][i],outputs[0][i]);
	}
	return mse;
}

std::vector<NeuralNetwork*>* particle_swarm_optimization_by_gradient_batch(std::vector<std::vector<std::vector<DTYPE> *>*> *inputs,std::vector<std::vector<std::vector<DTYPE> *>*> *outputs,DWORD particle_num,DWORD epochs,DWORD input_nodes,DWORD hidden_nodes,DWORD output_nodes,ACTIVATION_TYPE hidden_type,ACTIVATION_TYPE output_type,DTYPE C1,DTYPE C2,DTYPE alpha,DWORD gradient_list_size,DWORD mse_list_size,DTYPE bound){
	//local variables
	DWORD i,j,temp_score;
	std::vector<Particle*> *particles=new std::vector<Particle*>;
	std::vector<DTYPE> *phi1=new std::vector<DTYPE>(particle_num);
	std::vector<DTYPE> *phi2=new std::vector<DTYPE>(particle_num);
	Particle *temp_best=NULL;
	DTYPE temp,best;
	NeuralNetworkVector *global_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
	//initialize all particles
	particles->resize(particle_num);
	for(i=0;i<particle_num;i++){
		//printf("check1-%lld\n",i);
		particles[0][i]=Calloc(1,Particle);
		particles[0][i]->current=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		particles[0][i]->local_best=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		particles[0][i]->velocity=initialize_neural_network_vector(input_nodes,hidden_nodes,output_nodes);
		randomize_weight(particles[0][i]->current);
		//printf("check2-%lld\n",i);
		copy_neural_network_to_vector(particles[0][i]->current,particles[0][i]->local_best);
	}
	//Compute g_best before execution
	best=MAX_DTYPE;
	#pragma omp parallel private(i) num_threads(PROCS)
	{
		#pragma omp for schedule(guided)
		for(i=0;i<particle_num;i++){
			//printf("i=%lld\n",i);
			particles[0][i]->mse=evaluate_network_multiple(particles[0][i]->current,inputs,outputs);
			//printf("initial %lld=%lf\n",i,particles[0][i]->mse);
		}
	}
	for(i=0;i<particle_num;i++){
		if(particles[0][i]->mse<best){
			best=particles[0][i]->mse;
			temp_best=particles[0][i];
		}
	}
	copy_neural_network_to_vector(temp_best->current,global_best);
	//PSO algorithm
	PriorityQueue* gradient_list=create_priority_queue();			//If gradient_list is not full.
	std::vector<Particle*> *mse_list=new std::vector<Particle*>;
	Particle* p;

	for(i=0;i<epochs;i++){
		//printf("epoch %lld, g_best=%lf\n",i,best);
		for(j=0;j<particle_num;j++){
			phi1[0][j]=genrand_real2();
			phi2[0][j]=genrand_real2();
		}
		#pragma omp parallel private(j,temp) num_threads(PROCS)
		{
			#pragma omp for schedule(guided)
			for(j=0;j<particle_num;j++){
				//printf("    processing particle %lld\n",j);
				update_particle(particles[0][j],global_best,C1,C2,alpha,phi1[0][j],phi2[0][j]);
				//temp=evaluate_network_multiple(particles[0][j]->current,inputs,outputs);
				particles[0][j]->gradient_score=average_gradient_score(particles[0][j]->current,bound,inputs,outputs,&temp);
				//gradient_descent_batch(particles[0][j]->current,inputs,outputs,2,.25);
				particles[0][j]->current_mse=temp;
				if(temp<=particles[0][j]->mse){
					particles[0][j]->mse=temp;
					copy_neural_network_to_vector(particles[0][j]->current,particles[0][j]->local_best);
				}
				/*
				if(particles[0][j]->mse<best){
					best=particles[0][j]->mse;
					copy_vector_to_vector(particles[0][j]->local_best,global_best);
				}
				*/
			}
		}
		temp_best=NULL;
		for(j=0;j<particle_num;j++){
			//compute_gradient(particles[0][j]->current,inputs,outputs);
			if(particles[0][j]->mse<best){
				best=particles[0][j]->mse;
				temp_best=particles[0][j];
			}
		}
		if(temp_best!=NULL){
			copy_vector_to_vector(temp_best->local_best,global_best);
		}

		//Sort particles by mse
		std::sort(particles->begin(), particles->end(), mse_compare);
		//copy best particles
		mse_list->insert(mse_list->end(), particles->begin(), particles->begin()+mse_list_size);
		//Sort best particles by gradient
		std::sort(mse_list->begin(), mse_list->end(), gradient_compare);
		for(j=0;j<mse_list_size;j++){
			if(PQ_LENGTH(gradient_list)<gradient_list_size){
				//If gradient_list is not full, insert best particles in order of gradient score
				p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
				insert_to_priority_queue(gradient_list,p);
			}else{
				//If gradient list is full, only insert if the worst particle in the list has gradient score worst than best gradient score 
				p=top_of_priority_queue(gradient_list);
				temp_score=p->gradient_score;
				if(p->gradient_score<mse_list[0][j]->gradient_score||(temp_score==mse_list[0][j]->gradient_score&&j==0)){
					//printf("replacement triggered at i=%lld,j=%lld,new score=%lld,old score=%lld\n",i,j,mse_list[0][j]->gradient_score,p->gradient_score);
					p=pop_from_priority_queue(gradient_list);
					destroy_neural_network(p->current);
					Free(p);
					p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
					insert_to_priority_queue(gradient_list,p);
					if(temp_score==mse_list[0][j]->gradient_score){
						break;
					}
				}else{
					//Mutation factor, force ejection of worst mse particle and replace it with current best gradient.
					
					if(j==0&&p->current_mse>mse_list[0][j]->current_mse){
						//printf("mutation triggered at i=%lld,j=%lld,new score=%lld,old score=%lld\n",i,j,mse_list[0][j]->gradient_score,p->gradient_score);
						p=pop_from_priority_queue(gradient_list);
						destroy_neural_network(p->current);
						Free(p);
						p=copy_current_particle(mse_list[0][j],input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
						insert_to_priority_queue(gradient_list,p);
					}
					
					break;
				
				}
			}
		}
		mse_list->clear();
	}
	//clean up
	delete phi1;
	delete phi2;
	for(i=0;i<particle_num;i++){
		destroy_neural_network_vector(particles[0][i]->local_best);
		destroy_neural_network_vector(particles[0][i]->velocity);
		destroy_neural_network(particles[0][i]->current);
		Free(particles[0][i]);
	}
	delete particles;
	//printf("best=%lf\n",best);
	// Keep global best and all particles in gradient list.
	std::vector<NeuralNetwork*> *result=new std::vector<NeuralNetwork*>;
	NeuralNetwork *network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	copy_vector_to_neural_network(global_best,network);
	result->push_back(network);
	while(!is_priority_queue_empty(gradient_list)){
		p=pop_from_priority_queue(gradient_list);
		result->push_back(p->current);
		Free(p);
	}
	Free(global_best);
	destroy_priority_queue(gradient_list);
	delete mse_list;
	return result;
}


void print_neural_network(NeuralNetwork* network){
	printf("Input dimension: %ld, hidden dimension: %ld, output dimension: %ld\n",network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size());
	DWORD i,j;
	for(i=0;i<network->input_nodes->size();i++){
		printf("Input node %lld\n",i);
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			printf("	link to %lld has weight %lf, gradient is %lf\n",j,network->input_nodes[0][i].weights[0][j],network->input_nodes[0][i].weights_gradient[0][j]);
		}
	}
	for(i=0;i<network->hidden_nodes->size();i++){
		printf("Hidden node %lld, bias=%lf (%lf),context=%lf\n",i,network->hidden_nodes[0][i].bias,network->hidden_nodes[0][i].bias_gradient,network->hidden_nodes[0][i].context);
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			printf("	link to output node %lld has weight %lf, gradient=%lf\n",j,network->hidden_nodes[0][i].weights[0][j],network->hidden_nodes[0][i].weights_gradient[0][j]);
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			printf("	link to hidden node %lld has weight %lf, gradient=%lf\n",j,network->hidden_nodes[0][i].hidden_weights[0][j],network->hidden_nodes[0][i].hidden_weights_gradient[0][j]);
		}
	}
	for(i=0;i<network->output_nodes->size();i++){
		printf("Output bias=%lf (%lf)\n",network->output_nodes[0][i].bias,network->output_nodes[0][i].bias_gradient);
		printf("Output %lld=%lf\n",i,network->output_nodes[0][i].result);
	}
}
