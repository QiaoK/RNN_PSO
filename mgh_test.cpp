#include "neural_network.h"
#include "data_functions.h"
#include <math.h>
#include <omp.h>
#define TRAINING_SIZE 1800
#define NORMAL_SIZE 3600

void test_result(NeuralNetwork* network,std::vector<std::vector<DTYPE>*> *inputs,std::vector<std::vector<DTYPE>*> *outputs){
	clean_network_states(network);
	DWORD i;
	printf("Training model\n");
	for(i=0;i<inputs->size();i++){
		compute_network(network,inputs[0][i]);
		//print_neural_network(network);
		printf("%lf,",network->output_nodes[0][0].result);
	}
	printf("\n");
	printf("Training targets\n");
	for(i=0;i<outputs->size();i++){
		printf("%lf,",outputs[0][i][0][0]);
	}
	printf("\n");
}



int main(){
	struct timeval start,end; /*Timing variables*/
	INTEGER time_elapse;

	FILE *ptr_myfile;
	//used parameters 5234
	DWORD particle_num=5041,epochs=40,input_nodes=3,output_nodes=3,hidden_nodes=40,seed=226425,gradient_list_size=31,mse_list_size=10,i;
	DTYPE C1=2,C2=2,alpha=0.729,bound=4.79;
	DTYPE learning_rate=.05;
	ACTIVATION_TYPE hidden_type=SIGMOID,output_type=SIGMOID;
	std::vector<std::vector<DTYPE>*>* data=read_csv("MGHMF/mgh003.csv",MGHMF,TRUE);
	std::vector<std::vector<DTYPE>*>* testing_data=new std::vector<std::vector<DTYPE>*>;
	std::vector<std::vector<DTYPE>*>* training_data=new std::vector<std::vector<DTYPE>*>(TRAINING_SIZE);
	char filename[200];
	DWORD score,pre_score;
	DTYPE mse,pre_mse;
	data=sax_encoding(data,256);
	soft_normalize_data(data,.2,.8);


	init_genrand(seed);
	std::copy(data->begin(), data->begin()+TRAINING_SIZE, training_data->begin());

	std::vector<std::vector<DTYPE>*>* inputs=new std::vector<std::vector<DTYPE>*>(training_data->size()-1);
	std::vector<std::vector<DTYPE>*>* outputs=new std::vector<std::vector<DTYPE>*>(training_data->size()-1);
	fill_inputs_outputs(inputs,outputs,training_data);

	data->resize(NORMAL_SIZE);
	testing_data->resize(NORMAL_SIZE-TRAINING_SIZE);
	std::vector<std::vector<DTYPE>*>* test_inputs=new std::vector<std::vector<DTYPE>*>(testing_data->size()-1);
	std::vector<std::vector<DTYPE>*>* test_outputs=new std::vector<std::vector<DTYPE>*>(testing_data->size()-1);
	std::vector<DTYPE> *n_mse=new std::vector<DTYPE>(32);
	std::vector<DTYPE> *n_test_mse=new std::vector<DTYPE>(n_mse->size());
	std::copy(data->begin()+TRAINING_SIZE, data->begin()+NORMAL_SIZE, testing_data->begin());
	fill_inputs_outputs(test_inputs,test_outputs,testing_data);
	epochs=10;
/*
	gettimeofday(&start,NULL);
	std::vector<NeuralNetwork*> *networks=particle_swarm_optimization_by_gradient(inputs,outputs,particle_num,epochs,input_nodes,hidden_nodes,output_nodes,hidden_type,output_type,C1,C2,alpha,gradient_list_size,mse_list_size,bound);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"mgh%lld.bin",i);
		write_neural_network(filename,networks[0][i]);
	}
	gettimeofday(&end,NULL);
	time_elapse=end.tv_sec-start.tv_sec;
	printf("Time taken for particle swarm optimization is %lld seconds.\n",time_elapse);
*/

	std::vector<NeuralNetwork*> *networks=new std::vector<NeuralNetwork*>;
	init_genrand(1632);
	//init_genrand(42351);//For purpose of new/old test
	networks->resize(n_mse->size());
	for(i=0;i<networks->size();i++){
		networks[0][i]=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		randomize_weight(networks[0][i]);
	}


/*
	for(i=0;i<networks->size();i++){
		sprintf(filename,"mgh%lld.bin",i);
		networks[0][i]=read_neural_network(filename);
	}
*/
	learning_rate=0.5;

	NeuralNetwork *network;
	DWORD j;
	gettimeofday(&start,NULL);
	#pragma omp parallel private(i,j,network,pre_score,mse,ptr_myfile) num_threads(16)
	{
	#pragma omp for schedule(static)
	for(j=0;j<n_mse->size();j++){
		std::vector<DTYPE> *mse_list=new std::vector<DTYPE>(1501);
		std::vector<DTYPE> *mse_test_list=new std::vector<DTYPE>(mse_list->size());
		network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		copy_neural_network_to_neural_network(networks[0][j],network);
		NeuralNetwork *v=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
		NeuralNetwork *m=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
		NeuralNetwork *temp=initialize_neural_network(network->input_nodes->size(),network->hidden_nodes->size(),network->output_nodes->size(),network->hidden_type,network->output_type);
		for(i=0;i<mse_list->size();i++){
			//printf("epoch %lld\n",i);
			gradient_score(network,bound,inputs,outputs,&mse);
			mse_list[0][i]=mse;
			gradient_score(network,bound,test_inputs,test_outputs,&mse);
			mse_test_list[0][i]=mse;//-mse_list[0][i];
			gradient_descent_network(network,inputs,outputs,1,learning_rate);
			//adam_gradient_descent_network_one_spoch(network,test_inputs,test_outputs,.01,0.1,0.03,0.00000001,v,m,temp,i+1);
		}

		char filename2[200];

		sprintf(filename2,"MGH_curve_benchmark1-%lld.csv",j);
		ptr_myfile=fopen(filename2,"w");
		for(i=0;i<mse_list->size();i++){
			//printf("%lf,",mse_list[0][i]);
			fprintf(ptr_myfile,"%lf,%lf\n",mse_list[0][i],mse_test_list[0][i]);
		}
		fclose(ptr_myfile);

		n_mse[0][j]=mse_list[0][mse_list->size()-1];
		n_test_mse[0][j]=mse_test_list[0][mse_list->size()-1];
		destroy_neural_network(network);
		destroy_neural_network(v);
		destroy_neural_network(m);
		destroy_neural_network(temp);
		delete mse_list;
		delete mse_test_list;
	}
	}
	gettimeofday(&end,NULL);
	time_elapse=end.tv_sec-start.tv_sec;
	printf("Time taken for back propagation is %lld seconds.\n",time_elapse);
	for(j=0;j<n_mse->size();j++){
		printf("%lf,%lf\n",n_mse[0][j],n_test_mse[0][j]);
	}

	clean_2d_data(data);
	clean_2d_data(inputs);
	clean_2d_data(outputs);
}
