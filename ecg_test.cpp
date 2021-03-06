#include "neural_network.h"
#include "data_functions.h"
#include <math.h>
#include <omp.h>
#define TRAINING_SIZE 500
#define NORMAL_SIZE 1000


void test_result(NeuralNetwork* network,std::vector<std::vector<DTYPE>*> *inputs,std::vector<std::vector<DTYPE>*> *outputs){
	clean_network_states(network);
	DWORD i;
	printf("MSE=%lf\n",evaluate_network(network,inputs,outputs));
	printf("Training model\n");
	FILE *ptr_myfile=fopen("ecg_training.csv","w");
	FILE *error=fopen("ecg_error.csv","w");
	DTYPE square_error;
	std::vector<DTYPE>* next=inputs[0][0];
	std::vector<DTYPE>* temp=new std::vector<DTYPE>(1);
	for(i=0;i<inputs->size();i++){
		compute_network(network,next);
		square_error=fabs(network->output_nodes[0][0].result-outputs[0][i][0][0])*fabs(network->output_nodes[0][0].result-outputs[0][i][0][0]);
		if(i!=inputs->size()-1){
			if(square_error>0.025){
				printf("triggered at time stamp %lld, error=%lf\n",i,square_error);
				temp[0][0]=network->output_nodes[0][0].result;
				next=temp;
			}else{
				next=inputs[0][i+1];
			}
				next=inputs[0][i+1];
		}
		//print_neural_network(network);
		fprintf(ptr_myfile,"%lf\n",network->output_nodes[0][0].result);
		fprintf(error,"%lf\n",square_error);
		//printf("%lf,",network->output_nodes[0][0].result);
	}
	fclose(ptr_myfile);
	fclose(error);
	printf("\n");
	ptr_myfile=fopen("ecg_targets.csv","w");
	printf("Training targets\n");
	for(i=0;i<outputs->size();i++){
		fprintf(ptr_myfile,"%lf\n",outputs[0][i][0][0]);
		//printf("%lf,",outputs[0][i][0][0]);
	}
	fclose(ptr_myfile);
	printf("\n");
	delete temp;
}



int main(){
	struct timeval start,end; /*Timing variables*/
	INTEGER time_elapse;

	FILE *ptr_myfile;
	DWORD particle_num=5361,epochs=60,input_nodes=1,output_nodes=1,hidden_nodes=40,seed=125343,gradient_list_size=31,mse_list_size=10,i;
	DTYPE C1=2,C2=2,alpha=.729,bound=.8872983;
	DTYPE learning_rate=0.05;
	ACTIVATION_TYPE hidden_type=SIGMOID,output_type=SIGMOID;
	std::vector<std::vector<DTYPE>*>* ecg_data= read_csv("ECG_data.txt",UNIVARIATE,FALSE);
	std::vector<std::vector<DTYPE>*>* data=new std::vector<std::vector<DTYPE>*>(TRAINING_SIZE);
	std::vector<std::vector<DTYPE>*>* testing_data=new std::vector<std::vector<DTYPE>*>;

	char filename[200];
	DWORD score,pre_score;
	DTYPE mse,pre_mse;
	ecg_data->resize(1500);
	//ecg_data=filter_data(ecg_data,.2,.8);
	ecg_data=sax_encoding(ecg_data,256);
	//ecg_data=sax_encoding(ecg_data,128);
	//normalize_data(ecg_data);
	soft_normalize_data(ecg_data,.2,.8);


	std::copy(ecg_data->begin(), ecg_data->begin()+TRAINING_SIZE, data->begin());

	init_genrand(3423);
	epochs=10;
	std::vector<std::vector<DTYPE>*>* inputs=new std::vector<std::vector<DTYPE>*>(data->size()-input_nodes);
	std::vector<std::vector<DTYPE>*>* outputs=new std::vector<std::vector<DTYPE>*>(data->size()-output_nodes);
	//printf("check2\n");
	fill_inputs_outputs_multiple(inputs,outputs,data,output_nodes);

	ecg_data->resize(NORMAL_SIZE);
	testing_data->resize(NORMAL_SIZE-TRAINING_SIZE);
	std::vector<std::vector<DTYPE>*>* test_inputs=new std::vector<std::vector<DTYPE>*>(testing_data->size()-1);
	std::vector<std::vector<DTYPE>*>* test_outputs=new std::vector<std::vector<DTYPE>*>(testing_data->size()-1);
	std::vector<DTYPE> *n_mse=new std::vector<DTYPE>(32);
	std::vector<DTYPE> *n_test_mse=new std::vector<DTYPE>(n_mse->size());
	std::copy(ecg_data->begin()+TRAINING_SIZE, ecg_data->begin()+NORMAL_SIZE, testing_data->begin());
	fill_inputs_outputs(test_inputs,test_outputs,testing_data);
/*
	gettimeofday(&start,NULL);
	std::vector<NeuralNetwork*> *networks=particle_swarm_optimization_by_gradient(test_inputs,test_outputs,particle_num,epochs,input_nodes,hidden_nodes,output_nodes,hidden_type,output_type,C1,C2,alpha,gradient_list_size,mse_list_size,bound);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"ecg%lld.bin",i);
		write_neural_network(filename,networks[0][i]);
	}
	gettimeofday(&end,NULL);
	time_elapse=end.tv_sec-start.tv_sec;
	printf("Time taken for particle swarm optimization is %lld seconds.\n",time_elapse);
*/


	std::vector<NeuralNetwork*> *networks=new std::vector<NeuralNetwork*>;
/*

	networks->resize(11);

	for(i=0;i<networks->size();i++){
		sprintf(filename,"ecg%lld.bin",i);
		networks[0][i]=read_neural_network(filename);
	}


	networks->resize(1);
	printf("recurrent training starts\n");

	gettimeofday(&start,NULL);
	#pragma omp parallel private(i,pre_score,pre_mse,mse,score) num_threads(1)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			//pre_score=gradient_score(networks[0][i],bound,inputs,outputs,&pre_mse);
			//pre_mse=evaluate_network(networks[0][i],inputs,outputs);
			gradient_descent_network(networks[0][i],inputs,outputs,10000,learning_rate);
			//mse=evaluate_network(networks[0][i],inputs,outputs);
			//score=gradient_score(networks[0][i],bound,inputs,outputs,&mse);
			//sprintf(filename,"ecg_refined%lld.bin",i);
			//write_neural_network(filename,networks[0][i]);
			printf("Exerpiment %lld MSE=%lf, gradient score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
		}
	}
	gettimeofday(&end,NULL);
	time_elapse=end.tv_sec-start.tv_sec;
	printf("Time taken for back propagation is %lld seconds.\n",time_elapse);
	return 0;
*/

	//init_genrand(55211352);
	init_genrand(56783);

	networks->resize(n_mse->size());
	for(i=0;i<networks->size();i++){
		networks[0][i]=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		randomize_weight(networks[0][i]);
	}

	learning_rate=.5;


	NeuralNetwork *network;
	DWORD j;
	gettimeofday(&start,NULL);
	#pragma omp parallel private(i,j,network,pre_score,mse,ptr_myfile) num_threads(16)
	{
	#pragma omp for schedule(static)
	for(j=0;j<n_mse->size();j++){
		std::vector<DTYPE> *mse_list=new std::vector<DTYPE>(2001);
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
			//gradient_descent_network(network,inputs,outputs,1,learning_rate);
			adam_gradient_descent_network_one_spoch(network,inputs,outputs,.01,0.1,0.03,0.00000001,v,m,temp,i+1);
		}
		char filename2[200];

		sprintf(filename2,"ecg_adam_curve_new1-%lld.csv",j);
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
/*
	init_genrand(52172);
	#pragma omp parallel private(i,network,pre_score,pre_mse,mse,score) num_threads(32)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			//network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
			//randomize_weight(network);
			network=networks[0][i];
			pre_score=gradient_score(network,bound,inputs,outputs,&pre_mse);
			//pre_mse=evaluate_network(network,inputs,outputs);
			gradient_descent_network(network,inputs,outputs,100,learning_rate);
			//gradient_descent_network(network,inputs,outputs,10000,learning_rate);
			//mse=evaluate_network(network,inputs,outputs);
			score=gradient_score(network,bound,inputs,outputs,&mse);
			sprintf(filename,"ecg_control%lld.bin",i);
			write_neural_network(filename,networks[0][i]);
			//printf("control experiment %lld MSE=%lf, score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
			mse_list[0][i]=mse;
			//destroy_neural_network(network);
		}
	}
*/
/*
	for(i=0;i<mse_list->size();i++){
		printf("%lf,",mse_list[0][i]);
	}
	printf("\n");

	delete mse_list;
*/

/*
	for(i=0;i<mse_list->size();i++){
		printf("%lf,",mse_list[0][i]);
		gradient_score(networks[0][i],bound,test_inputs,test_outputs,&mse);
		mse_list[0][i]=mse;
	}
	printf("\nstart to print testing data\n");
	for(i=0;i<mse_list->size();i++){
		printf("%lf,",mse_list[0][i]);
	}
	printf("\n");
*/


	//test_result(networks[0][0], test_inputs, test_outputs);

	clean_2d_data(data);
	clean_2d_data(inputs);
	clean_2d_data(outputs);

}
