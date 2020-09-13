#include "neural_network.h"
#include "data_functions.h"
#include <math.h>
#include <omp.h>

void test_result(NeuralNetwork* network,std::vector<std::vector<DTYPE>*> *inputs,std::vector<std::vector<DTYPE>*> *outputs){
	clean_network_states(network);
	DWORD i;
	printf("MSE=%lf\n",evaluate_network(network,inputs,outputs));
	printf("Training model\n");
	FILE *ptr_myfile=fopen("ecg_training2.csv","w");
	FILE *error=fopen("ecg_error2.csv","w");
	DTYPE square_error;
	std::vector<DTYPE>* next=inputs[0][0];
	std::vector<DTYPE>* temp=new std::vector<DTYPE>(1);
	for(i=0;i<inputs->size();i++){
		compute_network(network,next);
		square_error=(network->output_nodes[0][0].result-outputs[0][i][0][0])*(network->output_nodes[0][0].result-outputs[0][i][0][0]);
		if(i!=inputs->size()-1){
			if(square_error>0.05){
				printf("triggered at time stamp %lld, error=%lf\n",i,square_error);
				//temp[0][0]=network->output_nodes[0][0].result;
				//next=temp;
			}else{
				//next=inputs[0][i+1];
			}
			temp[0][0]=network->output_nodes[0][0].result*0.1+inputs[0][i+1][0][0]*0.9;
			next=temp;
		}
		//print_neural_network(network);
		fprintf(ptr_myfile,"%lf\n",network->output_nodes[0][0].result);
		fprintf(error,"%lf\n",square_error);
		//printf("%lf,",network->output_nodes[0][0].result);
	}
	fclose(ptr_myfile);
	fclose(error);
	printf("\n");
	ptr_myfile=fopen("ecg_targets2.csv","w");
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
	DWORD particle_num=5361,epochs=70,input_nodes=1,output_nodes=1,hidden_nodes=40,seed=3195,gradient_list_size=10,mse_list_size=30,i;
	DTYPE C1=2,C2=2,alpha=1,bound=.729;
	DTYPE learning_rate=.05;
	ACTIVATION_TYPE hidden_type=SIGMOID,output_type=SIGMOID;
	std::vector<std::vector<DTYPE>*>* ecg_data= read_csv("ECG_long.txt",UNIVARIATE,FALSE);
	std::vector<std::vector<DTYPE>*>* data=new std::vector<std::vector<DTYPE>*>(700);
	char filename[200];

	//ecg_data=filter_data(ecg_data,.2,.8);

	ecg_data=sax_encoding(ecg_data,128);
	//normalize_data(ecg_data);

	soft_normalize_data(ecg_data,.2,.8);

	std::copy(ecg_data->begin()+3000, ecg_data->begin()+3700, data->begin());

	init_genrand(seed);

	std::vector<std::vector<DTYPE>*>* inputs=new std::vector<std::vector<DTYPE>*>(data->size()-output_nodes);
	std::vector<std::vector<DTYPE>*>* outputs=new std::vector<std::vector<DTYPE>*>(data->size()-output_nodes);
	//printf("check2\n");
	fill_inputs_outputs_multiple(inputs,outputs,data,output_nodes);

	std::vector<NeuralNetwork*> *networks=particle_swarm_optimization_by_gradient(inputs,outputs,particle_num,epochs,input_nodes,hidden_nodes,output_nodes,hidden_type,output_type,C1,C2,alpha,gradient_list_size,mse_list_size,bound);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"ecg_long%lld.bin",i);
		write_neural_network(filename,networks[0][i]);
	}

/*
	std::vector<NeuralNetwork*> *networks=new std::vector<NeuralNetwork*>;
	networks->resize(11);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"ecg_long_refined%lld.bin",i);
		networks[0][i]=read_neural_network(filename);
	}

*/

	DWORD score,pre_score;
	DTYPE mse,pre_mse;
	printf("recurrent training starts\n");

	#pragma omp parallel private(i,pre_score,pre_mse,mse,score) num_threads(11)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			pre_score=gradient_score(networks[0][i],bound,inputs,outputs,&pre_mse);
			//pre_mse=evaluate_network(networks[0][i],inputs,outputs);
			gradient_descent_network(networks[0][i],inputs,outputs,10000,learning_rate);
			//mse=evaluate_network(networks[0][i],inputs,outputs);
			score=gradient_score(networks[0][i],bound,inputs,outputs,&mse);
			sprintf(filename,"ecg_long_refined%lld.bin",i);
			write_neural_network(filename,networks[0][i]);
			printf("Exerpiment %lld MSE=%lf, gradient score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
		}
	}

	//test_result(networks[0][1], inputs, outputs);

/*
	networks->resize(1);
	for(i=0;i<networks->size();i++){
		networks[0][i]=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		randomize_weight(networks[0][i]);
	}
	std::vector<DTYPE> *mse_list=new std::vector<DTYPE>(networks->size());
*/


	NeuralNetwork *network;
	#pragma omp parallel private(i,network,pre_score,pre_mse,mse,score) num_threads(11)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
			randomize_weight(network);
			//network=networks[0][i];
			pre_score=gradient_score(network,bound,inputs,outputs,&pre_mse);
			//pre_mse=evaluate_network(network,inputs,outputs);
			//gradient_descent_network(network,inputs,outputs,15000,learning_rate);
			gradient_descent_network(network,inputs,outputs,20000,learning_rate);
			//mse=evaluate_network(network,inputs,outputs);
			score=gradient_score(network,bound,inputs,outputs,&mse);
			printf("control experiment %lld MSE=%lf, score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
			//mse_list[0][i]=mse;
			//destroy_neural_network(network);
		}
	}

/*
	for(i=0;i<mse_list->size();i++){
		printf("%lf,",mse_list[0][i]);
	}
	printf("\n");

	delete mse_list;
*/
	
	data->clear();
	data->resize(700);
	std::copy(ecg_data->begin()+3700, ecg_data->begin()+4400, data->begin());
	std::vector<std::vector<DTYPE>*>* test_inputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);
	std::vector<std::vector<DTYPE>*>* test_outputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);

	fill_inputs_outputs(test_inputs,test_outputs,data);
	test_result(networks[0][0], test_inputs, test_outputs);

	clean_2d_data(data);
	clean_2d_data(inputs);
	clean_2d_data(outputs);

}
