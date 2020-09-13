#include "neural_network.h"
#include "data_functions.h"
#include <math.h>
#include <omp.h>

void test_result(NeuralNetwork* network,std::vector<std::vector<DTYPE>*> *data,DWORD n_output){
	clean_network_states(network);
	DWORD i,j;
	printf("Training model\n");
	FILE *ptr_myfile=fopen("power_training.csv","w");
	FILE *error=fopen("power_error.csv","w");
	DTYPE square_error,mean;
	std::vector<DTYPE>* next=data[0][0];
	std::vector<DTYPE>* temp=new std::vector<DTYPE>(1);
	std::vector<std::vector<DTYPE>*> *outputs=new std::vector<std::vector<DTYPE>*>(data->size());
	for(i=0;i<outputs->size();i++){
		outputs[0][i]=new std::vector<DTYPE>;
	}

	for(i=0;i<data->size()-n_output;i++){
		compute_network(network,next);
		for(j=0;j<n_output;j++){
			outputs[0][i+1+j]->push_back(network->output_nodes[0][j].result);
		}
		if(i>0){
			mean=0;
			for(j=0;j<outputs[0][i]->size();j++){
				mean+=outputs[0][i][0][j];
			}
			mean/=outputs[0][i]->size();
			square_error=(mean-data[0][i][0][0])*(mean-data[0][i][0][0]);
			if(square_error>0.05){
				//printf("triggered at time stamp %lld, error=%lf\n",i,square_error);
				temp[0][0]=network->output_nodes[0][0].result;
				next=temp;
			}else{
				next=data[0][i+1];
			}
			next=data[0][i+1];
			//print_neural_network(network);
			fprintf(ptr_myfile,"%lf\n",mean);
			fprintf(error,"%lf\n",square_error);
			//printf("%lf,",network->output_nodes[0][0].result);
		}
	}
	for(i=data->size()-n_output;i<data->size();i++){
		mean=0;
		for(j=0;j<outputs[0][i]->size();j++){
			mean+=outputs[0][i][0][j];
		}
		mean/=outputs[0][i]->size();
		square_error=(mean-data[0][i][0][0])*(mean-data[0][i][0][0]);
		fprintf(ptr_myfile,"%lf\n",mean);
		fprintf(error,"%lf\n",square_error);
	}
	fclose(ptr_myfile);
	fclose(error);
	printf("\n");
	ptr_myfile=fopen("power_targets.csv","w");
	printf("Training targets\n");
	for(i=1;i<data->size();i++){
		fprintf(ptr_myfile,"%lf\n",data[0][i][0][0]);
		//printf("%lf,",outputs[0][i][0][0]);
	}
	fclose(ptr_myfile);
	printf("\n");
	delete temp;
}




int main(){
	DWORD particle_num=5361,epochs=40,input_nodes=1,output_nodes=3,hidden_nodes=40,seed=158692,gradient_list_size=10,mse_list_size=5,i;
	DTYPE C1=1.5,C2=1.5,alpha=.729,bound=.8872983;
	DTYPE learning_rate=.05;
	ACTIVATION_TYPE hidden_type=TANH,output_type=SIGMOID;
	std::vector<std::vector<DTYPE>*>* power_data= read_csv("power_demand.txt",UNIVARIATE,FALSE);
	std::vector<std::vector<DTYPE>*>* data=new std::vector<std::vector<DTYPE>*>(6000);
/*
	for(i=0;i<power_data->size();i++){
		printf("%lf\n",power_data[0][i][0][0]);
	}
*/
	//power_data=filter_data(power_data,.2,.8);

	power_data=sax_encoding(power_data,512);

	normalize_data(power_data);

	for(i=0;i<power_data->size();i++){
		//printf("%lf\n",power_data[0][i][0][0]);
	}


	std::copy(power_data->begin(), power_data->begin()+6000, data->begin());

	init_genrand(seed);

	std::vector<std::vector<DTYPE>*>* inputs=new std::vector<std::vector<DTYPE>*>(data->size()-output_nodes);
	std::vector<std::vector<DTYPE>*>* outputs=new std::vector<std::vector<DTYPE>*>(data->size()-output_nodes);
	//printf("check2\n");
	fill_inputs_outputs_multiple(inputs,outputs,data,output_nodes);

	//std::vector<NeuralNetwork*> *networks=particle_swarm_optimization_by_gradient(inputs,outputs,particle_num,epochs,input_nodes,hidden_nodes,output_nodes,hidden_type,output_type,C1,C2,alpha,gradient_list_size,mse_list_size,bound);
	std::vector<NeuralNetwork*> *networks=new std::vector<NeuralNetwork*>;
	char filename[200];


	networks->resize(11);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"power%lld.bin",i);
		networks[0][i]=read_neural_network(filename);
	}


	DWORD score,pre_score;
	DTYPE mse,pre_mse;
	printf("recurrent training starts\n");
	#pragma omp parallel private(i,pre_score,pre_mse,mse,score) num_threads(11)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			pre_score=gradient_score(networks[0][i],bound,inputs,outputs,&pre_mse);
			sprintf(filename,"power%lld.bin",i);
			write_neural_network(filename,networks[0][i]);
			//pre_mse=evaluate_network(networks[0][i],inputs,outputs);
			gradient_descent_network(networks[0][i],inputs,outputs,10000,learning_rate);
			//mse=evaluate_network(networks[0][i],inputs,outputs);
			score=gradient_score(networks[0][i],bound,inputs,outputs,&mse);
			sprintf(filename,"powered_refined%lld.bin",i);
			write_neural_network(filename,networks[0][i]);
			printf("Exerpiment %lld MSE=%lf, gradient score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
		}
	}
	//test_result(networks[0][1], inputs, outputs);

	NeuralNetwork *network;
	#pragma omp parallel private(i,network,pre_score,pre_mse,mse,score) num_threads(11)
	{
		#pragma omp for schedule(static)
		for(i=0;i<networks->size();i++){
			network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
			randomize_weight(network);
			pre_score=gradient_score(network,bound,inputs,outputs,&pre_mse);
			//pre_mse=evaluate_network(network,inputs,outputs);
			gradient_descent_network(network,inputs,outputs,10000,learning_rate);
			//mse=evaluate_network(network,inputs,outputs);
			score=gradient_score(network,bound,inputs,outputs,&mse);
			printf("control experiment %lld MSE=%lf, score=%lld, prescore=%lld, previous mse=%lf\n",i,mse,score,pre_score,pre_mse);
			destroy_neural_network(network);
		}
	}

	data->clear();
	data->resize(9000);
	std::copy(power_data->begin(), power_data->begin()+9000, data->begin());
	std::vector<std::vector<DTYPE>*>* test_inputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);
	std::vector<std::vector<DTYPE>*>* test_outputs=new std::vector<std::vector<DTYPE>*>(data->size()-1);
	fill_inputs_outputs_multiple(test_inputs,test_outputs,data,output_nodes);

	test_result(networks[0][0], data, output_nodes);

	clean_2d_data(data);
	clean_2d_data(inputs);
	clean_2d_data(outputs);

}
