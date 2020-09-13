#include "data_functions.h"
#include "neural_network.h"
#include "random.h"
#include <math.h>

int main(){
	DWORD particle_num=10051,epochs=40,input_nodes=8,output_nodes=8,hidden_nodes=50,seed=3123,gradient_list_size=10,mse_list_size=5,i;
	DTYPE C1=2,C2=2,alpha=0.729,bound=0.9899492;
	DTYPE learning_rate=.05;
	ACTIVATION_TYPE hidden_type=TANH,output_type=SIGMOID;
	char filename[200];
	std::vector<std::vector<std::vector<DTYPE>*>*>* mgh_data=read_MGH_data(11,50);
	std::vector<std::vector<std::vector<DTYPE>*>*> *inputs=new std::vector<std::vector<std::vector<DTYPE>*>*>(mgh_data->size());
	std::vector<std::vector<std::vector<DTYPE>*>*> *outputs=new std::vector<std::vector<std::vector<DTYPE>*>*>(mgh_data->size());
	std::vector<std::vector<std::vector<DTYPE>*>*>* mgh_test_data=read_MGH_data(1,10);
	std::vector<std::vector<std::vector<DTYPE>*>*> *test_inputs=new std::vector<std::vector<std::vector<DTYPE>*>*>(mgh_test_data->size());
	std::vector<std::vector<std::vector<DTYPE>*>*> *test_outputs=new std::vector<std::vector<std::vector<DTYPE>*>*>(mgh_test_data->size());
	for(i=0;i<mgh_data->size();i++){
		inputs[0][i]=new std::vector<std::vector<DTYPE>*>(mgh_data[0][i]->size()-1);
		outputs[0][i]=new std::vector<std::vector<DTYPE>*>(mgh_data[0][i]->size()-1);
		normalize_data(mgh_data[0][i]);
		mgh_data[0][i]=sax_encoding(mgh_data[0][i],512);
		fill_inputs_outputs(inputs[0][i],outputs[0][i],mgh_data[0][i]);
	}
	for(i=0;i<mgh_test_data->size();i++){
		test_inputs[0][i]=new std::vector<std::vector<DTYPE>*>(mgh_test_data[0][i]->size()-1);
		test_outputs[0][i]=new std::vector<std::vector<DTYPE>*>(mgh_test_data[0][i]->size()-1);
		normalize_data(mgh_test_data[0][i]);
		mgh_test_data[0][i]=sax_encoding(mgh_test_data[0][i],512);
		fill_inputs_outputs(test_inputs[0][i],test_outputs[0][i],mgh_test_data[0][i]);
	}
	init_genrand(seed);
	//NeuralNetwork *network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);

/*
	std::vector<std::vector<DTYPE>*>* test_data= read_csv("MGHMF/mgh042.csv",MGHMF,FALSE);

	normalize_data(test_data);
	test_data=sax_encoding(test_data,512);

	std::vector<std::vector<DTYPE>*>* input=new std::vector<std::vector<DTYPE>*>(test_data->size()-1);
	std::vector<std::vector<DTYPE>*>* output=new std::vector<std::vector<DTYPE>*>(test_data->size()-1);
	fill_inputs_outputs(input,output,test_data);
*/
/*
	std::vector<NeuralNetwork*> *networks=particle_swarm_optimization_by_gradient_batch(inputs,outputs,particle_num,epochs,input_nodes,hidden_nodes,output_nodes,hidden_type,output_type,C1,C2,alpha,gradient_list_size,mse_list_size,bound);

	for(i=0;i<networks->size();i++){
		sprintf(filename,"mgh_batch%lld.bin",i);
		write_neural_network(filename,networks[0][i]);
	}
*/
	std::vector<NeuralNetwork*> *networks=new std::vector<NeuralNetwork*>;

	networks->resize(11);
	for(i=0;i<networks->size();i++){
		sprintf(filename,"mgh_batch%lld.bin",i);
		networks[0][i]=read_neural_network(filename);
	}
	


	DWORD score,pre_score;
	DTYPE mse,pre_mse,multiple_pre_mse,multiple_mse;
	printf("recurrent training starts\n");

	for(i=0;i<networks->size();i++){
		pre_score=average_gradient_score(networks[0][i],bound,inputs,outputs,&multiple_pre_mse);
		pre_mse=evaluate_network_multiple(networks[0][i],test_inputs,test_outputs);
		//multiple_pre_mse=evaluate_network_multiple(networks[0][i],inputs,outputs);
		gradient_descent_batch_parallel(networks[0][i],inputs,outputs,10000,learning_rate);
		mse=evaluate_network_multiple(networks[0][i],test_inputs,test_outputs);
		//multiple_mse=evaluate_network_multiple(networks[0][i],inputs,outputs);
		score=average_gradient_score(networks[0][i],bound,inputs,outputs,&multiple_mse);
		sprintf(filename,"mgh_batch_refined%lld.bin",i);
		write_neural_network(filename,networks[0][i]);
		printf("Exerpiment %lld MSE=%lf (%lf), gradient score=%lld, prescore=%lld, previous mse=%lf (%lf)\n",i,mse,multiple_mse,score,pre_score,pre_mse,multiple_pre_mse);

	}

	networks->resize(11);
	for(i=0;i<networks->size();i++){
		networks[0][i]=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		randomize_weight(networks[0][i]);
	}

	//init_genrand(63421);
	NeuralNetwork *network;
	std::vector<DTYPE> *mse_list=new std::vector<DTYPE>(networks->size());
	for(i=0;i<networks->size();i++){
		//network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
		//randomize_weight(network);
		network=networks[0][i];
		pre_score=average_gradient_score(network,bound,inputs,outputs,&multiple_pre_mse);
		pre_mse=evaluate_network_multiple(network,test_inputs,test_outputs);
		//multiple_pre_mse=evaluate_network_multiple(network,inputs,outputs);
		gradient_descent_batch_parallel(network,inputs,outputs,10000,learning_rate);
		mse=evaluate_network_multiple(network,test_inputs,test_outputs);
		//multiple_mse=evaluate_network_multiple(network,inputs,outputs);
		score=average_gradient_score(network,bound,inputs,outputs,&multiple_mse);
		mse_list[0][i]=multiple_mse;
		printf("Controlled exerpiment %lld MSE=%lf (%lf), gradient score=%lld, prescore=%lld, previous mse=%lf (%lf)\n",i,mse,multiple_mse,score,pre_score,pre_mse,multiple_pre_mse);
		destroy_neural_network(network);
	}
	for(i=0;i<mse_list->size();i++){
		printf("%lf,",mse_list[0][i]);
	}
	printf("\n");
	delete mse_list;

/*
	print_neural_network(networks[0][0]);
	clean_2d_data(test_data);
	clean_2d_data(input);
	clean_2d_data(output);
*/
	clean_3d_data(mgh_data);
	clean_3d_data(inputs);
	clean_3d_data(outputs);
	clean_3d_data(mgh_test_data);
	clean_3d_data(test_inputs);
	clean_3d_data(test_outputs);

}
