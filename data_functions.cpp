#include "data_functions.h"
#include "random.h"
#include <math.h>


DWORD count_lines(const char* filename){
	DWORD lines=0;
	FILE* fd=fopen(filename,"r");
	char c;
	while((c=fgetc(fd))!=EOF){
		if(c=='\n'){
			lines++;
		}
	}
	return lines;
}

std::vector<DTYPE>* read_MGHMF(char* line){
	std::vector<DTYPE>* result=new std::vector<DTYPE>;
	result->resize(3);

	char* cpt=line;
	DWORD i,size;
	//Filter out the first value, which is time stamp.
	while(*cpt!=','){
		cpt++;
	}
	cpt++;
	//parse first seven values
	for(i=0;i<7;i++){
		line=cpt;
		size=0;
		while(*cpt!=','){
			cpt++;
			size++;
		}
		cpt++;
		char x[size+1];
		x[size]='\0';
		memcpy(x,line,sizeof(char)*size);
		//printf("%s",x);
		if(i>=3&&i<=5)
		result[0][i-3]=atof(x);
	}
	//parse for last value
	size=0;
	line=cpt;
	while(*cpt!='\0'){
		cpt++;
		size++;
	}
	cpt++;
	char t[size+1];
	t[size]='\0';
	memcpy(t,line,sizeof(char)*size);
	//result[0][7]=atof(t);
	return result;
}

std::vector<DTYPE>* read_space_shuttle(char* line){
	std::vector<DTYPE>* result=new std::vector<DTYPE>;
	char* cpt=line;
	DWORD size;
	result->resize(1);
	//Filter out the first value, which is time stamp.
	while(*cpt!=','){
		cpt++;
	}
	//printf("check 2\n");
	cpt++;
	//parse value
	line=cpt;
	size=0;
	while(*cpt!='\0'){
		cpt++;
		size++;
	}
	cpt++;
	char x[size+1];
	x[size]='\0';
	memcpy(x,line,sizeof(char)*size);
	//printf("%s",x);
	result[0][0]=atof(x);
	return result;
}

std::vector<DTYPE>* read_power_demand(char* line){
	std::vector<DTYPE>* result=new std::vector<DTYPE>;
	char* cpt=line;
	DWORD size;
	result->resize(1);
	//parse value
	line=cpt;
	size=0;
	while(*cpt!='\0'){
		cpt++;
		size++;
	}
	cpt++;
	char x[size+1];
	x[size]='\0';
	memcpy(x,line,sizeof(char)*size);
	//printf("%s",x);
	result[0][0]=atof(x);
	return result;
}

std::vector<std::vector<DTYPE>*>* read_csv(const char* filename,WORD type,BOOLEAN header){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD lines=count_lines(filename);
	std::vector<std::vector<DTYPE>*>* result=new std::vector<std::vector<DTYPE>*>;
	DWORD length=0;
	char* line;
	char c;
	if(header){
		lines-=2;
		fgetc(stream2);
		while((c=fgetc(stream1))!='\n'){
			fgetc(stream2);
		}
		fgetc(stream2);
		while((c=fgetc(stream1))!='\n'){
			fgetc(stream2);
		}
	}
	DWORD i;
	result->resize(lines);
	i=0;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			switch(type){
				case MGHMF:
					result[0][i]=read_MGHMF(line);
					//printf("%s\n",line);
					break;
				case  SPACE_SHUTTLE:
					//printf("%s\n",line);
					result[0][i]=read_space_shuttle(line);
					break;
				case  UNIVARIATE:
					//printf("%s\n",line);
					result[0][i]=read_power_demand(line);
					break;
				default:
					printf("unrecognized type, exit.\n");
					return NULL;
			}
			i++;
			length=0;
			Free(line);
		}
	}
	fclose(stream1);
	fclose(stream2);
	return result;
}

std::vector<char*>* read_stock_list(const char* filename){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD length=0;
	char* line;
	char c;
	std::vector<char*>* result=new std::vector<char*>;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			result->push_back(line);
			length=0;
		}
	}
	fclose(stream1);
	fclose(stream2);
	return result;
}

void normalize_data(std::vector<std::vector<DTYPE>*>* data){
	std::vector<Bound>* bound=new std::vector<Bound>;
	bound->resize(data[0][0]->size());
	DWORD i,j;
	for(i=0;i<bound->size();i++){
		bound[0][i].upper=data[0][0][0][i];
		bound[0][i].lower=data[0][0][0][i];
	}

	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			if(data[0][i][0][j]>bound[0][j].upper){
				bound[0][j].upper=data[0][i][0][j];
			}
			if(data[0][i][0][j]<bound[0][j].lower){
				bound[0][j].lower=data[0][i][0][j];
			}
		}
	}	
	BOOLEAN debug=0;
	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			if(bound[0][j].upper==bound[0][j].lower){
				data[0][i][0][j]=.5;
				if(debug){
					printf("triggered column %lld %lf=%lf\n",j,bound[0][j].upper,bound[0][j].lower);
					debug=0;
				}
			}else{	
				data[0][i][0][j]=(data[0][i][0][j]-bound[0][j].lower)/(bound[0][j].upper-bound[0][j].lower);
			}
		}
	}
	delete bound;
}

void read_feature(char* line,std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,DWORD input_size,DWORD output_size){
	std::vector<DTYPE>* input=new std::vector<DTYPE>;
	std::vector<DTYPE>* output=new std::vector<DTYPE>;
	input->resize(input_size);
	output->resize(output_size);

	char* cpt=line;
	DWORD i,size;
	//Filter out the first value, which is time stamp.
	//parse inputs
	for(i=0;i<input_size;i++){
		line=cpt;
		size=0;
		while(*cpt!=','){
			cpt++;
			size++;
		}
		cpt++;
		char x[size+1];
		x[size]='\0';
		memcpy(x,line,sizeof(char)*size);
		input[0][i]=atof(x);
	}
	//parse outputs
	for(i=0;i<output_size-1;i++){
		line=cpt;
		size=0;
		while(*cpt!=','){
			cpt++;
			size++;
		}
		cpt++;
		char x[size+1];
		x[size]='\0';
		memcpy(x,line,sizeof(char)*size);
		output[0][i]=atof(x);
	}
	//parse for last value
	size=0;
	line=cpt;
	while(*cpt!='\0'){
		cpt++;
		size++;
	}
	cpt++;
	char x[size+1];
	x[size]='\0';
	memcpy(x,line,sizeof(char)*size);
	output[0][output_size-1]=atof(x);
	//result[0][7]=atof(t);
	inputs->push_back(input);
	outputs->push_back(output);
}



void read_features(const char* filename,std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,DWORD input_size,DWORD output_size){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD length=0;
	char* line;
	char c;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			read_feature(line,inputs,outputs,input_size,output_size);
			length=0;
			Free(line);
		}
	}
	fclose(stream1);
	fclose(stream2);
}

std::vector<char*>* read_stock_codes(const char* filename){
	std::vector<char*>* codes=new std::vector<char*>;
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD lines=count_lines(filename);
	codes->resize(lines);
	DWORD length=0;
	char* line;
	char c;
	DWORD i=0;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			codes[0][i]=line;
			i++;
			length=0;
		}
	}
	fclose(stream1);
	fclose(stream2);
	return codes;
}

void read_feature_meta(const char* filename,std::vector<char*> *types,std::vector<std::vector<char*>*>* stock_codes){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD lines=count_lines(filename);
	stock_codes->resize(lines);
	types->resize(lines);
	DWORD length=0;
	char* line;
	char c;
	DWORD i=0;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			fgetc(stream2);
			line[length]='\0';
			char name[strlen(line)+20];
			strcpy(name,"meta_data/");
			strcat(name,line);
			strcat(name,".txt");
			//printf("%s\n",name);
			stock_codes[0][i]=read_stock_codes(name);
			types[0][i]=line;
			i++;
			length=0;
		}
	}
	fclose(stream1);
	fclose(stream2);
}

char* make_stock_filename(char* type,char* code){
	char* result=Calloc(strlen(type)+strlen(code)+20,char);
	strcpy(result,"features/");
	strcat(result,type);
	strcat(result,"/");
	strcat(result,code);
	strcat(result,".csv");
	return result;
}

void get_feature_dimensions(const char* filename,DWORD* input_size,DWORD* output_size){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD length=0;
	char* line;
	char c;
	DWORD i=0;
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			i++;
			switch(i){
				case 1:{
					*input_size=atoi(line);
					break;
				}
				case 2:{
					*output_size=atoi(line);
					break;
				}
			}
			length=0;
		}
	}
	fclose(stream1);
	fclose(stream2);
}

void soft_normalize_data(std::vector<std::vector<DTYPE>*>* data,DTYPE lower,DTYPE upper){
	std::vector<Bound>* bound=new std::vector<Bound>;
	bound->resize(data[0][0]->size());
	DWORD i,j;
	for(i=0;i<bound->size();i++){
		bound[0][i].upper=data[0][0][0][i];
		bound[0][i].lower=data[0][0][0][i];
	}

	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			if(data[0][i][0][j]>bound[0][j].upper){
				bound[0][j].upper=data[0][i][0][j];
			}
			if(data[0][i][0][j]<bound[0][j].lower){
				bound[0][j].lower=data[0][i][0][j];
			}
		}
	}	
	BOOLEAN debug=0;
	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			if(bound[0][j].upper==bound[0][j].lower){
				data[0][i][0][j]=(upper-lower)/2;
				if(debug){
					printf("triggered column %lld %lf=%lf\n",j,bound[0][j].upper,bound[0][j].lower);
					debug=0;
				}
			}else{	
				data[0][i][0][j]=lower+(upper-lower)*(data[0][i][0][j]-bound[0][j].lower)/(bound[0][j].upper-bound[0][j].lower);
			}
		}
	}
	delete bound;
}


std::vector<std::vector<std::vector<DTYPE>*>*>* read_MGH_data(DWORD start,DWORD end){
	std::vector<std::vector<std::vector<DTYPE>*>*>* result=new std::vector<std::vector<std::vector<DTYPE>*>*>;
	std::vector<std::vector<DTYPE>*>* mgh_data;
	DWORD i;
	char filename[200];
	for(i=start;i<=end;i++){
		if(i==122){
			continue;
		}
		if(i<10){
			sprintf(filename,"MGHMF/mgh00%lld.csv",i);
		}else{
			if(i<100){
				sprintf(filename,"MGHMF/mgh0%lld.csv",i);
			}else{
				sprintf(filename,"MGHMF/mgh%lld.csv",i);
			}
		}
		printf("check %s\n",filename);
		mgh_data=read_csv(filename,MGHMF,TRUE);
		normalize_data(mgh_data);
		printf("data size=%ld\n",mgh_data->size());
		result->push_back(mgh_data);
	}
	return result;
}

NeuralNetwork* read_neural_network(const char* filename){
	FILE *ptr_myfile=fopen(filename,"rb");
	ACTIVATION_TYPE hidden_type,output_type;
	DWORD input_nodes,hidden_nodes,output_nodes;
	fread(&input_nodes, sizeof(DWORD), 1, ptr_myfile);
	fread(&hidden_nodes, sizeof(DWORD), 1, ptr_myfile);
	fread(&output_nodes, sizeof(DWORD), 1, ptr_myfile);
	fread(&hidden_type, sizeof(ACTIVATION_TYPE), 1,ptr_myfile);
	fread(&output_type, sizeof(ACTIVATION_TYPE), 1, ptr_myfile);
	NeuralNetwork *network=initialize_neural_network(input_nodes,hidden_nodes,output_nodes,hidden_type,output_type);
	DWORD i,j;
	DTYPE temp;
	//Read input to hidden weights
	for(i=0;i<network->input_nodes->size();i++){
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			fread(&temp, sizeof(DTYPE), 1, ptr_myfile);
			network->input_nodes[0][i].weights[0][j]=temp;
		}
	}
	//Read hidden to output weights, self loop coefficient, and bias
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			fread(&temp, sizeof(DTYPE), 1, ptr_myfile);
			network->hidden_nodes[0][i].weights[0][j]=temp;
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			fread(&temp, sizeof(DTYPE), 1, ptr_myfile);
			network->hidden_nodes[0][i].hidden_weights[0][j]=temp;
		}
		fread(&temp, sizeof(DTYPE), 1, ptr_myfile);
		network->hidden_nodes[0][i].bias=temp;
	}
	//Read bias for output node
	for(i=0;i<network->output_nodes->size();i++){
		fread(&temp, sizeof(DTYPE), 1, ptr_myfile);
		network->output_nodes[0][i].bias=temp;
	}
	fclose(ptr_myfile);
	return network;
}

void write_neural_network(const char* filename,NeuralNetwork *network){
	FILE *ptr_myfile=fopen(filename,"wb");
	DWORD input_nodes=network->input_nodes->size(),hidden_nodes=network->hidden_nodes->size(),output_nodes=network->output_nodes->size();
	fwrite(&input_nodes, sizeof(DWORD), 1, ptr_myfile);
	fwrite(&hidden_nodes, sizeof(DWORD), 1, ptr_myfile);
	fwrite(&output_nodes, sizeof(DWORD), 1, ptr_myfile);
	fwrite(&(network->hidden_type), sizeof(ACTIVATION_TYPE), 1, ptr_myfile);
	fwrite(&(network->output_type), sizeof(ACTIVATION_TYPE), 1, ptr_myfile);
	DWORD i,j;
	DTYPE temp;

	//Write input to hidden weights
	for(i=0;i<network->input_nodes->size();i++){
		for(j=0;j<network->input_nodes[0][i].weights->size();j++){
			temp=network->input_nodes[0][i].weights[0][j];
			fwrite(&temp, sizeof(DTYPE), 1, ptr_myfile);
		}
	}
	//Write hidden to output weights, self loop coefficient, and bias
	for(i=0;i<network->hidden_nodes->size();i++){
		for(j=0;j<network->hidden_nodes[0][i].weights->size();j++){
			temp=network->hidden_nodes[0][i].weights[0][j];
			fwrite(&temp, sizeof(DTYPE), 1, ptr_myfile);
		}
		for(j=0;j<network->hidden_nodes[0][i].hidden_weights->size();j++){
			temp=network->hidden_nodes[0][i].hidden_weights[0][j];
			fwrite(&temp, sizeof(DTYPE), 1, ptr_myfile);
		}
		temp=network->hidden_nodes[0][i].bias;
		fwrite(&temp, sizeof(DTYPE), 1, ptr_myfile);
	}
	//Write bias for output node
	for(i=0;i<network->output_nodes->size();i++){
		temp=network->output_nodes[0][i].bias;
		fwrite(&temp, sizeof(DTYPE), 1, ptr_myfile);
	}
	fclose(ptr_myfile);
}

void fill_inputs_outputs(std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,std::vector<std::vector<DTYPE>*>* data){
	DWORD i,j;
	for(i=0;i<data->size()-1;i++){
		inputs[0][i]=new std::vector<DTYPE>(data[0][i]->size());
		outputs[0][i]=new std::vector<DTYPE>(data[0][i]->size());
		for(j=0;j<data[0][i]->size();j++){
			inputs[0][i][0][j]=data[0][i][0][j];
			outputs[0][i][0][j]=data[0][i+1][0][j];
		}
	}
}

void fill_inputs_outputs_multiple(std::vector<std::vector<DTYPE>*>* inputs,std::vector<std::vector<DTYPE>*>* outputs,std::vector<std::vector<DTYPE>*>* data,DWORD n_output){
	DWORD i,j,k;
	for(i=0;i<data->size()-n_output;i++){
		inputs[0][i]=new std::vector<DTYPE>(data[0][i]->size());
		outputs[0][i]=new std::vector<DTYPE>(data[0][i]->size()*n_output);
	}
	for(i=0;i<data->size()-n_output;i++){
		for(j=0;j<data[0][i]->size();j++){
			inputs[0][i][0][j]=data[0][i][0][j];
			for(k=0;k<n_output;k++){
				outputs[0][i][0][j*n_output+k]=data[0][i+k+1][0][j];
			}
		}
	}
}

std::vector<std::vector<DTYPE>*>* filter_data(std::vector<std::vector<DTYPE>*>* data,DTYPE lower_bound,DTYPE upper_bound){
	DWORD i;
	std::vector<std::vector<DTYPE>*>* result=new std::vector<std::vector<DTYPE>*>;
	for(i=0;i<data->size()-1;i++){
		if(data[0][i][0][0]>lower_bound&&data[0][i][0][0]<upper_bound){
			result->push_back(data[0][i]);
		}else{
			delete data[0][i];
		}
	}
	delete data;
	return result;
}

std::vector<std::vector<DTYPE>*>* sax_encoding(std::vector<std::vector<DTYPE>*>* data,DWORD intervals){
	DTYPE mean,square,var,temp;
	DTYPE unit_length=1/(DTYPE)intervals;
	//printf("%lf\n",unit_length);
	DWORD i,j,dimension=data[0][0]->size();
	std::vector<std::vector<DTYPE>*>* result=new std::vector<std::vector<DTYPE>*>(data->size());
	for(i=0;i<data->size();i++){
		result[0][i]=new std::vector<DTYPE>(dimension);
	}
	for(j=0;j<dimension;j++){
		mean=0;
		square=0;
		for(i=0;i<data->size();i++){
			mean+=data[0][i][0][j];
			square+=data[0][i][0][j]*data[0][i][0][j];
		}
		var=(square-mean*mean/data->size())/(data->size());
		mean/=data->size();

		//printf("mean=%lf,var=%lf\n",mean,var);
		var=sqrt(var);
		for(i=0;i<data->size();i++){
			if(var==0){
				result[0][i][0][j]=0;
			}else{
				temp=normal_cdf((data[0][i][0][j]-mean)/var);
				//printf("%lf\n",temp);
				temp=(DWORD)(temp/unit_length);
				result[0][i][0][j]=temp*unit_length;
			}
		}
	}
	clean_2d_data(data);
	return result;
}

void clean_2d_data(std::vector<std::vector<DTYPE>*>* data){
	DWORD i;
	for(i=0;i<data->size();i++){
		delete data[0][i];
	}
	delete data;
}

void clean_3d_data(std::vector<std::vector<std::vector<DTYPE>*>*> *data){
	DWORD i,j;
	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			delete data[0][i][0][j];
		}
		delete data[0][i];
	}
	delete data;
}

/*

void print_data(std::vector<std::vector<DTYPE>*>* data){
	DWORD i,j;
	for(i=0;i<data->size();i++){
		for(j=0;j<data[0][i]->size();j++){
			printf("%lf,",data[0][i][0][j]);
		}
		printf("\n");
	}
}


int main(void){
	std::vector<std::vector<DTYPE>*>* data=read_csv("mgh002.csv",MGHMF,TRUE);
	normalize_data(data);
	print_data(data);
}



void print_data(Objects* data){
	DWORD i;
	for(i=0;i<data->size;i++){
		printf("x=%lf,y=%lf,attribute=%lf,time=%lf\n",data->objects[i]->spatial_coordinates[0],data->objects[i]->spatial_coordinates[1],data->objects[i]->attribute,data->objects[i]->time);
	}
}

void main(){
	Objects* data=read_csv("pollution_data.csv",SPATIAL_TEMPORAL_DATA,TRUE);
	print_data(data);
	//printf("%lf\n",parse_time("2011-12-12 22:59:48"));
	//printf("%lf\n",parse_time("2011-12-12 22:52:48"));
}
*/
