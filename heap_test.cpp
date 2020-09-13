#include "heap.h"

int main(){
	PriorityQueue *q=create_priority_queue();
	Particle *particles=Calloc(8,Particle);
	particles[0].current_mse=5;
	particles[1].current_mse=4;
	particles[2].current_mse=1;
	particles[3].current_mse=6;
	particles[4].current_mse=3;
	particles[5].current_mse=8;
	particles[6].current_mse=2;
	particles[7].current_mse=4;

	DWORD i;
	for(i=0;i<8;i++){
		insert_to_priority_queue(q,&particles[i]);
	}
	while(!is_priority_queue_empty(q)){
		printf("%lf\n",pop_from_priority_queue(q)->current_mse);
	}
}
