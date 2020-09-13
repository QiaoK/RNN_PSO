#include "heap.h"
#include <stdlib.h>
#include <stdio.h>

PriorityQueue* create_priority_queue(){
	PriorityQueue *result=Calloc(1,PriorityQueue);
	result->heap=new std::vector<Particle*>;
	result->heap->resize(1);
	return result;
}

BOOLEAN is_priority_queue_empty(PriorityQueue *q){
	return q->heap->size()==1;
}

void insert_to_priority_queue(PriorityQueue *q,Particle* value){
	//insertion
	q->heap->push_back(value);
	//relocation
	DWORD loc=q->heap->size()-1,parent;
	Particle* temp;
	parent=loc/2;
	while(loc!=1&&q->heap[0][parent]->current_mse<q->heap[0][loc]->current_mse){
		temp=q->heap[0][parent];
		q->heap[0][parent]=q->heap[0][loc];
		q->heap[0][loc]=temp;
		loc=parent;
		parent=loc/2;;
	}
}

Particle* top_of_priority_queue(PriorityQueue *q){
	if(!is_priority_queue_empty(q)){
		return q->heap[0][1];
	}else{
		return NULL;
	}
}

Particle* pop_from_priority_queue(PriorityQueue *q){
	//swap with last element
	DWORD last=q->heap->size()-1;
	Particle* result=q->heap[0][1];
	q->heap[0][1]=q->heap[0][last];
	q->heap->pop_back();
	DWORD next=2,loc=1;
	Particle* temp;
	while((next<last&&q->heap[0][next]->current_mse>q->heap[0][loc]->current_mse)||(next+1<last&&q->heap[0][next+1]->current_mse>q->heap[0][loc]->current_mse)){
		if(next+1<last&&q->heap[0][next]->current_mse<q->heap[0][next+1]->current_mse){
			next++;
		}
		temp=q->heap[0][next];
		q->heap[0][next]=q->heap[0][loc];
		q->heap[0][loc]=temp;
		loc=next;
		next=loc*2;
	}


	return result;
}

void destroy_priority_queue(PriorityQueue *q){
	delete q->heap;
	Free(q);
}
