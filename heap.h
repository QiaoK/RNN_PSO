#ifndef HEAP_H
#define HEAP_H

#include "neural_network.h"
#define PQ_LENGTH(q) (q->heap->size()-1) 

typedef struct{
	std::vector<Particle*> *heap;
}PriorityQueue;

extern PriorityQueue* create_priority_queue();
extern BOOLEAN is_priority_queue_empty(PriorityQueue *q);
extern void insert_to_priority_queue(PriorityQueue *q,Particle* value);
extern Particle* pop_from_priority_queue(PriorityQueue *q);
extern Particle* top_of_priority_queue(PriorityQueue *q);
extern void destroy_priority_queue(PriorityQueue *q);
#endif
