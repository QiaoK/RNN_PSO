CC=g++
CFLAGS=-c -Wall -Wextra -O3
LIBS = -lm -fopenmp

NETWORK_OBJS = random.o neural_network.o heap.o data_functions.o distributions.o

GRADIENT_TEST_OBJS = random.o neural_network.o gradient_test.o heap.o
HEAP_TEST_OBJS = heap.o heap_test.o
MGH_TEST_OBJS = $(NETWORK_OBJS) mgh_test.o
DATA_TEST_OBJS = $(NETWORK_OBJS) data_functions_test.o
BATCH_TEST_OBJS = $(NETWORK_OBJS) mgh_batch_test.o
SPACE_SHUTTLE_OBJS = $(NETWORK_OBJS) space_shuttle_test.o
POWER_DEMAND_OBJS = $(NETWORK_OBJS) power_demand_test.o
NETWORK_VIEWER_OBJS = $(NETWORK_OBJS) network_viewer.o
SPACE_SHUTTLE_OBJS2 = $(NETWORK_OBJS) space_shuttle_test2.o
ECG_OBJS = $(NETWORK_OBJS) ecg_test.o
ECG_OBJS2 = $(NETWORK_OBJS) ecg_test2.o

All: mgh_test space_shuttle_test space_shuttle_test2 ecg_test ecg_test2
gradient_test : $(GRADIENT_TEST_OBJS)
	$(CC) -o $@ $(GRADIENT_TEST_OBJS) $(LIBS)
mgh_test : $(MGH_TEST_OBJS)
	$(CC) -o $@ $(MGH_TEST_OBJS) $(LIBS)
heap_test : $(HEAP_TEST_OBJS)
	$(CC) -o $@ $(HEAP_TEST_OBJS) $(LIBS)
data_test : $(DATA_TEST_OBJS)
	$(CC) -o $@ $(DATA_TEST_OBJS) $(LIBS)
batch_test : $(BATCH_TEST_OBJS)
	$(CC) -o $@ $(BATCH_TEST_OBJS) $(LIBS)
space_shuttle_test : $(SPACE_SHUTTLE_OBJS)
	$(CC) -o $@ $(SPACE_SHUTTLE_OBJS) $(LIBS)
power_demand_test : $(POWER_DEMAND_OBJS)
	$(CC) -o $@ $(POWER_DEMAND_OBJS) $(LIBS)
network_viewer : $(NETWORK_VIEWER_OBJS)
	$(CC) -o $@ $(NETWORK_VIEWER_OBJS) $(LIBS)
space_shuttle_test2 : $(SPACE_SHUTTLE_OBJS2)
	$(CC) -o $@ $(SPACE_SHUTTLE_OBJS2) $(LIBS)
ecg_test : $(ECG_OBJS)
	$(CC) -o $@ $(ECG_OBJS) $(LIBS)
ecg_test2 : $(ECG_OBJS2)
	$(CC) -o $@ $(ECG_OBJS2) $(LIBS)



%.o: %.cpp
	$(CC) $(CFLAGS) -c $(LIBS) $<
clean:
	rm -rf *.o
	rm -rf *.stackdump
	rm -rf network_test
	rm -rf network_test.exe
	rm -rf gradient_test
	rm -rf gradien_test.exe
	rm -rf mgh_test
	rm -rf mgh_test.exe
	rm -rf heap_test
	rm -rf heap_test.exe
	rm -rf data_test
	rm -rf data_test.exe
	rm -rf batch_test
	rm -rf batch_test.exe
	rm -rf space_shuttle_test
	rm -rf space_shuttle_test.exe
	rm -rf power_demand_test
	rm -rf power_demand_test.exe
	rm -rf network_viewer
	rm -rf network_viewer.exe
	rm -rf space_shuttle_test2
	rm -rf space_shuttle_test2.exe
	rm -rf ecg_test
	rm -rf ecg_test.exe
	rm -rf ecg_test2
	rm -rf ecg_test2.exe
	rm -rf stock_test
	rm -rf stock_test.exe
	rm -rf stock_predict
	rm -rf stock_predict.exe
