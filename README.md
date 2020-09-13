This software implements the training algorithms for recurrent neural network proposed in paper https://ieeexplore.ieee.org/abstract/document/8215744.
Data cited in this paper is also provided.

# System requirement

gcc compiler that supports C99.

# Compile

Use make file to compile code and test cases with command "make".

# Executables

All executables have built in inputs without arguments. The output of the programs are training curves of methods used in the paper. To reproduce the comparisons in the paper, you need to modify the function "update_network_by_gradient" in https://github.com/QiaoK/RNN_PSO/blob/master/neural_network.cpp. The previous RNN training method do not have the loop commented with "gradient within hidden node".

- ./ecg_test
  Running QT ECG data mentioned in the paper
- ./ecg_test2
  A longer version of QT ECG data.
- ./space_shuttle_test
 Puppet withdraw data mentioned in the paper.
- ./space_shuttle_test2
 Puppet pulled out data mentioned in the paper.
- ./mgh_test
 MGHMF record 003 used in the paper.
# Contact

Contact me at qiao.kang@eecs.northwestern.edu if you have questions.
