# Software requirements: 
* Python v > 3.5
* Numpy v > 1.12 
* Tensorflow v > 1.0 
* Pandas > v -0.18
* scikit-learn v > 0.17.1 (for feature extraction in text task) 


## Summary of the input and outputs of the code: 
* The specifications for the particular run, including the task, dataset, model, form of the recurrent matrices, the device to run on, initialization, optimizer are given inside the dictionary named 'input_dictionary' as an argument to the main function. The outputs are written in a .npy file after being done with each random hyper-parameter configuration. 


## Loading custom dataset:  
* The function 'load_data()' in rnns.py handles the data loading. The users can customize this function accordingly.    
