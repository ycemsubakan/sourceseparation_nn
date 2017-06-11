import numpy as np
import tensorflow as tf
import time
import pdb 
import os 
import socket
import multiprocessing
from multiprocessing import Process
from multiprocessing.pool import Pool
from rnns import *
from music_utilities import *

def model_driver(d,data):
    """This function builds the computation graph for the specified model
    The input is the dictionary d with fields:
        model (model to be learnt), wform (form of the W matrices - full/diagonal/scalar/constant) 
        K (number of states), L1 (input dimensions), L2 (output dimensions), numlayers (number of layers)
    """
    # Reset graph
    tf.reset_default_graph()
    
    # device configuration
    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    if d['task'] == 'source_sep':
        
        #build graph 1
        with tf.variable_scope("model1"):
            rnn1 = rnn( model_specs = d, initializer = d['init'], model_num = 1)
            rnn1_handles = rnn1.build_graph()  
        
        with tf.Session(config = config) as sess:
            sess.run(tf.initialize_all_variables())
            all_times, tr_logls, _, _ = rnn1.optimizer(data = data['Train1'], 
                    rnn_handles = rnn1_handles, sess = sess)
            
            vars_np1 = rnn1.save_modelvars_np(sess)

        tf.reset_default_graph()

        #build graph 2
        with tf.variable_scope("model2"):
            rnn2 = rnn( model_specs = d, initializer = d['init'], model_num = 2)
            rnn2_handles = rnn2.build_graph()  

        with tf.Session(config = config) as sess:
            sess.run(tf.initialize_all_variables())
            all_times, tr_logls, _, _ = rnn2.optimizer(data = data['Train2'], 
                    rnn_handles = rnn2_handles, sess = sess) 

            vars_np2 = rnn2.save_modelvars_np(sess)

        tf.reset_default_graph()

        all_vars_np = vars_np1 + vars_np2

        rnn_sep_opt = rnn( model_specs = d, initializer = all_vars_np, input_opt = True)
        rnn_sep_opt_handles = rnn_sep_opt.build_separation_graph()

        with tf.Session(config = config) as sess:
            sess.run(tf.initialize_all_variables())
        
            x1hat, x2hat = rnn_sep_opt.input_optimizer(data = data['Test'], 
                                        rnn_handles = rnn_sep_opt_handles, sess = sess) 

        Z = d['audio_files']
        mixture = np.concatenate(data['Test']['data'].values, axis = 1)    
        phase = np.concatenate(data['Test']['phase'].values, axis = 1)    
        
        x1hat = x1hat[:,:mixture.shape[1]]
        x2hat = x2hat[:,:mixture.shape[1]]
        eps = 1e-20

        o1 = d['FE'].ife( x1hat * (mixture/(x1hat+x2hat+eps)), phase)
        o2 = d['FE'].ife( x2hat * (mixture/(x1hat+x2hat+eps)), phase)

        sxr = bss_eval( o1, 0, vstack( (Z[2],Z[3]))) + bss_eval( o2, 1, vstack( (Z[2],Z[3])))
        
        print('BSS eval values are: ',str(sxr))

        res_dictionary = {'valid':  None, 
                      'tst':  np.array(sxr), 
                      'tr': np.array(tr_logls), 
                      'all_times':all_times, 
                      'tnparams':rnn1.tnparams}

        res_dictionary.update(d)

        return res_dictionary
    
    elif d['task'] == 'toy_example': 
        
        with tf.variable_scope("model1"):
            rnn1 = rnn( model_specs = d, initializer = d['init'], model_num = 1)
            rnn1_handles = rnn1.build_graph()  
        
        with tf.Session(config = config) as sess:
            sess.run(tf.initialize_all_variables())
            all_times, tr_logls, _, _ = rnn1.optimizer(data = data, 
                    rnn_handles = rnn1_handles, sess = sess)


def main(dictionary):
    #set the model categories
    model_categories = {'one_basis_rnns': ['ob_mod_lstm','ob_gru'],
                        'mult_basis_rnns' : ['mb_mod_lstm','mb_mod_lstm'],
                        'ff_and_conv': ['feed_forward','convolutive']}
    dictionary.update(model_categories)

    # next thing is determining the hyperparameters
    # np.random.seed( dictionary['seedin'][0] )
    tf.set_random_seed( dictionary['seedin'][1] ) 
    timestamp = str(round(time.time()))

    # lower and upper limits for hyper-parameters to be sampled
    lr_min ,lr_max = dictionary['lr_min'], dictionary['lr_max']
    num_layers_min, num_layers_max = dictionary['num_layers_min'], dictionary['num_layers_max']
    K_min, K_max, min_params, max_params = return_Klimits(
            model = dictionary['encoder'], 
            wform = dictionary['wform'], 
            data = dictionary['data']) 

    records = [] #information will accumulate in this
    for i in range(dictionary['num_configs']):
        # np.random.seed(seed = 1234)
        # first get the data and the resulting model parameters  
        # data, parameters = load_data(dictionary = dictionary)
        # dictionary.update(parameters) # add the necessary model parameters to the dictionary here
        # pdb.set_trace()
        while True:  
            try:
                # pdb.set_trace()
                lr, K, num_layers, momentum, set_seed = generate_random_hyperparams(
                        lr_min =  lr_min, lr_max = lr_max, 
                        K_min = K_min, K_max = K_max , 
                        num_layers_min = num_layers_min, 
                        num_layers_max = num_layers_max,
                        load_hparams = (dictionary['load_hparams'],i),
                        num_configs = dictionary['num_configs'])
                
                if set_seed:
                    np.random.seed(1234)
                
                data, parameters = load_data(dictionary = dictionary)
                dictionary.update(parameters) # add the necessary model parameters to the dictionary here
                
                    
                print("Configuration ",i,
                      "K = ",K, 
                      "num_layers = ", num_layers,
                      "Learning Rate = ", lr,
                      "Momentum = ", momentum)  
                #this if clause enables the user to restart an experiment from a specific point the experiment
                if i < dictionary['start']:
                    break
                
                try: # Sometimes resources may get exhausted, this exception handles that 
                    dictionary.update({'LR': lr, 
                                        'K': K,
                                        'K_in': 9, # include this in search later
                                        'num_layers': num_layers,
                                        'min_params': min_params,
                                        'max_params': max_params,
                                        'momentum' : momentum} ) 
                    run_info = model_driver(d = dictionary, data = data) 
                    
                    #remove the unnecessary big fields
                    run_info.pop('FE')
                    run_info.pop('audio_files')

                    #append the performance records
                    records.append(run_info)
                except KeyboardInterrupt:
                    raise 
                except:
                    print('Resouces exhausted for this configuration, moving on')
                    raise
                break
        
            except num_paramsError:
                print('This parameter configuration is not valid!!!!!!!') 

        # Save in directory
        savedir = 'experiment_data/'
        np.save( savedir + dictionary['server'] 
                + '_data_' + dictionary['data'] 
                + '_encoder_' + dictionary['encoder']
                + '_decoder_' + dictionary['decoder']
                + '_'+ dictionary['wform_global'] 
                + '_optimizer_' + dictionary['optimizer']
                +'_device_'+ dictionary['device']
                + '_' + timestamp, records)
    
    return records


#import matplotlib.pyplot as plt
wform = 'block_diagonal'# either diagonal, block_diagonal or full
input_dictionary = {'seedin' : [1144, 1521], #setting the random seed. First is for numpy, second is for tensorflow 
            'task' : 'source_sep', #this helps us how to load the data with the load_data function in rnns.py 
            'data' : 'timit', #the dataset, options are inside the load_data function 
            'encoder': 'mb_mod_lstm', #options are: feed_forward, convolutive, mb_mod_lstm
            'decoder': 'convolutive',
            'wform' : wform, 
            'wform_global' : wform,
            'num_configs' : 60, #number of hyper parameter configurations to be tried 
            'start' : 0,  #this is used to start from a certain point (can be useful with fixed seed, or when hyper-parameters are loaded) 
            'EP' : 2000, #number of epochs per run 
            'dropout' : [1, 1], #first is the input second is the output keep probability 
            'device' : 'gpu:0', #the device to be used in the computations 
            'server': socket.gethostname(),
            'verbose': True, #this prints out the batch location
            'load_hparams': True, #this loads hyper-parameters from a results file
            'count_mode': False, #if this is True, the code will stop after printing the number of trainable parameters
            'init':'xavier', #initialization method, options are 'xavier','random_unform' 
            'lr_min':-4, 'lr_max':-2, #the lower and upper limits for the exponent of the learning rate
            'num_layers_min':1, 'num_layers_max':1, #lower and upper limits for number of layers
            'ntaps':8, #filter length for convolutive model
            'optimizer':'RMSProp', #options are, Adam, RMSProp, Adadelta
            'activation':'softplus',
            'separation_method':'complete',
            'notes':''} 

perfs = main(input_dictionary)
