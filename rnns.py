# This file has the rnn functions and classes  
import numpy as np
import tensorflow as tf
import pdb
import time
import pandas as pd
import sklearn.feature_extraction.text as ft
import _pickle as pickle
import itertools
import sys
import os
import matplotlib.pyplot as plt

class rnn(object):
    """This class has the build_graph function  that builds the rnn computation graph, and the optimizer that optimizes the model parameters given the graph handle and the data""" 
    def __init__(self, model_specs, initializer = 'xavier'):
        'model specs is a dictionary'
        self.model_specs = model_specs
        self.initializer = initializer
    
    def build_graph(self): 
        'this function builds a graph with the specifications in self.model_specs'
        
        d = self.model_specs #unpack the model specifications
        with tf.device('/' + d['device']):
            x = tf.placeholder(tf.float32, [None, d['batchsize'], d['L1']],"x")
            mask = tf.placeholder(tf.float32, [None])
            y = tf.placeholder(tf.float32, [None, d['L1']],"y") 

            dropout_kps = tf.placeholder(tf.float32, [2], "dropout_params")
            seq_lens = tf.placeholder(tf.int32, [None])
            
            hhat = self.define_model(x, seqlens = seq_lens, dropout_kps = dropout_kps)

            with tf.variable_scope("decoder"):
                V_initializer = b_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                
                V = tf.get_variable("V", dtype= tf.float32, 
                        shape = [d['K'], d['L2']], initializer = V_initializer)  
                b = tf.get_variable("b", dtype= tf.float32, 
                        shape = [d['L2']], initializer = b_initializer)  

            yhat = tf.matmul(hhat,V) + tf.reshape(b, (1, d['L2']))

            #compute the number of parameters to be trained
            tvars = tf.trainable_variables()
            tnparams = np.sum([np.prod(var.get_shape().as_list()) for var in tvars])
            saver = tf.train.Saver(tvars) 

            if d['count_mode']:
                print('The total number of parameters to be trained =', tnparams,
                      'The model is: ', d['model'])
                os._exit(0)

            #raise an error if we are outside the allowed range
            if d['min_params'] > tnparams or d['max_params'] < tnparams:
                raise num_paramsError
            else:
                self.tnparams = tnparams
                self.tvars_names = [var.name for var in tvars] 

            #define the cost         
            cost = tf.reduce_sum(tf.square(y - yhat))

            #define the optimizer
            if d['optimizer'] == 'Adam':
                train_step = tf.train.AdamOptimizer(d['LR']).minimize(cost)
            elif d['optimizer'] == 'RMSProp':
                train_step = tf.train.RMSPropOptimizer(d['LR'], 
                                    momentum = d['momentum'],
                                    centered = True).minimize(cost)   
            elif d['optimizer'] == 'Adadelta':
                train_step = tf.train.AdadeltaOptimizer(d['LR']).minimize(cost)   

            if d['task'] == 'source_sep':
                relevant_inds = tf.squeeze(tf.where(tf.cast(mask,tf.bool)))
                preds = tf.gather(yhat,relevant_inds) 
                targets = tf.gather(y,relevant_inds) 
            
            #return the graph handles 
            graph_handles = {'train_step':train_step,
                             'x':x,
                             'y':y,
                             'h':hhat,
                             'mask':mask,
                             'cost':cost,
                             'dropout_kps':dropout_kps,
                             'seq_lens':seq_lens,
                             'saver':saver,
                             'preds':preds,
                             'relevant_inds':relevant_inds,
                             'targets':targets,
                             }
                                           
                             
            return graph_handles

    def define_model(self, x, seqlens ,dropout_kps = tf.constant([1,1])):  
        p1 = dropout_kps[0]
        p2 = dropout_kps[1]

        onedir_rnns = ['ss_lstm','ss_gru']

        # unpack model specifications 
        d = self.model_specs
        wform, model, K, num_layers, mapping_mode, L1, L2 = d['wform'], d['model'], d['K'], d['num_layers'], d['mapping_mode'], d['L1'], d['L2']

        if model in onedir_rnns:

            if model == 'ss_lstm': 
                cell = tf.contrib.rnn.BasicLSTMCell(K, forget_bias=1.0)
            elif model == 'ss_gru':
                cell = tf.contrib.rnn.GRUCell(K )  

            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p1)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=p2)

            outputs, _= tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, 
                    time_major = True, sequence_length = seqlens )
            
            outputs = tf.transpose(outputs, [1,0,2] ) 
            outputs = tf.unstack(outputs,axis = 0)
            outputs = tf.concat(outputs, axis = 0)

            return outputs
        
        elif model == 'feed_forward':

            with tf.variable_scope('encoder'):
                V_initializer = b_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                
                V = tf.get_variable("V", dtype= tf.float32, 
                        shape = [d['L1'], d['K']], initializer = V_initializer)  
                b = tf.get_variable("b", dtype= tf.float32, 
                        shape = [d['K']], initializer = b_initializer)  

            x = tf.transpose(x, [1, 0, 2])
            x = tf.unstack(x, axis = 0)
            x = tf.concat(x, axis = 0)

            outputs = ( tf.matmul(x,V) + b ) 

            return outputs

    def optimizer(self, data, rnn_handles, sess, model_n = 1):
        """This function runs the optimizer for the given data and given rnn graph referenced by rnn_handles """

        d = self.model_specs # unpack the variables 

        tr = SimpleDataIterator(data['Train'], num_buckets = d['num_buckets'])
        #tst = SimpleDataIterator(data['Test'])
        #valid = SimpleDataIterator(data['Validation'])

        all_times, tr_logls, test_logls, valid_logls = [], [], [], [] 
        for ep in range(d['EP']):
            t1, tr_logl = time.time(), []
            while tr.epochs == ep:
                trb = tr.next_batch(
                        n = d['batchsize'], 
                        task = d['task'], 
                        verbose = d['verbose'])      

                feed = {rnn_handles['x']:trb[0], 
                        rnn_handles['y']:trb[1], 
                        rnn_handles['mask']:trb[2],
                        rnn_handles['seq_lens']:trb[3], 
                        rnn_handles['dropout_kps']:d['dropout'] }  

                tr_cost,_ = sess.run( 
                        [rnn_handles['cost'], rnn_handles['train_step']], feed) 
                
                if d['verbose']:
                    print("Training cost = ", tr_cost) 
            t2 = time.time()
            #tr_logl = np.mean(tr_logl)

            tst_logl = 0
            logls_len_total = 0
            # while tst.epochs == ep:
            #     tsb = tst.next_batch( n = d['batchsize'], task = d['task'], 
            #             verbose = d['verbose'])  
            #     
            #     tst_feed = {rnn_handles['x']: tsb[0], 
            #         rnn_handles['y']: tsb[1], 
            #         rnn_handles['mask']:tsb[2],
            #         rnn_handles['seq_lens']: tsb[3], 
            #         rnn_handles['dropout_kps']:np.array([1,1])} 

            #     logls = sess.run( rnn_handles['accuracy_nn'], tst_feed ) 
            #     tst_logl = tst_logl + logls.sum()
            #     logls_len_total = logls_len_total + logls.shape[0]

            # tst_logl = tst_logl / logls_len_total

            # vld_logl = 0 
            # logls_len_total = 0
            # while valid.epochs == ep:
            #     vlb = valid.next_batch( n = d['batchsize'], task = d['task'], 
            #             verbose=d['verbose'])  
            #                 
            #     vld_feed = {rnn_handles['x']: vlb[0], 
            #             rnn_handles['y']: vlb[1], 
            #             rnn_handles['mask']: vlb[2],
            #             rnn_handles['seq_lens']: vlb[3], 
            #             rnn_handles['dropout_kps']:np.array([1,1])} 
       
            #     logls = sess.run( rnn_handles['accuracy_nn'], vld_feed ) 
            #     vld_logl = vld_logl + logls.sum() 
            #     logls_len_total = logls_len_total + logls.shape[0]

            # vld_logl = vld_logl / logls_len_total
    
            print("The Model is ",d['model'],d['wform'],
                  "Optimizer is ",d['optimizer'],
                  " ,Iteration = ", ep, 
                  #" ,Training Accuracy", np.mean(tr_logl),
                  #",Test Accuracy = ", tst_logl, 
                  #",Validation Accuracy = ", vld_logl, 
                  ",Elapsed Time = ", t2-t1) 
            
            tst_logl = None
            vld_logl = None

            all_times.append(t2-t1)
            tr_logls.append(tr_logl)
            test_logls.append(tst_logl)
            valid_logls.append(vld_logl)

        Hhat, Yhat = sess.run([rnn_handles['h'], rnn_handles['preds']], feed) 
        pdb.set_trace()

        return all_times, tr_logls, test_logls, valid_logls

def return_Klimits(model, wform, data):
    """We use this function to select the upper and lower limits of number of 
    hidden units per layer depending on the task and the dataset. The user can also choo    se to limit the upper and lower limit of allowable number of trainable parameters"""

    if model in ['mod_lstm', 'lstm']:
        min_params = 1e1; max_params =  7e7 # in our waspaa paper we basically did not use lower and upper bounds for number of parameters
        K_min, K_max = 30, 350

    elif model in ['ss_lstm']:
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 3, 3

    elif model == 'feed_forward':
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 5, 5

    elif model == 'tanh_lds':
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 50, 400
    
    elif model == 'vector_w_conv':
        min_params = 1e1; max_params = 7e7
        K_min, K_max = 300, 600

    return K_min, K_max, min_params, max_params 

def generate_random_hyperparams(lr_min, lr_max, K_min, K_max, num_layers_min, num_layers_max,load_hparams):
    """This function generates random hyper-parameters for hyperparameter search"""

    #this is for new random parameters
    if not load_hparams[0]:
        lr_exp = np.random.uniform(lr_min, lr_max)
        lr = 10**(lr_exp)
        K = np.random.choice(np.arange(K_min, K_max+1),1)[0]
        num_layers = np.random.choice(np.arange(num_layers_min, num_layers_max + 1),1)[0]
        #momentum_exp = np.random.uniform(-8,0) 
        momentum = np.random.uniform(0,1) #(2**momentum_exp)

    #this loads hyperparameters from an existing file
    else:
        exp_data = np.load('experiment_data/nmf_data_timit_model_bi_mod_lstm_diag_to_full_device_cpu:0_1490813245.npy')[load_hparams[1]]
        lr = exp_data['LR']
        K = exp_data['K']
        num_layers = exp_data['num_layers']
        try:
            momentum = exp_data['num_layers']
        except:
            momentum = None

    return lr, K, num_layers, momentum

def load_data(dictionary):
    """this function loads the data, and sets the associated parameters (such as output and input dimensionality and batchsize) according to the specified task, which are either text, music, speech or digits """
    task, data = dictionary['task'], dictionary['data']

    if task == 'source_sep':
        L, T, step = 150, 200, 50  
       
        data = [generate_separationdata(L,T,step)]

        d = {'data':data, 'lengths': [T]}
        df_train = pd.DataFrame( d )
        
        df_test = df_valid = None

        batchsize = 1
        L1 = L 
        L2 = L
        outstage = 'relu'
        mapping_mode = 'seq2seq'
        num_steps = 200
        iterator = 'SimpleDataIterator'
        num_buckets = None
        len_th = None

    parameters = {'batchsize':batchsize,
                  'L1':L1,
                  'L2':L2,
                  'outstage':outstage,
                  'mapping_mode':mapping_mode,
                  'num_steps':num_steps,
                  'iterator':iterator,
                  'num_buckets':num_buckets,
                  'len_th':len_th}

    return {'Train':df_train, 'Test':df_test, 'Validation':df_valid}, parameters

class SimpleDataIterator():
    """
    This class is adapted (ripped off) from r2rt.com, in this version, the next_batch function uses a pandas dataframe, and outputs the batch in a reshaped format ready to input to the tensorflow function 
    """
    def __init__(self, df, num_buckets = None):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n, task, verbose = False):
        if verbose:
            print("The current cursor points to ",self.cursor," Data size is",self.size)

        part = self.df.ix[self.cursor:self.cursor+n-1]
        
        if task == 'digits': 
            #this part needs to be updated
            temp = list(part['data'].values)
            data = np.transpose(np.asarray(temp),[2,0,1])  
            labels = part['labels'].values
            mask = np.ones(labels.shape) 
            lengths = part['lengths'].values
        elif task in ['music','text']:
            max_len = np.max(part['lengths'].values)
            L = part['data'].values[0].shape[0] 
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len-1))
            labels = [np.zeros((max_len-1, L)) for i in range(n)] # better name is 'targets' 
            data = np.zeros((L, max_len -1, n)) 
            iterables = zip(part['lengths'].values, part['data'].values)
            for i,vals in enumerate(iterables):
                ln = vals[0] - 1
                lengths[i] = ln
                mask[i,0:ln] = 1
                labels[i][0:ln,:] = vals[1][:,1:].transpose()
                data[:,0:ln,i] = vals[1][:,:-1]

            #finally reshape things
            mask = mask.reshape(-1)
            labels = np.concatenate(labels, axis = 0)
            data = np.transpose(data, [1,2,0]) 
        if task == 'source_sep':
            max_len = np.max(part['lengths'].values)
            L = part['data'].values[0].shape[0] 
           
            lengths = np.zeros(n) 
            mask = np.zeros((n,max_len))
            labels = [np.zeros((max_len, L)) for i in range(n)]

            data = np.zeros((L, max_len, n)) 
            iterables = zip(part['lengths'].values, part['data'].values)
            for i,vals in enumerate(iterables):
                ln = vals[0]
                lengths[i] = ln
                mask[i,0:ln] = 1
                labels[i][0:ln,:] = vals[1].transpose()
                data[:,0:ln,i] = vals[1]
            
            #finally reshape things
            mask = mask.reshape(-1)
            labels = np.concatenate(labels, axis = 0)
            data = np.transpose(data, [1,2,0]) 

        if self.cursor+n >= self.size:
            self.epochs += 1
            self.shuffle()
        else:
            self.cursor += n

        return data, labels, mask, lengths

def generate_separationdata(L,T,step):
    
    x = np.zeros((L,T))
    cursor = t = 0
    while t<T:
        x[cursor, t] = 1
        t  = t + 1 
        cursor = np.mod( cursor + step, L)    

    return x

class Error(Exception):
    pass

class num_paramsError(Error):
    pass




