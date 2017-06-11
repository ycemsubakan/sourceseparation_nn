# This file has the rnn functions and classes  
import numpy as np
import tensorflow as tf
import pdb
import time
import pandas as pd
import sklearn.feature_extraction.text as ft
#import _pickle as pickle
import itertools
import sys
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from music_utilities import *


class ModLSTMCell(tf.contrib.rnn.RNNCell):
    """Modified LSTM Cell """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'diagonal', input_opt = False, model_num = None, num_rnns = 1):
        self._num_units = num_units
        self._num_rnns = num_rnns
        self.init = initializer
        self.wform = wform 
        self.input_opt = input_opt
        self.model_num = model_num

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units*self._num_rnns, 
                                             self._num_units*self._num_rnns)

    @property
    def output_size(self):
        return self._num_units*self._num_rnns

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            
            c, h = state
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]
            
            if self.wform == 'full':
                mats, biases = self.get_params_parallel()
            elif self.wform == 'block_diagonal':
                Ws, Us, biases = self.get_params_parallel()
            
            if self.wform == 'full':                 
                res = tf.matmul(tf.concat([h,inputs],axis=1),mats)
                res_wbiases = tf.nn.bias_add(res, biases)
           
                i,j,f,o = tf.split(res_wbiases, num_or_size_splits=4, axis=1) 
            elif self.wform == 'block_diagonal':
                h = tf.split(h, num_or_size_splits = int(self._num_rnns), 
                            axis=1)  
                c = tf.split(c, num_or_size_splits = int(self._num_rnns), 
                            axis=1)  

                h, c = tf.stack(h, axis = 2), tf.stack(c, axis = 2)

                Wres = tf.einsum('bik,ijk->bjk', h, Ws) 
                Ures = tf.einsum('bl,ljk->bjk', inputs, Us)

                res = Wres + Ures
                #biases = tf.tile(biases, [1, inputs.get_shape().as_list()[0]])
                #biases = tf.transpose(biases, [2, 0, 1])
                res_wbiases = res + biases

                i, j, f, o = tf.split(res_wbiases, num_or_size_splits=4, axis=1) 
            elif self.wform == 'diagonal':
                h_concat = tf.concat([h,h,h,h],axis=1)

                W_res = tf.multiply(h_concat,mats[0])

                U_res = tf.matmul(inputs,mats[1])

                res = tf.add(W_res,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases, num_or_size_splits=4, axis=1) 
                         
            new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i)*tf.nn.tanh(j))
            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)

            
            new_c = tf.split(new_c, num_or_size_splits=int(self._num_rnns)
                            ,axis=2)
            new_c = tf.squeeze(tf.concat(new_c, axis=1))

            
            new_h = tf.split(new_h, num_or_size_splits=int(self._num_rnns)
                            ,axis=2)
            new_h = tf.squeeze(tf.concat(new_h, axis=1))
            
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state

    def get_params_parallel(self):
        if self.input_opt:
            if self.wform == 'full':
                
                var_scope = 'model' + str(self.model_num) + '/' + tf.get_variable_scope().name
                #.name.replace('rnn2/','')
                
                #first filtering 
                vars_to_use = [var for var in self.init if var_scope in var[0]]  

                #next, assign the variables   
                for var in vars_to_use:
                    if '/mats' in var[0]:
                        mats = tf.constant( var[1] )               

                    elif '/biases' in var[0]:
                        biases = tf.constant( var[1] )
            
                return mats, biases

            elif self.wform == 'block_diagonal':           

                var_scope = 'model' + str(self.model_num) + '/' + tf.get_variable_scope().name
                #.name.replace('rnn2/','')
                
                #first filtering 
                vars_to_use = [var for var in self.init if var_scope in var[0]]  

                for var in vars_to_use:
                    if '/Ws' in var[0]:
                        Ws = tf.constant( var[1] )               

                    elif '/Us' in var[0]:
                        Us = tf.constant( var[1] )

                    elif '/biases' in var[0]:
                        biases = tf.constant( var[1] )

                return Ws, Us, biases

        else:
            if self.wform == 'full':
                mats = tf.get_variable("mats", 
                        shape = [self._num_units+self.L1, self._num_units*4], 
                        initializer = self.init )   
                biases = tf.get_variable("biases", 
                        shape = [self._num_units*4], 
                        initializer = self.init )   
            
                return mats, biases

            elif self.wform == 'block_diagonal':
                Ws = tf.get_variable("Ws",
                        shape = [self._num_units, self._num_units*4, self._num_rnns], 
                        initializer = self.init)
                Us = tf.get_variable("Us",
                        shape = [self.L1, self._num_units*4, self._num_rnns], 
                        initializer = self.init)

                biases = tf.get_variable("biases", 
                        shape = [self._num_units*4, self._num_rnns], 
                        initializer = self.init )   

                return Ws, Us, biases

class rnn(object):
    """This class has the build_graph function  that builds the rnn computation graph, and the optimizer that optimizes the model parameters given the graph handle and the data""" 
    def __init__(self, model_specs, 
            initializer = 'xavier', input_opt = False, model_num = None):
        'model specs is a dictionary'
        self.model_specs = model_specs
        self.initializer = initializer
        self.input_opt = input_opt
        self.model_num = model_num

    def save_modelvars_np(self, sess):
        """ This function saves the variables in diagonal to full transition """ 

        variables = tf.trainable_variables()
        vars_np = [(var.name,sess.run(var)) for var in variables]

        return vars_np 

    def build_graph(self): 
        'this function builds a graph with the specifications in self.model_specs'
        
        d = self.model_specs #unpack the model specifications
        
        if d['activation'] == 'relu':
            act = tf.nn.relu
        elif d['activation'] == 'softplus':
            act = tf.nn.softplus

        with tf.device('/' + d['device']):
            x = tf.placeholder(tf.float32, [None, d['batchsize'+str(self.model_num)], d['L1']],"x")
            mask = tf.placeholder(tf.float32, [None])
            y = tf.placeholder(tf.float32, [None, d['L1']],"y") 

            dropout_kps = tf.placeholder(tf.float32, [2], "dropout_params")
            seq_lens = tf.placeholder(tf.int32, [None])
            
            #encoder 
            hhat = self.define_encoder(x, seqlens = seq_lens, dropout_kps = dropout_kps)
            hhat = act(hhat) 

            #decoder
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
            if d['decoder'] == 'feed_forward':
                with tf.variable_scope("decoder"):
                   
                    V = tf.get_variable("V", dtype= tf.float32, 
                            shape = [d['K'], d['L2']], initializer = initializer)  
                    b = tf.get_variable("b", dtype= tf.float32, 
                            shape = [d['L2']], initializer = initializer)  

                yhat = act(tf.matmul(hhat,V) + tf.reshape(b, (1, d['L2'])))

            elif d['decoder'] == 'convolutive':
                with tf.variable_scope("decoder"):
                    
                    fltr = tf.get_variable("fltr", dtype = tf.float32,
                        shape = [d['ntaps'], 1, d['K'], d['L2'] ], initializer = initializer)
                    b = tf.get_variable("b", dtype = tf.float32,
                        shape = [d['L2']], initializer = initializer)

                    hhat_rev = tf.reverse( hhat, axis = [0])
                    hhat_pad = tf.concat([hhat_rev, tf.zeros([d['ntaps']-1,d['K']])],
                                        axis = 0)
                    hhat_reshape = tf.reshape( hhat_pad, [1, -1, 1, d['K']])
                    
                    yhat = tf.nn.conv2d( hhat_reshape, filter =  fltr,
                                         strides = [1,1,1,1], padding = "VALID")
                    yhat = tf.reshape( yhat, [-1,d['L2']])
                    yhat = tf.reverse(yhat, axis = [0])
                    yhat = act(yhat + tf.reshape(b, (1, d['L2'])))
           
            
            elif d['decoder'] in d['mult_basis_rnns']:
                with tf.variable_scope("decoder"):
                    #K_encoder = hhat.get_shape().as_list()[-1]

                    if d['decoder'] == 'mb_mod_lstm': 
                        cell = ModLSTMCell(d['K']*d['K_in'], initializer = initializer,
                                input_opt = self.input_opt, wform = d['wform'])   
                    elif d['decoder'] == 'mb_gru':
                        cell = tf.contrib.rnn.GRUCell(d['K']*d['K_in'])  

                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=d['dropout'][0])
                    cell = tf.contrib.rnn.MultiRNNCell([cell] * d['num_layers'])
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=d['dropout'][1])
                
                    outputs, _= tf.nn.dynamic_rnn(cell, hhat, dtype=tf.float32, 
                        time_major = True, sequence_length = seq_lens )

                    outputs = tf.transpose(outputs, [1,0,2] ) 
                    outputs = tf.unstack(outputs,axis = 0)
                    outputs = tf.concat(outputs, axis = 0)

                    outputs_split = tf.split(outputs, num_or_size_splits = int(d['K']), axis = 1) 
                    outputs_stack = tf.stack(outputs_split)
                    outputs = act(tf.reduce_mean(outputs_stack, axis = 0))

                    V = tf.get_variable("V", dtype= tf.float32, 
                            shape = [d['K'], d['L2']], initializer = initializer)  
                    b = tf.get_variable("b", dtype= tf.float32, 
                            shape = [d['L2']], initializer = initializer)  

                    yhat = act(tf.matmul(outputs,V) + tf.reshape(b, (1, d['L2'])))

            #compute the number of parameters to be trained
            tvars = tf.trainable_variables()
            tnparams = np.sum([np.prod(var.get_shape().as_list()) for var in tvars])
            saver = tf.train.Saver(tvars) 

            if d['count_mode']:
                print('The total number of parameters to be trained =', tnparams,
                      'The model is: ', d['decoder'])
                os._exit(0)

            #raise an error if we are outside the allowed range
            if d['min_params'] > tnparams or d['max_params'] < tnparams:
                raise num_paramsError
            else:
                self.tnparams = tnparams
                self.tvars_names = [var.name for var in tvars] 

            #define the cost         
            eps = 1e-16
            cost = tf.reduce_mean( y*(tf.log( y + eps ) - tf.log( yhat + eps ))  - y + yhat ) 
            if d['task'] == 'toy_example':
                cost = cost + d['sparsity_coef']*tf.reduce_sum(hhat)

            #define the optimizer
            if d['optimizer'] == 'Adam':
                train_step = tf.train.AdamOptimizer(d['LR']).minimize(cost)
            elif d['optimizer'] == 'RMSProp':
                train_step = tf.train.RMSPropOptimizer(d['LR'], 
                                    momentum = d['momentum'],
                                    centered = True).minimize(cost)   
            elif d['optimizer'] == 'Adadelta':
                train_step = tf.train.AdadeltaOptimizer(d['LR']).minimize(cost)   

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

            if d['task'] == 'toy_example':
                graph_handles.update({'all_variables': tvars})
                                           
            return graph_handles

    def define_encoder(self, x, seqlens ,dropout_kps = tf.constant([1,1]),def_model_num = None ):  
        p1 = dropout_kps[0]
        p2 = dropout_kps[1]

        # unpack model specifications 
        d = self.model_specs
        wform, model, K, num_layers, mapping_mode, L1, L2 = d['wform'], d['encoder'], d['K'], d['num_layers'], d['mapping_mode'], d['L1'], d['L2']
        
        if model in d['one_basis_rnns']:
            
            if self.input_opt: # If we are building the separation graph

                if model == 'ob_mod_lstm': 
                    cell = ModLSTMCell(K, initializer = self.initializer,
                            input_opt = self.input_opt, 
                            wform = d['wform'], model_num = def_model_num)   
                elif model == 'ob_gru':
                    cell = tf.contrib.rnn.GRUCell(K)  
            else: # If we are building the training graph
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                if model == 'ob_mod_lstm': 
                    cell = ModLSTMCell(K, initializer = initializer,
                            input_opt = self.input_opt, wform = d['wform'])   
                elif model == 'ob_gru':
                    cell = tf.contrib.rnn.GRUCell(K)  

            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p1)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=p2)

            outputs, _= tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, 
                    time_major = True, sequence_length = seqlens )
            
            #outputs = tf.transpose(outputs, [1,0,2] ) 
            #outputs = tf.unstack(outputs,axis = 0)
            #outputs = tf.concat(outputs, axis = 0)
            
            return outputs


        elif model in d['mult_basis_rnns']:
            #x_tiled = tf.tile(x, [1, 1, K]) 
           
            if self.input_opt: # If we are building the separation graph
                if model == 'mb_mod_lstm': 
                    cell = ModLSTMCell(d['K_in'], initializer = self.initializer,
                            input_opt = self.input_opt, 
                            wform = d['wform'], 
                            model_num = def_model_num,
                            num_rnns = K)   
                elif model == 'mb_gru':
                    cell = tf.contrib.rnn.GRUCell(d['K_in']*K)  
            else: # If we are building the training graph
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                if model == 'mb_mod_lstm': 
                    cell = ModLSTMCell(d['K_in'], initializer = initializer,
                            input_opt = self.input_opt, 
                            wform = d['wform'],
                            num_rnns = K)   

            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p1)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=p2)

            outputs, _= tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, 
                    time_major = True, sequence_length = seqlens )

            outputs = tf.transpose(outputs, [1,0,2] ) 
            outputs = tf.unstack(outputs,axis = 0)
            outputs = tf.concat(outputs, axis = 0)

            outputs_split = tf.split(outputs, 
                                     axis=1, 
                                     num_or_size_splits=int(K))
            outputs_split_sum = [tf.reduce_sum(output, 1, keep_dims=True) for output in outputs_split] 
            outputs = tf.concat(outputs_split_sum, axis=1)

            return outputs

        elif model == 'feed_forward':

            with tf.variable_scope('encoder'):
                if self.input_opt:
                    vars_to_use = [var for var in self.initializer if 'model'+str(def_model_num)+'/encoder' in var[0]] 
                    for var in vars_to_use:
                        if '/V' in var[0]:
                            V = tf.constant(var[1])
                        else:
                            b= tf.constant(var[1])
                else:
                    initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                    
                    V = tf.get_variable("V", dtype= tf.float32, 
                            shape = [d['L1'], d['K']], initializer = initializer)  
                    b = tf.get_variable("b", dtype= tf.float32, 
                            shape = [d['K']], initializer = initializer)  

            x = tf.transpose(x, [1, 0, 2])
            x = tf.unstack(x, axis = 0)
            x = tf.concat(x, axis = 0)

            outputs = ( tf.matmul(x,V) + b ) 

            return outputs
        
        elif model == 'convolutive':
            
            with tf.variable_scope('encoder'):
                if self.input_opt:
                    vars_to_use = [var for var in self.initializer if 'model'+ str(def_model_num)+'/encoder' in var[0]] 
                    for var in vars_to_use:
                        if '/fltr' in var[0]:
                            fltr = tf.constant(var[1])
                        else:
                            b = tf.constant(var[1])
                else:
                    initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32)
                    fltr = tf.get_variable("fltr", dtype= tf.float32, 
                        shape = [d['ntaps'], 1, d['L1'], d['K']], 
                        initializer = initializer)  
                    b = tf.get_variable("b", dtype= tf.float32, 
                        shape = [d['K']], initializer = initializer)  

            x = tf.transpose(x, [1, 0, 2])
            x = tf.unstack(x, axis = 0)
            x = tf.concat(x, axis = 0)

            x_rev = tf.reverse( x, axis = [0])
            x_pad = tf.concat([x_rev, tf.zeros([d['ntaps']-1,d['L1']])],
                                axis = 0)
            x_reshape = tf.reshape( x_pad, [1, -1, 1, d['L1']])
            
            yhat = tf.nn.conv2d( x_reshape, filter =  fltr,
                                 strides = [1,1,1,1], padding = "VALID")
            yhat = tf.reshape( yhat, [-1,d['K']])
            yhat = tf.reverse(yhat, axis = [0])

            yhat = (yhat + b)
            
            return yhat

    def build_separation_graph(self):
        d = self.model_specs #unpack the model specifications
        if d['activation'] == 'relu':
            act = tf.nn.relu
        elif d['activation'] == 'softplus':
            act = tf.nn.softplus

        with tf.device('/' + d['device']):
            mask = tf.placeholder(tf.float32, [None])
            y = tf.placeholder(tf.float32, [None, d['L1']],"y") 

            dropout_kps = tf.placeholder(tf.float32, [2], "dropout_params")
            seq_lens = tf.placeholder(tf.int32, [None])
           
            if d['separation_method'] == 'complete': 
                x1 = tf.get_variable("x1", dtype = tf.float32,
                    shape = [d['num_steps_test'], d['batchsize_t'], d['L1']])
                x2 =  tf.get_variable("x2", dtype = tf.float32,
                    shape = [d['num_steps_test'], d['batchsize_t'], d['L1']])

                h1 = self.define_encoder(x1, 
                        seqlens = seq_lens, dropout_kps = dropout_kps, def_model_num = 1)
                h2 = self.define_encoder(x2,
                        seqlens = seq_lens, dropout_kps = dropout_kps, def_model_num = 2)
                h1, h2 = act(h1), act(h2)
            else:
                h1 = tf.get_variable("h1", dtype = tf.float32,
                        shape = [d['num_steps_test'],d['K']] )
                h2 = tf.get_variable("h2", dtype = tf.float32,
                        shape = [d['num_steps_test'],d['K']] )

            #decoder 1
            vars_to_use = [var for var in self.initializer if 'model1/decoder' in var[0]] 

            if d['decoder'] == 'feed_forward':
                for var in vars_to_use:
                    if '/V' in var[0]:
                        V1 = tf.constant(var[1])
                    else:
                        b1 = tf.constant(var[1])

                y1hat = act(tf.matmul(h1,V1) + b1)

            elif d['decoder'] == 'convolutive':
                for var in vars_to_use:
                    if '/fltr' in var[0]:
                        fltr = tf.constant(var[1])
                    else:
                        b = tf.constant(var[1])
                    
                hhat_rev = tf.reverse( h1, axis = [0])
                hhat_pad = tf.concat([hhat_rev, tf.zeros([d['ntaps']-1,d['K']])],
                                    axis = 0)
                hhat_reshape = tf.reshape( hhat_pad, [1, -1, 1, d['K']])
                
                yhat = tf.nn.conv2d( hhat_reshape, filter =  fltr,
                                     strides = [1,1,1,1], padding = "VALID")
                yhat = tf.reshape( yhat, [-1,d['L2']])
                yhat = tf.reverse(yhat, axis = [0])
                
                y1hat = act(yhat + tf.reshape(b, (1, d['L2'])))

            #decoder 2
            vars_to_use = [var for var in self.initializer if 'model2/decoder' in var[0]] 
            if d['decoder'] == 'feed_forward':
                for var in vars_to_use:
                    if '/V' in var[0]:
                        V2 = tf.constant(var[1])
                    else:
                        b2 = tf.constant(var[1])
                
                y2hat = act(tf.matmul(h2,V2) + b2)
            
            elif d['decoder'] == 'convolutive':
                for var in vars_to_use:
                    if '/fltr' in var[0]:
                        fltr = tf.constant(var[1])
                    else:
                        b = tf.constant(var[1])
                    
                hhat_rev = tf.reverse( h2, axis = [0])
                hhat_pad = tf.concat([hhat_rev, tf.zeros([d['ntaps']-1,d['K']])],
                                    axis = 0)
                hhat_reshape = tf.reshape( hhat_pad, [1, -1, 1, d['K']])
                
                yhat = tf.nn.conv2d( hhat_reshape, filter =  fltr,
                                     strides = [1,1,1,1], padding = "VALID")
                yhat = tf.reshape( yhat, [-1,d['L2']])
                yhat = tf.reverse(yhat, axis = [0])
                
                y2hat = act(yhat + tf.reshape(b, (1, d['L2'])))
            
            yhat = y1hat + y2hat
    
            eps = 1e-16
            cost = tf.reduce_mean( y*(tf.log( y + eps ) - tf.log( yhat + eps ))  - y + yhat ) 

            #cost = tf.reduce_sum(tf.square((y - yhat1 - yhat2)))

            if d['optimizer'] == 'Adam':
                train_step = tf.train.AdamOptimizer(d['LR']).minimize(cost)
            elif d['optimizer'] == 'RMSProp':
                train_step = tf.train.RMSPropOptimizer(d['LR'], 
                                    momentum = d['momentum'],
                                    centered = True).minimize(cost)   
            elif d['optimizer'] == 'Adadelta':
                train_step = tf.train.AdadeltaOptimizer(d['LR']).minimize(cost)   

            #return the graph handles 
            graph_handles = {'train_step':train_step,
                             'y':y,
                             'y1hat':y1hat,
                             'y2hat':y2hat,
                             'mask':mask,
                             'cost':cost,
                             'dropout_kps':dropout_kps,
                             'seq_lens':seq_lens,
                             #'relevant_inds':relevant_inds,
                             #'targets':targets,
                             }
                                           
                             
            return graph_handles

    def optimizer(self, data, rnn_handles, sess):
        """This function runs the optimizer for the given data and given rnn graph referenced by rnn_handles """

        d = self.model_specs # unpack the variables 
    
        tr = SimpleDataIterator(data, num_buckets = d['num_buckets'])
        #tst = SimpleDataIterator(data['Test'])
        #valid = SimpleDataIterator(data['Validation'])

        all_times, tr_logls, test_logls, valid_logls = [], [], [], [] 
        for ep in range(d['EP']):
            t1, tr_logl = time.time(), []
            while tr.epochs == ep:
                trb = tr.next_batch(
                        n = d['batchsize'+str(self.model_num)], 
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
            print("The Model is ",d['encoder'],d['wform'],
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
        if d['task' ] == 'toy_example':
            all_variables = rnn_handles['all_variables']
            
            for var in all_variables:
                if 'decoder/fltr' in var.name:
                    filter_weights = sess.run(var, feed).squeeze()
                    filter_weights = np.transpose(filter_weights, [2, 0, 1])

        pdb.set_trace()

        return all_times, tr_logls, test_logls, valid_logls

    def input_optimizer(self, data, rnn_handles, sess):
        d = self.model_specs # unpack the variables 
    
        tr = SimpleDataIterator(data, num_buckets = d['num_buckets'])
        
        all_times, tr_logls, test_logls, valid_logls = [], [], [], [] 
        for ep in range(d['EP']):
            t1, tr_logl = time.time(), []
            while tr.epochs == ep:
                trb = tr.next_batch(
                        n = d['batchsize_t'], 
                        task = d['task'], 
                        verbose = d['verbose'])      

                feed = {rnn_handles['y']:trb[1], 
                        rnn_handles['mask']:trb[2],
                        rnn_handles['seq_lens']:trb[3], 
                        rnn_handles['dropout_kps']:d['dropout'] }  

                tr_cost,_ = sess.run( 
                        [rnn_handles['cost'], rnn_handles['train_step']], feed) 
                
                if d['verbose']:
                    print("Iteration ", ep,"Training cost = ", tr_cost) 
            t2 = time.time()

        
        y1hat = sess.run(rnn_handles['y1hat'],feed).transpose()
        y2hat = sess.run(rnn_handles['y2hat'],feed).transpose()

        return y1hat, y2hat

def return_Klimits(model, wform, data):
    """We use this function to select the upper and lower limits of number of 
    hidden units per layer depending on the task and the dataset. The user can also choo    se to limit the upper and lower limit of allowable number of trainable parameters"""

    if model in ['mod_lstm', 'lstm']:
        min_params = 1e1; max_params =  7e7 # in our waspaa paper we basically did not use lower and upper bounds for number of parameters
        K_min, K_max = 10, 100

    elif model in ['mb_mod_lstm']:
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 10, 100 

    elif model == 'feed_forward':
        min_params = 1e1; max_params = 7e7 
        K_min, K_max = 10, 100

    elif model == 'convolutive':
        min_params = 1e1; max_params = 7e7
        K_min, K_max = 10, 100

    return K_min, K_max, min_params, max_params 

def generate_random_hyperparams(lr_min, lr_max, K_min, K_max, num_layers_min, num_layers_max,load_hparams, num_configs = 60):
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
        #exp_data = np.load('experiment_data/nmf_data_timit_model_bi_mod_lstm_diag_to_full_device_cpu:0_1490813245.npy')[load_hparams[1]]
        #pick K in this range
        Krange = np.round(np.arange(start = K_min, stop = K_max+1, step = 10))
        Krange = np.tile(Krange, [int(num_configs/len(Krange)),1 ]).transpose()
        Krange = Krange.reshape(-1)
        K = Krange[load_hparams[1]]
        
        set_seed = False
        if (load_hparams[1]==0):
            set_seed = True
        elif (load_hparams[1]>0):
            if ( np.abs(Krange[(load_hparams[1])] - Krange[(load_hparams[1]-1)]) > 0):
                set_seed = True
        
        # lr_exp = np.random.uniform(lr_min, lr_max)
        # lr = 10**(lr_exp)
        # num_layers = np.random.choice(np.arange(num_layers_min, num_layers_max + 1),1)[0]
        # momentum = np.random.uniform(0,1)
        lr = 0.001
        num_layers = 1
        momentum = 0.7

    return lr, K, num_layers, momentum, set_seed

def load_data(dictionary):
    """this function loads the data, and sets the associated parameters (such as output and input dimensionality and batchsize) according to the specified task, which are either text, music, speech or digits """
    task, data = dictionary['task'], dictionary['data']

    #data = [generate_separationdata(L,T,step)]
    if task == 'toy_example':
        
        # load the toy data as M
        # put the length of M in lengths 1
        L, T, step = 10, 100, 25
        
        M = [generate_separationdata(L,T,step)]
        lengths = [T]
       
        # set the toy magnitude spectrogram as M
        d = {'data':M, 'lengths': lengths}
        df_train = pd.DataFrame( d )
        
        L1 = L2 = L
        outstage = 'relu'
        mapping_mode = 'seq2seq'
        iterator = 'SimpleDataIterator'

        parameters = {'batchsize1':2,
                      'L1':L1,'L2':L2,
                      'outstage':outstage,
                      'mapping_mode':mapping_mode,
                      'iterator':iterator,
                      'num_buckets':None
                      }
        
        return df_train, parameters


    elif task == 'source_sep':
        L, T, step = 150, 200, 50  

        #random.seed( s)
        #we pick the set according to trial number 
        Z_temp = sound_set(3) 
        Z = Z_temp[0:4]
        mf = Z_temp[4]
        ff = Z_temp[5]

        # Front-end details
        #if hp is None:
        sz = 1024       
        hp = sz/4
        wn = reshape( np.hanning(sz+1)[:-1], (sz,1))**.5

        # Make feature class
        FE = sound_feats( sz, hp, wn)
        al = 'rprop'
        hh = .0001

        len_th = 100

        #source 1
        M1,P = FE.fe( Z[0] )
        
        M1 = np.split(M1, np.arange(len_th,M1.shape[1],len_th), axis = 1)
        lengths1 = [spec.shape[1] for spec in M1]
        
        d = {'data':M1, 'lengths': lengths1}
        df_train1 = pd.DataFrame( d )
        
        #source 2
        M2,P = FE.fe( Z[1] )
        
        M2 = np.split(M2, np.arange(len_th,M2.shape[1],len_th), axis = 1)
        lengths2 = [spec.shape[1] for spec in M2]
        
        d = {'data':M2, 'lengths': lengths2}
        df_train2 = pd.DataFrame( d )

        #mixtures
        M_t, P_t = FE.fe( Z[2]+Z[3] )
                
        if dictionary['encoder'] in dictionary['mult_basis_rnns']:
            M_t = np.split(M_t, np.arange(len_th, M_t.shape[1], len_th), axis = 1)
            P_t = np.split(P_t, np.arange(len_th, P_t.shape[1],len_th), axis = 1)

            num_steps_test = len_th
            batchsize1, batchsize2, batchsize_t = len(M1), len(M2), len(M_t)
        else:
            M_t, P_t = [M_t], [P_t]
            
            num_steps_test = M_t[0].shape[1]
            batchsize1, batchsize2, batchsize_t = len(M1), len(M2), len(M_t)
        lengths_t = [spec.shape[1] for spec in M_t]
        
        d = {'data':M_t, 'lengths': lengths_t, 'phase': P_t }
        df_test = pd.DataFrame( d )
        
        df_valid = None

        L1 = L2 = M1[0].shape[0]
        outstage = 'relu'
        mapping_mode = 'seq2seq'
        iterator = 'SimpleDataIterator'
        num_buckets = None

        parameters = {'batchsize1':batchsize1,'batchsize2':batchsize2,
                      'batchsize_t':batchsize_t,
                      'L1':L1,'L2':L2,
                      'outstage':outstage,
                      'mapping_mode':mapping_mode,
                      'num_steps_test':num_steps_test,
                      'iterator':iterator,
                      'num_buckets':num_buckets,
                      'len_th':len_th,
                      'audio_files':Z,
                      'FE':FE,
                      'mf':mf,
                      'ff':ff}
        
        return {'Train1':df_train1, 
                'Train2':df_train2, 
                'Test':df_test, 
                'Validation':df_valid}, parameters

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
        
        if task in ['source_sep', 'toy_example']:
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

