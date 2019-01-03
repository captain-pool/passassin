import tensorflow as tf
import numpy as np
import os
from .settings import settings
#Architectural Notes:
#1. Generator: GRU Units
#    - Many to Many
#    - L1: Number units: 128, Activation = ELU
#    - L2: Number units: 32, Activation = ELU
#    - Dense: on each head and use tf.gather to get each output, and calculate avg loss in each step
#      and minimize that
#    - softmax activation, probablity sampling
#    - Softmax Cross Entropy
#2. Discriminator: GRU Units
#    - Many to One
#    - L1: Number of Units: 32 Activation = ELU
#    - Dense: 128: Activation: sigmoid
#    - Dense: 128: Activation: sigmoid
#    - Loss: Sigmoid Cross Entropy

class Generator:
    def __init__(self,**kwargs):
        '''
        Params:
            L1(int): Number of GRU Units in Layer 1
            L2(int): Number of GRU UNits in Layer 2
              noise: Tensor having Noise Component
                     Shape: [Batch_Size, Timesteps, Size]
               keep: Keep Probability for Dropout Layers
               disc: Discriminator Class Object
               real: Tensor of Real Data
        '''
        self.l1 = kwargs.get("L1",128)
        self.l2 = kwargs.get("L2",32)
        self.z = kwargs["noise"]
        self.kp = kwargs.get("keep",0.7)
        self.disc = kwargs["disc"]
        self.real = kwargs["real"]
        self.batch_size = tf.shape(self.z)[0]
    def dense(self,X,out):
        with tf.name_scope("dense"):
            w = tf.Variable(tf.random_normal([X.get_shape()[-1].value,out]))
            b = tf.Variable(tf.random_normal([out]))
        return tf.matmul(X,w)+b
    def use_disc(self,input_):       
        disc_out = self.disc.set_input(input_).build(reuse = True) # include variable reuse in build method of disc
        return disc_out
    def optimize(self,learning_rate = 0.01):
        return tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(self.loss,var_list = self.train_vars)
    def build(self):
        self.z = tf.transpose(self.z,[1,0,2])
        with tf.variable_scope("GEN"):
            with tf.name_scope("GRU"):
                cells = [tf.nn.rnn_cell.GRUCell(self.l1),tf.nn.rnn_cell.GRUCell(self.l2)]
                cells = list(map(lambda x:tf.nn.rnn_cell.DropoutWrapper(x,output_keep_prob = self.kp),cells))
                rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                out,state = tf.nn.dynamic_rnn(rnn_cell,self.z,dtype = tf.float32,time_major = True)
            out = tf.reshape(out,[-1,self.l2])
            out = tf.nn.relu(self.dense(out,self.l2))#),[-1,self.z.get_shape()[1].value,self.l2])
            self.z = tf.transpose(self.z,[1,0,2])
            targetFeature = self.z.get_shape()[-1].value#self.z.get_shape()[1:].num_elements()
            
            out = tf.reshape(self.dense(out,targetFeature),[-1,self.z.get_shape()[1].value,self.z.get_shape()[-1].value])
            out = tf.nn.softmax(out,axis = 1,name = "output") # To Find Probability Distribution over each batch, which starts from axis=1
            #out = tf.transpose(out,[1,0,2])
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="GEN")
        self.output = out
        disc_out = self.use_disc(out)
        self.adv_out = disc_out
        self.loss = -tf.reduce_mean(disc_out)
        return self
class Discriminator:
    def __init__(self,**kwargs):
        '''
        Params:
            L1(int): Number of GRU Units in Layer 1
            L2(int): Number of GRU UNits in Layer 2
              input: Tensor having inputs to Discriminator
                     Shape: [Batch_Size, Timesteps, Size]
               keep: Keep Probability for Dropout Layers
        '''
        self.L1 = kwargs.get("L1",128)
        self.L2 = kwargs.get("L2",32)
        self.input = kwargs["input"]
        self.kp = kwargs.get("keep",0.8)
        self.batch_size = tf.shape(self.input)[0]
    def set_input(self,tensor):
        assert self.input.get_shape()[1:] == tensor.get_shape()[1:]
        self.input_ = tensor
        return self
    def dense(self,X,out,name):
        w = tf.get_variable("w_"+name,[X.get_shape()[-1].value,out])
        b = tf.get_variable("b_"+name,[out])
        return tf.matmul(X,w)+b
    def optimize(self,lossTensor,learning_rate = 0.01):
        return tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(lossTensor,var_list = self.train_vars)
    def build(self,reuse = False):
        if not reuse:
            inp = self.input
        else:
            inp = self.input_
        with tf.variable_scope("DISC",reuse = reuse):
            with tf.name_scope("GRU"):
                cells = [tf.nn.rnn_cell.GRUCell(self.L1,reuse = reuse),tf.nn.rnn_cell.GRUCell(self.L2,reuse = reuse)]
                cells = list(map(lambda x:tf.nn.rnn_cell.DropoutWrapper(x,output_keep_prob = self.kp),cells))
                rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                inp = tf.transpose(inp,[1,0,2])
                out,state = tf.nn.static_rnn(rnn_cell,tf.unstack(inp),dtype = tf.float32)
            out = out[-1]
            out = tf.nn.relu(self.dense(out,128,"h1"))
            out = tf.nn.sigmoid(self.dense(out,1,"h2"))
            inp = tf.transpose(inp,[1,0,2])
        if not reuse:
            self.input = inp
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "DISC")
            self.output = out
            return self
        return out
