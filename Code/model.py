#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:47:14 2017

author: Irving she

Aim: Process our own data 
     - input_data.py: read in data and generate batches
     - model: build the model architecture
     - training: train

Reference:
    https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/input_data.py
"""
#%%

import tensorflow as tf

#%%
def inference(images,batch_size,n_classes):
    '''
    Build the model
    Args:
        image: image batch,4D tensor,tf.float32,[batch_size,width,height,channels]
    Returns:
        output tensor with the computed logits,float,[batch_size,n_classes]
    '''
    
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
 
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [11,11,3,48],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        conv = tf.nn.conv2d(images,weights,strides=[1,4,4,1],padding='VALID')
        biases = tf.get_variable('biases',
                                 shape=[48],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name= scope.name)

    #pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
      
        
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',shape=[5,5,48,128],
                                  dtype= tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[128],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name='conv2')
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pooling2')
        
    #conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',shape=[3,3,128,256],
                                  dtype= tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[256],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation,name='conv3')
    
    #conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',shape=[3,3,256,384],
                                  dtype= tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[384],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(pre_activation,name='conv4')
    
    #conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights',shape=[3,3,384,256],
                                  dtype= tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[256],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(pre_activation,name='conv5')
    
    #pool5
    with tf.variable_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    
    #Fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool5,shape=[batch_size,-1])
        drop1 = tf.nn.dropout(reshape,0.8)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',shape=[dim,256],
                                  dtype = tf.float32,
                                  initializer= tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1)
                                 )
        
        fc1 =tf.nn.relu(tf.matmul(drop1,weights)+biases,name=scope.name)
   
    #Fc2
    with tf.variable_scope('fc2') as scope:
         weights = tf.get_variable('weights',
                                  shape=[256,128],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
         biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1)
                                 )
          
         fc2 = tf.nn.relu(tf.matmul(fc1,weights)+biases,name='fc2')
   
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1/128,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                initializer = tf.truncated_normal_initializer(0.1)
                                )
        
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%


      
        