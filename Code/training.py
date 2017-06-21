#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:54:37 2017

@author: Irving she

Aim: Process our own data 
     - input_data.py: read in data and generate batches
     - model: build the model architecture
     - training: train

Reference:
    https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/01%20cats%20vs%20dogs/input_data.py
"""
#%%

import os
import numpy as np
import tensorflow as tf
import input_data
import model
#%%

N_CLASSES = 2
IMG_W = 227
IMG_H = 227
BATCH_SIZE = 20
CAPACITY = 2000
MAX_STEP = 20000
learning_rate = 0.0001

#%%
 
def run_training():
   
    train_dir = '/home/llc/TF_test/Cats_Vs_Dogs/Data/train/'
    logs_train_dir = '/home/llc/TF_test/Cats_Vs_Dogs/logs/'
    
    train, train_label = input_data.get_files(train_dir)
    train_batch,train_label_batch = input_data.get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    train_logits = model.inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = model.losses(train_logits,train_label_batch)
    train_op = model.trainning(train_loss,learning_rate)
    train_acc = model.evaluation(train_logits,train_label_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            
            if step % 50 == 0:
                print('Step %d,train loss = %.2f,train_accuracy=%.2f%%' %(step,tra_loss,tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 4000 ==0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
#%%  Evaluate one image

from PIL import Image  #PIL not support py3,so select Pillow (Pillow=PIL)
import matplotlib.pyplot as plt
import os

def get_one_image(test):
    '''
    Randomly pick one image from training data
    Return: ndarry
    '''
    n= len(test)
    ind = np.random.randint(0,n)
    img_dir = test[ind]
    
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([227,227])
    image = np.array(image)
    return image

def evaluate_one_image():
    '''
    Test one image against the saved models and parameters
    '''
    test_dir = '/home/llc/TF_test/Cats_Vs_Dogs/Data/test/'
    #test,test_label = input_data.get_files(test_dir)
    test_image = []
    for file in os.listdir(test_dir):
        test_image.append(test_dir + file)
        
    test_image = list(test_image)
    image_array = get_one_image(test_image)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image  = tf.cast(image_array,tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,227,227,3])
        logit = model.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32,shape=[227,227,3])
        
        logs_train_dir = '/home/llc/TF_test/Cats_Vs_Dogs/logs/'
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            print ("Reading checkpoint...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
             
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
    
