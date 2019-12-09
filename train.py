import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import glob
import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider_xiu
import tf_util
#from model_promerg_dec2 import *
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--num_pixel', type=int, default=125, help='img pixel number [default: 125]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--num_cls', type=int, help='number of cls')
parser.add_argument('--rgb', type=int, default=0, help='use only xyz and intensity')
parser.add_argument('--train_data_loc', type=str, help='pcl train data location')
parser.add_argument('--test_data_loc', type=str, help='pcl test data location')
parser.add_argument('--train_img_loc', type=str, help='img train data location')
parser.add_argument('--test_img_loc', type=str, help='img test data location')
#parser.add_argument('--test_percentage', type=int, default=0.2, help='percentatge of samples used for test')
parser.add_argument('--train_model', type=str, help='model name and location')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_IMG_PIXELS = FLAGS.num_pixel
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CLASSES = FLAGS.num_cls
#INPUT_DIR = FLAGS.input_dir
RGB = FLAGS.rgb
#test_percentage = FLAGS.test_percentage
train_data_loc = FLAGS.train_data_loc
test_data_loc = FLAGS.test_data_loc
train_img_loc = FLAGS.train_img_loc
test_img_loc = FLAGS.test_img_loc

LOG_DIR = FLAGS.log_dir
train_model = FLAGS.train_model

print (train_model)
#from train_model import *
X = __import__(train_model)

#print (X)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp model_xiu.py %s' % (LOG_DIR)) # bkp of model def
#os.system('cp train_xiu.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')



BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# new edits
train_data = np.load(train_data_loc)
test_data = np.load(test_data_loc)

#data = np.load(data_loc)
final_order_pcl = np.array([6,7,3,4,0,1,2,5]) #x,y,z,i,r,g,b,l
train_data = train_data[:,:,final_order_pcl]
test_data = test_data[:,:,final_order_pcl]

#data = data[:,:,final_order_pcl]     # change the column of -1 axis

train_data[:,:,4:7] /= 255.0
train_data[:,:,3] /= train_data[:,:,3].max()

test_data[:,:,4:7] /= 255.0
test_data[:,:,3] /= test_data[:,:,3].max()

print ("++++++++++++++++++++++++++++++++++")

print (" test dimension0 : " + str(test_data[:,:,0].max()) + ": " +   str(train_data[:,:,0].max()))
print (" test dimension1 : " + str(test_data[:,:,1].max()) + ": " +   str(train_data[:,:,1].max()))
print (" test dimension2 : " + str(test_data[:,:,2].max()) + ": " +   str(train_data[:,:,2].max()))
print (" test dimension3 : " + str(test_data[:,:,3].max()) + ": " +   str(train_data[:,:,3].max()))
print (" test dimension4 : " + str(test_data[:,:,4].max()) + ": " +   str(train_data[:,:,4].max()))
print (" test dimension5 : " + str(test_data[:,:,5].max()) + ": " +   str(train_data[:,:,5].max()))
print (" test dimension6 : " + str(test_data[:,:,6].max()) + ": " +   str(train_data[:,:,6].max()))
print (" test dimension7 : " + str(test_data[:,:,7].max()) + ": " +   str(train_data[:,:,7].max()))
print ("++++++++++++++++++++++++++++++++++")
img_train = np.load(train_img_loc)
img_test = np.load(test_img_loc)
img_train = np.delete(img_train, [3,4],-1)
img_test = np.delete(img_test, [3,4],-1)
#delete descreete (DI)
#img = np.delete(img, [3,4],-1)
img_train[:,:,:,:3] /=255.0
img_test[:,:,:,:3] /=255.0

#test_label = data[:test_samples,:,-1].astype(int)
test_label = test_data[:,:,-1].astype(int)
test_data = test_data[:, :,:7]
img_test = img_test[:,:,:,:3]

#for fname in glob.glob('script_and_dataset/test_8m/' + '*raw.npy'):
train_label = train_data[:,:,-1].astype(int)
train_data = train_data[:, :,:7]
img_train = img_train[:,:,:,:3]

print("train_data: " + str(train_data.shape) + "train_label: " + str(train_label.shape) + "img_train: "+ str(img_train.shape))
print("test_data: " + str(test_data.shape) + "test_label: " + str(test_label.shape) + "img_test: " + str(img_test.shape))

feature_num = train_data.shape[-1]
########################################################
#### only test with x, y, z, intensity ####
if RGB == 0:
  feature_num = 4
  train_data = np.delete(train_data, [4,5,6], -1)
  test_data = np.delete(test_data, [4,5,6], -1)
  print(train_data.shape)
  print(test_data.shape)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = X.placeholder_inputs(BATCH_SIZE, NUM_POINT, feature_num)
            img_pl = X.placeholder_img(BATCH_SIZE,NUM_IMG_PIXELS,3)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred,pred_ne = X.get_model(pointclouds_pl,img_pl, is_training_pl, NUM_CLASSES, feature_num, bn_decay=bn_decay)
            loss = X.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER == 'Adelta':
                optimizer = tf.keras.optimizers.Adadelta(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'img_pl': img_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'pred_ne': pred_ne,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 5 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_xiu.ckpt" + str(epoch)))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, img_data, current_label, _ = provider_xiu.shuffle_data(train_data[:,0:NUM_POINT,:],img_train[:,0:NUM_IMG_PIXELS,0:NUM_IMG_PIXELS,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    gt_cls = [0 for _ in range(NUM_CLASSES)]
    tp_cls = [0 for _ in range(NUM_CLASSES)]
    #gt_cls = [0 for _ in range(7)]
    #tp_cls = [0 for _ in range(7)]
    
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['img_pl']: img_data[start_idx:end_idx,:,:,:],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, pred_net = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['pred_ne']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        print ("loss: " + str(loss_val)) 
        print ("++++++++++++++  pred_net value  ++++++++++++++++++")
        print(str(pred_net.shape))
        print (str(pred_net.max()))
        #print(pred_net)
        print ("++++++++++++++  pred_val value  ++++++++++++++++++")
        print(str(pred_val.shape))
        print (str(pred_val.max()))
        #print(pred_val)
        print ("++++++++++++++  data value   +++++++++++++++++++++")
        print(str(current_data[start_idx:end_idx].shape))
        #print(current_data[start_idx:end_idx])
        print ("+++++++++++++++++++++++++++++++++++++++++++++++++")
        pred_val = np.argmax(pred_val, -1)
        # output feature  map
        #x,y,z,i,r,g,b,l
        xyzlpf = np.concatenate([current_data[start_idx:end_idx,:,:3],np.expand_dims(current_label[start_idx:end_idx], -1), np.expand_dims(pred_val, -1), pred_net],-1)
        np.save('../result_eval/' + str(train_model) + 'xyzlpf.npy', xyzlpf)
       
        #print ("current_label")
        unique, counts = np.unique(current_label[start_idx:end_idx], return_counts=True)
        # accumulate the total cls-wise number
        for i in range(len(unique)):
          #ignore unclassified (unique[0])
          #if unique[i] == 0:
          #  continue
          gt_cls[unique[i]] += counts[i] # total num of each cls accumulation 
        # true positive (tp) index
        tp_idx = pred_val == current_label[start_idx:end_idx]
        #print ('tp_idx_id')
        #print (tp_idx.shape)
        # correct and prediction element-wise comparison and count
        a = pred_val.flatten()
        b = current_label[start_idx:end_idx].flatten()
        c = tp_idx.flatten()

        for j in range(b.shape[0]):
          if (c[j]):
            #if b[j] == 0:
            #   continue
            #print("b[j]",b[j])
            tp_cls[b[j]] += 1
        
        #print('after argmax:')
        #print(pred_val.shape)
        #print(pred_val)
        #print('current_label: ')
        #print "current_label[start_idx:end_idx]"
        #print(current_label[start_idx:end_idx].shape)
        #print(current_label[start_idx:end_idx])
        excl_uncls = np.count_nonzero(current_label[start_idx:end_idx] == 0)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT - excl_uncls)
        loss_sum += loss_val
    print('gt_cls: ' + str(gt_cls))
    print('tp_cls: ' + str(tp_cls)) 
    gt_cls_arr = np.asarray(gt_cls, dtype=float)
    tp_cls_arr = np.asarray(tp_cls, dtype=float)
    per_cls_acc = tp_cls_arr / gt_cls_arr
    #print('element_wise acc: ')
    #print(tp_cls_arr/gt_cls_arr)
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    for i in range(per_cls_acc.shape[0]):
      log_string('class %d: %f'%(i, per_cls_acc[i])) 

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    #gt_cls = [0 for _ in range(7)]
    #tp_cls = [0 for _ in range(7)]
    
    gt_cls = [0 for _ in range(NUM_CLASSES)]
    tp_cls = [0 for _ in range(NUM_CLASSES)]
    
    log_string('===============================================')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    img_data = img_test[:,0:NUM_IMG_PIXELS,0:NUM_IMG_PIXELS,:]    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['img_pl']: img_data[start_idx:end_idx,:,:,:],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        #print ("pred_val for evaluation")
        #print (pred_val.shape)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        unique, counts = np.unique(current_label[start_idx:end_idx], return_counts=True)
        # accumulate the total cls-wise number
        for i in range(len(unique)):
          gt_cls[unique[i]] += counts[i] # total num of each cls accumulation 
        # true positive (tp) index
        tp_idx = pred_val == current_label[start_idx:end_idx]
        #print ('tp_idx_id')
        #print (tp_idx.shape)
        # correct and prediction element-wise comparison and count
        a = pred_val.flatten()
        b = current_label[start_idx:end_idx].flatten()
        c = tp_idx.flatten()
        for j in range(b.shape[0]):
          if (c[j]):
            #print(b[j])
            tp_cls[b[j]] += 1

        total_correct += correct

        excl_uncls = np.count_nonzero(current_label[start_idx:end_idx] == 0)
        #print ("excl_uncls",excl_uncls) 
        total_seen += (BATCH_SIZE*NUM_POINT - excl_uncls)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
    print('gt_cls: ' + str(gt_cls))
    print('tp_cls: ' + str(tp_cls))
    gt_cls_arr = np.asarray(gt_cls, dtype=float)
    tp_cls_arr = np.asarray(tp_cls, dtype=float)
    per_cls_acc = tp_cls_arr / gt_cls_arr        
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    for i in range(per_cls_acc.shape[0]):
      log_string('class %d: %f'%(i, per_cls_acc[i]))
     

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
