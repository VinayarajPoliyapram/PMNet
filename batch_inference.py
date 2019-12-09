import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import time
#import gdal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
#from model_promerg_dec2 import *
#from model_xiu_only_xyz_for_pro_merge import *
import indoor3d_util_xiu
import preprocessing_tools2 as preprocessing_tools1
#import preprocessing_tools_unet as preprocessing_tools1
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--num_pixel', type=int, default=125, help='img pixel number [default: 125]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
#parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--data_path', required=True, help='path to data.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
parser.add_argument('--cls_num', type=int, default=0, help='number of classes')
parser.add_argument('--block_size', type=int, help='size of block')
parser.add_argument('--rgb', type=int, help='use only geometrical info')


#parser.add_argument('--train_data_loc', type=str, help='pcl train data location')
parser.add_argument('--test_data_loc', type=str, help='pcl test data location')
#parser.add_argument('--train_img_loc', type=str, help='img train data location')
parser.add_argument('--test_img_loc', type=str, help='img test data location')

parser.add_argument('--train_model', type=str, help='model name and location')
parser.add_argument('--save_probability', type=str,default="No", help='save probability if Yes')
parser.add_argument('--save_time', type=str,default="time.txt", help='time of running')
FLAGS = parser.parse_args()

TRAIN=False
RGB = FLAGS.rgb
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_IMG_PIXELS = FLAGS.num_pixel
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
train_model = FLAGS.train_model
save_probability =FLAGS.save_probability
save_time = FLAGS.save_time
X = __import__(train_model)

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
#ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)] # data_path
DATA_PATH = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.data_path)] # data_path

NUM_CLASSES = FLAGS.cls_num
#train_data_loc = FLAGS.train_data_loc
test_data_loc = FLAGS.test_data_loc
#train_img_loc = FLAGS.train_img_loc
test_img_loc = FLAGS.test_img_loc

BLOCK_SIZE = FLAGS.block_size

# save time
f=open(str(save_time),"w+")
f.write("%s\n" % ("time for running"))
#train_data = np.load(train_data_loc)
test_data = np.load(test_data_loc)

#data = np.load(data_loc)
final_order_pcl = np.array([6,7,3,4,0,1,2,5]) #x,y,z,i,r,g,b,l
#data = data[:,:,final_order_pcl]     # change the column of -1 axis
#img = np.load(img_loc)

#train_data = train_data[:,:,final_order_pcl]
test_data = test_data[:,:,final_order_pcl]

#train_data[:,:,4:7] /= 255.0
#train_data[:,:,3] /= train_data[:,:,3].max()

test_data[:,:,4:7] /= 255.0
test_data[:,:,3] /= test_data[:,:,3].max()

print ("++++++++++++++++++++++++++++++++++")

print (" test dimension0 : " + str(test_data[:,:,0].max()))
print (" test dimension1 : " + str(test_data[:,:,1].max()))
print (" test dimension2 : " + str(test_data[:,:,2].max()))
print (" test dimension3 : " + str(test_data[:,:,3].max()))
print (" test dimension4 : " + str(test_data[:,:,4].max()))
print (" test dimension5 : " + str(test_data[:,:,5].max()))
print (" test dimension6 : " + str(test_data[:,:,6].max()))
print (" test dimension7 : " + str(test_data[:,:,7].max()))
print ("++++++++++++++++++++++++++++++++++")
#img_train = np.load(train_img_loc)
img_test = np.load(test_img_loc)
start = time.time()

#img_train = np.delete(img_train, [3,4],-1)
img_test = np.delete(img_test, [3,4],-1)
#delete descreete (DI)
#img = np.delete(img, [3,4],-1)
#img_train[:,:,:,:3] /=255.0
img_test[:,:,:,:3] /=255.0

#test_label = data[:test_samples,:,-1].astype(int)
label = test_data[:,:,-1].astype(int)
data = test_data[:, :,:7]
img = img_test[:,:,:,:3]

#for fname in glob.glob('script_and_dataset/test_8m/' + '*raw.npy'):
#train_label = train_data[:,:,-1].astype(int)
#train_data = train_data[:, :,:7]
#img_train = img_train[:,:,:,:3]

#print("train_data: " + str(train_data.shape) + "train_label: " + str(train_label.shape) + "img_train: "+ str(img_train.shape))
#print("test_data: " + str(test_data.shape) + "test_label: " + str(test_label.shape) + "img_test: " + str(img_test.shape))


# Take the percentage for test samples

FEATURE_NUM = data.shape[-1]

if RGB ==0:
  FEATURE_NUM = 4

print("RGB: " +str(RGB))
print(FEATURE_NUM)
#NUM_POINT = 12063
############
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = X.placeholder_inputs(BATCH_SIZE, NUM_POINT, FEATURE_NUM)
        img_pl = X.placeholder_img(BATCH_SIZE,NUM_IMG_PIXELS,3)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, pred_ne = X.get_model(pointclouds_pl,img_pl, is_training_pl, NUM_CLASSES, FEATURE_NUM)
        #pred = get_model(pointclouds_pl, is_training_pl,NUM_CLASSES, FEATURE_NUM)
        loss = X.get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'img_pl': img_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_ne': pred_ne,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    total_correct = 0
    total_seen = 0
    fout_out_filelist = open(FLAGS.output_filelist, 'w')
    ##########################################

    for data_path in DATA_PATH:
        out_data_label_filename = os.path.basename(data_path)[:-4] + str(BLOCK_SIZE) +'box_feature_newdata_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = os.path.basename(data_path)[:-4] + str(BLOCK_SIZE) + 'box_feature_newdata_gt.txt'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
        print(data_path, out_data_label_filename)
        a, b = eval_one_epoch(sess, ops, data, label,img, data_path, out_data_label_filename, out_gt_label_filename)
        total_correct += a
        total_seen += b
        fout_out_filelist.write(out_data_label_filename+'\n')
    fout_out_filelist.close()
    log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, data, label,img, data_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    # if visu, produce obj file
    if FLAGS.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(data_path)[:-4]+str(BLOCK_SIZE) +'box_promerg_pred.obj'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(data_path)[:-4]+str(BLOCK_SIZE) +'box_promerg_gt.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w') # pred.txt
    fout_gt_label = open(out_gt_label_filename, 'w') # gt.txt
    
    #############################################################
    # current_data, current_label = indoor3d_util_xiu.room2blocks_wrapper_normalized(data_path, NUM_POINT)
    current_data, current_label,current_img = data, label, img

    #current_data, current_label, dup_num_list = preprocessing_tools1.scene2blocks_plus_normalized(data_label, num_point=NUM_POINT, block_size=BLOCK_SIZE, stride=BLOCK_SIZE, random_sample=False, sample_num=None, sample_aug=1)
    # print('right after scene2block' + str(current_data.dtype))
    current_data = current_data[:,0:NUM_POINT,:]
    
    print(current_data.shape)
    #current_label = np.squeeze(current_label)
    if RGB == 0:
      feature_num = 4
      current_data = np.delete(current_data, [4,5,6], 2)

    # print(current_data.dtype)
    ################
    
    meanx = np.mean(data[:, 0])
    stdx = np.std(data[:, 0])
    meany = np.mean(data[:, 1])
    stdy = np.std(data[:, 1])
    meanz = np.mean(data[:, 2])
    stdz = np.std(data[:, 2])
#####################################

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print (num_batches)
    print(file_size)
    print(BATCH_SIZE)

    #
    pred_list = []
    pred_eval_list = []
    xyzlpf_list = []  # save feature maps
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        print("data_shape" + str(current_data.shape))
        #print("img_data_shape" + str(img_data.shape))
        print("current_shape" + str(current_img.shape))
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['img_pl']: current_img[start_idx:end_idx,:,:,:],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        #loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],  feed_dict=feed_dict)
        loss_val, pred_val,pred_net = sess.run([ops['loss'], ops['pred_softmax'], ops['pred_ne']],  feed_dict=feed_dict)
        print ("pred_val:" + str(pred_val.shape))
        pred_eval_list.append(pred_val)
        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN
    #################
        # print('current data: ' + str(current_data.dtype))
        current_data_label = np.concatenate([current_data[start_idx:end_idx, :, :], np.expand_dims(current_label[start_idx:end_idx], -1)], -1)
        current_data_label_pred = np.concatenate([current_data_label[:, :, :], np.expand_dims(pred_label, -1).astype(np.float32)], -1)
        xyzlpf = np.concatenate([current_data[start_idx:end_idx,:,:3],np.expand_dims(current_label[start_idx:end_idx], -1), np.expand_dims(pred_label, -1), pred_net],-1)
        pred_list.append(current_data_label_pred)
        print ('xyzlpf: '  + str(xyzlpf.shape))
        xyzlpf_list.append(xyzlpf)
        print ('length xyzlpf: '  + str(len(xyzlpf_list)))

    xyzlpf_all =np.concatenate(xyzlpf_list, 0)
    print ('xyzlpf all: '  + str(xyzlpf_all.shape))
    np.save(str(train_model) + 'xyzlpf.npy', xyzlpf_all) #save  feature map
 
    pred_val_probability = np.concatenate(pred_eval_list, 0)
    data_label_pred = np.concatenate(pred_list, 0) # B x N x 12
    print('data_label_pred: ' + str(data_label_pred.shape))
    if save_probability == "Yes":
       np.save('pred_val_probability_' + str(train_model) + str(BLOCK_SIZE) + 'box.npy',pred_val_probability)
       np.save('data_for_prob'+ str(train_model)+ str(BLOCK_SIZE) + 'box.npy',data_label_pred)
    #########erase dup points####
    data_list = []

    # No dup list is used, meaning that duplicate values are also predicted
    for b in range(data_label_pred.shape[0]):
        data = np.reshape(data_label_pred[b, :, :], (data_label_pred.shape[1], -1))
        #if dup_num_list[b] != 0:
        #    data = data[0:-dup_num_list[b]]
        data_list.append(data)
    print(len(data_list))
    data_concated = np.concatenate(data_list, 0)
    print('shape of data_concated: ' + str(data_concated.shape))
    ######### per cls acc######
    gt_cls, gt_counts = np.unique(data_concated[:, -2], return_counts=True)#count: cls-wise total  number
    print(gt_cls)
    #pred_cls, pred_counts = np.unique(data_concated[:, -1], return_counts=True) # counts: cls-wise pred num
    #tp_cls_counts = [0 for _ in range(len(gt_cls))]
    tp_cls_counts = [0 for _ in range(len(gt_cls))]
    print (gt_cls.max())
    #print(gt_cls)
    print(gt_counts)
    #print(pred_cls)
    #print(pred_counts)
    np.save('../result_eval/' +str(train_model) + 'data_concated.npy',data_concated)

    for i in range(data_concated.shape[0]):
      if (data_concated[i, -2]==data_concated[i, -1]):
        tp_cls_counts[int(data_concated[i, -2])] += 1

    print (tp_cls_counts)
    ######accuracy####

    total_correct = np.sum(data_concated[:, -2] == data_concated[:, -1])
    # exclude class0 (unclassified)
    excl_uncls = np.count_nonzero(current_label[start_idx:end_idx] == 0)
    total_seen = data_concated.shape[0] - excl_uncls
    ########################visualize
    #log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    print (tp_cls_counts)
    print (gt_counts)
    log_string('eval accuracy: %f'% (float(sum(tp_cls_counts[1:8]))/float(sum(gt_counts[1:8]))))
    log_string ('tp_cls: %d,%d,%d,%d,%d,%d,%d' %(tp_cls_counts[1],tp_cls_counts[2],tp_cls_counts[3],tp_cls_counts[4],tp_cls_counts[5],tp_cls_counts[6],tp_cls_counts[7]))
    #log_string ('tp_cls: %d,%d' %(tp_cls_counts[1],tp_cls_counts[2]))
    #log_string ('gt_cls: %d,%d' %(gt_counts[1],gt_counts[2]))
    log_string ('gt_cls: %d,%d,%d,%d,%d,%d,%d' %(gt_counts[1],gt_counts[2],gt_counts[3],gt_counts[4],gt_counts[5],gt_counts[6],gt_counts[7]))
    for i in range(len(gt_cls)):
      log_string('class %d: %f'%(i, tp_cls_counts[i]/float(gt_counts[i])))
    fout_gt_label.close()
    if FLAGS.visu:
        fout.close()
        fout_gt.close()
    return total_correct, total_seen

end = time.time()
total_time = end-start
print("time" + str(total_time))
f.write("%s\n" % (start))
f.write("%s\n" % (end))
f.write("%s\n" % (total_time))

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
