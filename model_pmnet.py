#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import tensorflow.contrib.slim as slim
def placeholder_inputs(batch_size, num_point, feature_num):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, feature_num))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

#placeholder for image

def placeholder_img(batch_size,num_pix, RGB):
    img_pl = tf.placeholder(tf.float32, shape=(batch_size,num_pix,num_pix, RGB))
    return img_pl


def get_point_corres_features(box_size, pixel_size,feature_img, point_cloud):
  
  #feature_pcl = tf.squeeze(point_cloud, [-2])
  feature_pcl = point_cloud
  print (feature_pcl.shape)
  size = int(box_size/pixel_size) 
  batch = feature_pcl.get_shape()[0]
  num_points = feature_pcl.get_shape()[1]
  print ("num_points: "+ str(num_points))
  print ("batch: "+ str(batch))

  # replace x  and y  for correct indexing  
  feature_y = feature_pcl[:,:,1]
  feature_x = feature_pcl[:,:,0]
  feature_index  =  tf.concat(axis =-1, values = [tf.expand_dims(feature_y,-1),tf.expand_dims(feature_x,-1)]) #  just shift the column location of x  and y

  feature_index = tf.multiply(feature_index,tf.constant([20.0], dtype=tf.float32))
  feature_index = tf.cast(feature_index, tf.int32)
  batch_list = []
  for s in range(batch):
    result = tf.gather_nd(feature_img[s,], feature_index[s,])
    batch_list.append(result)
  batch_out = tf.stack(batch_list)
  print ("batch_out: "+ str(batch_out.shape))
  batch_out = tf.expand_dims(batch_out,axis=-2)
  print ("batch_out_expand: "+ str(batch_out.shape))
  return batch_out

def get_model(point_cloud, img, is_training, num_cls, feature_num,  bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    # b x n x 10
    
    #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    regularizer = tf.keras.regularizers.l2(l=0.1)
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    #image pixel size
    num_pix = img.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    #CONV for RGB images
    conv1 = tf.layers.conv2d(inputs=img,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)   
    print (conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    #conv = tf.layers.conv2d(inputs=pool,filters=32,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
    conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
    print( conv4.shape)
   
    Up1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv4)
    print (Up1.shape)
    dconv1 = tf.layers.conv2d(inputs=Up1,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    con1 =tf.concat(axis=-1, values=[dconv1, conv2])
    dconv2 = tf.layers.conv2d(inputs=con1,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    Up2 = tf.keras.layers.UpSampling2D(size=(2,2))(dconv2)
    dconv3 = tf.layers.conv2d(inputs=Up2,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    con2 =tf.concat(axis=-1, values=[dconv3, conv1])
    feature_img = tf.layers.conv2d(inputs=con2,filters=128,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    print (feature_img.shape)


    # CONV pointcloud
    print ("input_image.shape: "+ str(input_image.shape))
    #net = tf_util.conv2d(input_image, 64, [1,feature_num], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv0', bn_decay=bn_decay)

    net = tf.keras.layers.Convolution2D(filters=64, kernel_size=[1,feature_num], strides=(1, 1), padding='valid', activation='elu', use_bias=True, kernel_initializer='random_uniform')(input_image)

    net = tf.keras.layers.Convolution2D(filters=64, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(net)
    print("second_layer pointcloud: " + str(net.shape))

    #net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)

    print(net.shape)

    #net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)

    net_test = tf.keras.layers.Convolution2D(filters=128, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(net)

    #net_test= tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)

    #points_feat1  = tf_util.conv2d(net_test, 1024, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)

    points_feat1 = tf.keras.layers.Convolution2D(filters=1024, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(net_test)

    print("points_feat" + str(points_feat1.shape))

    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    print("mapool1: " + str(pc_feat1.shape))
    
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    print(pc_feat1.shape)
    
    print (pc_feat1.shape)
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    print (points_feat1.shape)
    print("pc_feeat1: " + str(pc_feat1_expand.shape))

    #dense = tf.tile(tf.reshape(dense, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    #print(dense.shape)


    points_feat1_concat = tf.concat(axis=-1, values=[net_test,pc_feat1_expand])
    print(points_feat1_concat.shape)

    #net = tf_util.conv2d(points_feat1_concat, 128, [1,1], padding='VALID', stride=[1,1],
    #                    bn=True, is_training=is_training, scope='conv7')

    net = tf.keras.layers.Convolution2D(filters=128, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(points_feat1_concat)

    # point-wise concatanation of features between image and pointcloud from the first layer
    #point_corres_featur1 = get_point_corres_features(box_size=50, pixel_size=0.16,feature_img=conv1, point_cloud=point_cloud)

    # point-wise concatanation of features between image and pointcloud final layer
    point_corres_featur = get_point_corres_features(box_size=15.6, pixel_size=0.05,feature_img=feature_img, point_cloud=point_cloud)
    point_corres_featur = tf.concat(axis=-1, values=[point_corres_featur,net])
    print ("point_corres_featur" + str(point_corres_featur.shape) )


    #net = tf_util.conv2d(point_corres_featur, 256, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv8')


    net = tf.keras.layers.Convolution2D(filters=256, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(point_corres_featur)
    #net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv9')
    net_print = tf.squeeze(net, [2])
    net = tf.keras.layers.Convolution2D(filters=128, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(net)

    print(net.shape)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp1')
    
    #net = tf_util.conv2d(net, num_cls , [1,1], padding='VALID', stride=[1,1],
    #                     activation_fn=None, scope='conv10')

    net = tf.keras.layers.Convolution2D(filters=num_cls, kernel_size=[1,1], strides=(1, 1), padding='valid', activation='elu', use_bias=True)(net)
    print(net.shape)
    net = tf.squeeze(net, [2])
    print(net.shape)
    model_vars = tf.trainable_variables()
    print (slim.model_analyzer.analyze_vars(model_vars, print_info=True))
    return (net, net_print)

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    # specify some class weightings
    #class_weights = tf.constant([0, 0.7, 0.7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    class_weights = tf.constant([0,1.0,1.0,1.0,0.7,1.0,1.0,1.0])
    # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
    
    weights = tf.gather(class_weights, label)
    # compute the loss
    loss = tf.losses.sparse_softmax_cross_entropy(label, pred, weights)
    #loss = tf.losses.sparse_softmax_cross_entropy(label, pred)
    #return tf.reduce_mean(loss)
    return loss

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,4096,9))
        #a = tf.compat.v1.placeholder(tf.float32, shape=(32,4096,9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            init = tf.global_variables_initializer() #tf.compat.v1.zeros_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,4096,9)})
            print(time.time() - start)
