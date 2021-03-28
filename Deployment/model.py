import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.losses import binary_crossentropy

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1_basic = Conv2D(filters=filters, kernel_size=(3, 3),strides=stride, padding="same", activation = "relu")
        self.batchNorm1_basic = BatchNormalization()
        self.conv2_basic = Conv2D(filters=filters,kernel_size=(3, 3),strides=1,padding="same", activation = "relu")
        self.batchNorm2_basic = BatchNormalization()
        if stride != 1:
            self.downsample_basic = tf.keras.Sequential()
            self.downsample_basic.add(Conv2D(filters=filters, kernel_size=(1, 1), strides=stride,  activation = "relu"))
            self.downsample_basic.add(BatchNormalization())
        else:
            self.downsample_basic = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample_basic(inputs)

        x = self.conv1_basic(inputs)
        x = self.batchNorm1_basic(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2_basic(x)
        x = self.batchNorm2_basic(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block

class UNETBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1_unet = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv1_1_unet = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv1_2_unet = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv1_3_unet = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.pool1_unet = MaxPooling2D(pool_size=(2, 2))

        self.conv2_unet = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv2_1_unet = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv2_2_unet = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv2_3_unet = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.conv3_unet = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv3_1_unet = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv3_2_unet = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv3_3_unet = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.conv4_unet = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv4_1_unet = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv4_2_unet = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv4_3_unet = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.drop1_unet = Dropout(0.2)
        self.conv5_unet = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv5_1_unet = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.up1_unet = UpSampling2D(size = (2,2))
        self.conv6_unet = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv6_1_unet = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        
        self.conv7_unet = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv8_unet = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv9_unet = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv10_unet = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv11_unet = Conv2D(1, 1, activation = 'sigmoid')
    
    def call(self, inputs):
        x = self.conv1_unet(inputs)
        conv_1 = self.conv1_1_unet(x)
        x = self.pool1_unet(conv_1)
        x = self.conv2_unet(x)
        conv_2 = self.conv2_1_unet(x)
        x = self.pool1_unet(x)
        x = self.conv3_unet(x)
        conv_3 = self.conv3_1_unet(x)
        x = self.pool1_unet(conv_3)
        x = self.conv4_unet(x)
        x = self.conv4_1_unet(x)
        drop_1 = self.drop1_unet(x)
        x = self.pool1_unet(drop_1)
        
        x = self.conv5_unet(x)
        x = self.conv5_1_unet(x)
        drop_2 = self.drop1_unet(x)
        
        x = self.conv6_unet(self.up1_unet(drop_2))
        x = concatenate([drop_1, x], axis = -1)

        x = self.conv4_2_unet(x)
        x = self.conv4_3_unet(x)
        
        x = self.conv7_unet(self.up1_unet(x))
        x = concatenate([conv_3, x], axis = -1)
        x = self.conv3_2_unet(x)
        x = self.conv3_3_unet(x)
        
        x = self.conv8_unet(self.up1_unet(x))
        x = concatenate([conv_2, x], axis = -1)
        x = self.conv2_2_unet(x)
        x = self.conv2_3_unet(x)
        
        x = self.conv9_unet(self.up1_unet(x))
        x = self.conv1_2_unet(x)
        x = self.conv1_3_unet(x)
        x = self.conv10_unet(x)
        output = self.conv11_unet(x)
        
        return output
                                   
                                   
class UnetResnet34(tf.keras.Model):
    def __init__(self, input_shape, layer_params):
        super().__init__()

        self.conv1_resnet = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1_resnet = tf.keras.layers.BatchNormalization()
        self.pool1_resnet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1_resnet = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[0])
        self.layer2_resnet = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3_resnet = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4_resnet = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool_resnet = tf.keras.layers.AveragePooling2D()
        
        self.unetBlock_resnet = UNETBlock()
        
        self.upsample_resnet = tf.keras.layers.UpSampling2D(size = (64, 64))
        self.conv1_resize = tf.keras.layers.Conv2D(filters=1,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_resnet(inputs)
        x = self.bn1_resnet(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_resnet(x)
        x = self.layer1_resnet(x, training=training)
        x = self.layer2_resnet(x, training=training)
        x = self.layer3_resnet(x, training=training)
        x = self.layer4_resnet(x, training=training)
        x = self.avgpool_resnet(x)
        x = self.upsample_resnet(x)
        x = self.conv1_resize(x)
        output = self.unetBlock_resnet(x)

        return output

class Resnet34(tf.keras.Model):
    def __init__(self, input_shape, layer_params):
        super().__init__()

        self.conv1_resnet = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1_resnet = tf.keras.layers.BatchNormalization()
        self.pool1_resnet = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1_resnet = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[0])
        self.layer2_resnet = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3_resnet = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4_resnet = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool_resnet = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(300, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(50, activation = 'relu')
        self.final = tf.keras.layers.Dense(1, activation= 'sigmoid')
        

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_resnet(inputs)
        x = self.bn1_resnet(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_resnet(x)
        x = self.layer1_resnet(x, training=training)
        x = self.layer2_resnet(x, training=training)
        x = self.layer3_resnet(x, training=training)
        x = self.layer4_resnet(x, training=training)
        x = self.avgpool_resnet(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.final(x)
        return output
        
def buildPredictor():
    model = UnetResnet34((256, 256, 3),[3, 4, 6, 3])
    model.build([None, 256, 256, 3])
    return model
    
def buildClassifier():
    model = Resnet34((256, 256, 3),[3, 4, 6, 3])
    model.build([None, 256, 256, 3])
    return model