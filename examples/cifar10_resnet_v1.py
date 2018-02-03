# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'ooo'
__date__ = ' 15:34'

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l2

import numpy as np
import os


#####################
#    指定训练参数
#####################
# 训练参数
epochs = 200
batch_size = 32
num_classes = 10
data_augmentation = False
subtract_pixel_mean = True


# 学习率策略
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# 版本 深度 名称
version = 1
n = 3
depth = n * 6 + 2

model_type = 'ResNet%dv%d' % (depth, version)

#####################
#    加载训练数据
#####################

# 加载cifar10数据
root_dir = os.getcwd()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 输入图片的维数
input_shape = x_train.shape[1:]

# 数据归一化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 数据去均值
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# 转换类标格式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#####################
#    构造和编译模型
#####################

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """
    :param inputs:
    :param num_filters:
    :param kernel_size:
    :param strides:
    :param activation:
    :param batch_normalization:
    :param conv_first:

    :return: x (tensor): tensor as input to the next layer
    """

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def build_resnet_v1(input_shape, depth, num_calsses=10):
    """
    :param input_shape:
    :param depth:
    :param num_calsses:
    :return:
    """
    if (depth % 2) != 0:
        raise ValueError('深度必须是2的整数倍')
    # 开始定义模型
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # 构造残差网络的堆叠单元
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2 # 降采样
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_calsses,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # 初始化模型 在Container中添加训练和测试中的数据处理事项
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_resnet_v1(input_shape=input_shape,depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

model.summary()


#####################
#    设置存储路径
#####################
save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


#####################
#    设置回调函数
#####################
check_point = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only=True)

lr_schedule = LearningRateScheduler(schedule=lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

tensorboard = TensorBoard(log_dir='E:\\keras\\alldatasets\\cifar10_resnet_result',
                          write_grads=False,
                          write_graph=True,
                          write_images=False)

callbacks = [check_point, lr_schedule, lr_reducer, tensorboard]


#######################
#   开始运行训练程序
#######################
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # 将数据集的整体均值设为0
        featurewise_center=False,
        # 将每个样本的均值设为0
        samplewise_center=False,
        # 对数据集整体除以std
        featurewise_std_normalization=False,
        # 对每个样本除以std
        samplewise_std_normalization=False,
        # 进行ZCA白化变换
        zca_whitening=False,
        # 随机旋转0-180°
        rotation_range=0,
        # 水平随机移动
        width_shift_range=0.1,
        # 垂直随机移动
        height_shift_range=0.1,
        # 水平翻转图片
        horizontal_flip=True,
        # 处置翻转图片
        vertical_flip=False)
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
####################
# 模型评估
####################
scores = model.evaluate(x=x_test, y=y_test, verbose=1)
print('Test Loss:', scores[0])
print('Test Accuracy:', scores[1])

