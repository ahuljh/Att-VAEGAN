import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist

from sklearn import preprocessing
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.layers import Input, Dense, Lambda, merge, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import numpy as np
# import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
import glob, os
from sklearn.preprocessing import normalize
import random
import keras.backend.tensorflow_backend as KTF
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append(r'F:\代码项目\1115ZeroShot_CVAE-master\Disjoint\tools')
from StackingClassifier import StackingClassifier as SC

min_max_scaler = preprocessing.MinMaxScaler()
#配置系统的环境变量
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#获取Keras的session
def get_session(gpu_fraction=0.4):
    #占用40%的GP
    #从环境变量中得到线程数目
    num_threads = os.environ.get('OMP_NUM_THREADS')
    #定义GPU所占内存的比例
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    #如果可用线程数目不为0
    if num_threads:
        #返回多线程，其中占GPU比例为40%
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        #返回单线程，其中占GPU比例为40%
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


#tensorflow作为背景（tensorflow_backend KTF）
KTF.set_session(get_session())
#设置占用GP
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
########################################################################
#                           参数的设置                                 #
########################################################################
#m不知是干啥的
m = 50
#图像特征的维度
n_x = 2048
#语义特征的维度
n_y = 85
#噪声的维度
n_z = 50
#512=2048/4为隐藏层的维度
interNo = int(n_x/4)
#迭代次数
n_epoch = 25
#数据集的路径
path = '../../Datasets/xlsa17/data/AWA2/'
#生成样本类别个数
nSamples = 200
#训练类别的数目
nTrain = 40
#测试类别的数目
nTest = 10
randomDim = 135
adam = Adam(lr=0.0002, beta_1=0.5)
########################################################################
#                           CVAE的网络结构                             #
########################################################################
#定义输入的维度为2048+312维度，输入的为图像_类别拼接（语义信息/类原型）
input_ic = Input(shape=[n_x+n_y], name = 'img_class' )
#条件变量（CVAE中的C），维度为312维
cond  = Input(shape=[n_y] , name='class')
# print(type(input_ic))
#将图像_类别拼接输入进隐藏层，2048+312--->512
temp_h_q = Dense(interNo, activation='relu')(input_ic)
#512维度的隐藏层进行一次比例为0.7的Dropout
h_q_zd = Dropout(rate=0.7)(temp_h_q)
#将经过Dropout的神经元进行一次全连接层，并使用了relu激活函数，得到了512维度
h_q = Dense(interNo, activation='relu')(h_q_zd)
#对512维度的经过一次全连接，使用线性激活函数，得到了50维的噪声向量作为μ（均值为50维度）
mu = Dense(n_z, activation='linear')(h_q)
#对512维度的经过一次全连接，使用线性激活函数，得到了50维的噪声向量作为∑（方差为50维度）
log_sigma = Dense(n_z, activation='linear')(h_q)

#噪声的随机采样，传入方差、均值，返回最终得到的z
def sample_z(args):
    #从参数中获得均值方差
    mu, log_sigma = args
    #从标准正态分布中随机得到一个e
    eps = K.random_normal(shape=[n_z], mean=0., stddev=1.)
    #返回均值+方差*e，得到最终的z
    return mu + K.exp(log_sigma / 2) * eps
#使用Lambda匿名函数进行了函数的调用
z = Lambda(sample_z)([mu, log_sigma])

# Depending on the keras version...
# z_cond = merge([z, cond] , mode='concat', concat_axis=1)
#50维的噪声和312维的条件变量拼接，形成噪声_条件
z_cond = concatenate([z, cond])
#建立解码器的隐藏层，使用relu激活函数
decoder_hidden = Dense(1024, activation='relu')
#解码器的输出为全连接层+线性激活函数，输出维度为2048维
decoder_out = Dense(n_x, activation='linear')
#解码器的输入为50+312=362，输出维度为1024维
h_p = decoder_hidden(z_cond)
#解码器的输出为重建的图像，输入维度为1024维，输出维度为2048维
reconstr = decoder_out(h_p)
#定义编码器模型的输入为[图像_类别拼接312+2048，条件变量312维度]，输出为重建的维度2048维
vae = Model(inputs=[input_ic , cond], outputs=[reconstr])
#定义编码器的输入为[图像_类别拼接312+2048，条件变量312维度]，输出为重建的维度2048维
encoder = Model(inputs=[input_ic , cond], outputs=[mu])
#定义解码器的输入，维度为50+312=362维
d_in = Input(shape=[n_z+n_y])
#362维的向量经过解码器的隐藏层，输出为1024维度
d_h = decoder_hidden(d_in)
#1024维的向量经过解码器的输出层，输出维度为2048维
d_out = decoder_out(d_h)
#建立解码器模型，输入362维，输出为2048维
decoder = Model(d_in, d_out)

#变分自编码器的损失函数定义，输入参数为真实标签和预测的标签
def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    #重建损失为L2损失
    recon = K.mean(K.square(y_pred - y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    #KL散度的定义？？？？？？？？？？？？？
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    #print 'kl : ' + str(kl)
    return recon + kl

#输出编码器和解码器模型各层的参数
encoder.summary()
decoder.summary()
#编译自编码器的函数，定义了adam优化器和损失函数
vae.compile(optimizer='adam', loss=vae_loss)
########################################################################
#                           AttGAN的网络结构                           #
########################################################################
# 创建 Generator 操作序列
generator = Sequential()
generator.add(Dense(512, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
# generator.add(Dense(512))
# generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(2048, activation='tanh'))
# 编译 Generator 操作序列
generator.compile(loss='binary_crossentropy', optimizer=adam)
# 创建 Discriminator 操作序列
discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=2048, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
# 编译 Discriminator 操作序列
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
# Combined network
# 设置 Discriminator 不可训练
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
# 编译 GAN 网络流程
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
# def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
#     noise = np.random.normal(0, 1, size=[examples, randomDim])
#     print(noise.shape)
#     generatedImages = generator.predict(noise)
#     print(generatedImages.shape)
#     generatedImages = generatedImages.reshape(examples,1024)
#     print(generatedImages.shape)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

########################################################################
#                              获取数据                                #
########################################################################

np.random.seed(1000)

randomDim = 135
path = '../../Datasets/xlsa17/data/AWA1/'
#读取训练图像特征，应为n*2048的矩阵
trainData = np.load(open(path+'out/trainData' , 'rb'))
#读取训练的标签，应为n*1的向量
trainLabels = np.load(open(path+'out/trainLabels' , 'rb'))
#读取类原型，即类别对应的语义向量，如200*312或n*312
trainLabelVectors = np.load(open(path+'out/trainAttributes' , 'rb'))
testData = np.load(open(path+'out/testData','rb'))
testLabels = np.load(open(path+'out/testLabels','rb'))
testLabelVectors = np.load(open(path+'out/testAttributes' , 'rb'))
X_train = trainData
y_train=trainLabels
X_test=testData
y_test=testLabels
# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
########################################################################
#                              训练CVAE                                #
########################################################################
CVAE_X_train = np.concatenate([trainData , trainLabelVectors], axis=1)
print('Fitting VAE Model...')
#训练VAE模型，传入对应的参数vae = Model(inputs=[input_ic（n*312+2048） , cond(n*312)], outputs=[reconstr])，trainData为训练图像特征
vae.fit({'img_class' : CVAE_X_train , 'class' : trainLabelVectors}, trainData, batch_size=m, nb_epoch=n_epoch)

 #打开所有的类别文件
fp = open(path + 'allclasses.txt' ,'rb')
#读取所有的类别
allclasses = [x.split()[0] for x in fp.readlines()]
fp.close()

from scipy.io import loadmat
CUB=loadmat('../../Datasets/xlsa17/data/AWA1/AWA.mat')
testClasses=CUB["testclasses"]
trainClasses=CUB["trainclasses"]

#获取所有的属性

Data = loadmat('../../Datasets/xlsa17/data/AWA1/att_splits.mat')
ATTR =Data["att"]
ATTR=np.transpose(ATTR)

ATTR.shape

#启动session
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# ==================================================

#虚假的训练数据和标签及属性
pseudoTrainData = []
pseudoTrainLabels =[]
pseudoTrainAttr = []

#生成的训练未见类别数目50*200=10000
totalExs = len(testClasses)*nSamples
##########啊哈哈哈哈
with sess.as_default():
    #得到生成的噪声10000*50
	noise_gen = K.random_normal(shape=(totalExs, n_z), mean=0., stddev=1.).eval()

#测试类别有问题！！！！
#在测试类别中不存在
#[4,63,159]
tl=[]
for  i in testClasses:
    for j in i:
        tl.append(j)

testClasses=tl
#50个测试类
#50个类*200个图=10000
for tc in testClasses:
    for ii in range(0,nSamples):
        pseudoTrainAttr.append(ATTR[tc-1])
        pseudoTrainLabels.append(tc)
pseudoTrainAttr = np.array(pseudoTrainAttr)
print(pseudoTrainAttr)
print(pseudoTrainLabels)
pseudoTrainLabels = np.array(pseudoTrainLabels)
#10000*312
#共10000个生成的样本[ 43  43  43 ... 168 168 168]
#解码拼接
dec_ip = np.concatenate((noise_gen, pseudoTrainAttr) , axis=1)
###########################################################
#解码器进行预测得到的标签
pseudoTrainData = decoder.predict(dec_ip)
# pseudoTrainData = min_max_scaler.fit_transform(pseudoTrainData)
print(pseudoTrainData)

tl=[]
for  i in testLabels:
    for j in i:
        tl.append(j)

testLabels=tl


########################################################################
#                              训练AttGAN                              #
########################################################################
epochs=5
batchSize=128
#批次数量，如8832/128=69次
batchCount = int(X_train.shape[0] / batchSize)
print('Epochs:', epochs)
print('Batch size:', batchSize)
print('Batches per epoch:', batchCount)

for e in range(1, epochs+1):
    print('-'*15, 'Epoch %d' % e, '-'*15)
    #从1到468
    for _ in tqdm(range(batchCount)):
        #128个50维的噪声
        with sess.as_default():
            # 得到生成的噪声10000*50
            noise = K.random_normal(shape=(batchSize, n_z), mean=0., stddev=1.).eval()
        # noise = np.random.normal(0, 1, size=[batchSize, 50])
        #X_train[0-60000的随机噪声，大小为128]
        #从X_train所有的训练图像特征（6000*2048）中，60000张中随机生成128项，作为所需要进行训练的imageBatch128张批次图片。
        num=np.random.randint(0, X_train.shape[0], size=batchSize)
        imageBatch = X_train[num]
        #用Generator 生成假数据
        label_x=y_train[num]
        att_x=trainLabelVectors[num]
        att_noise=np.column_stack((att_x,noise))
        generatedImages = generator.predict(att_noise)
        # 将假数据与真实数据进行混合在一起
        X=np.concatenate([imageBatch, generatedImages])
        # Labels for generated and real data
        # 标记所有数据都是假数据
        yDis = np.zeros(2*batchSize)

        # 按真实数据比例，标记前半数据为 0.9 的真实度
        yDis[:batchSize] = 0.9

        # Train discriminator
        # 先训练 Discriminator 让其具有判定能力，同时Generator 也在训练，也能更新参数。
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(X, yDis)

        # Train generator训练生成器
        # 然后训练 Generator, 注意这里训练 Generator 时候，把 Generator 生成出来的结果置为全真，及按真实数据的方式来进行训练。
        # 先生成相应 batchSize 样本 noise 数据
        noise = np.random.normal(0, 1, size=[batchSize, 50])
        att_noise= np.column_stack((att_x, noise))
        # 生成相应的 Discriminator 输出结果
        yGen = np.ones(batchSize)
        # 将 Discriminator 设置为不可训练的状态
        discriminator.trainable = False
        # 训练整个 GAN 网络即可训练出一个能生成真实样本的 Generator
        gloss = gan.train_on_batch(att_noise, yGen)
        # Store loss of most recent batch from this epoch
    dLosses.append(dloss)
    gLosses.append(gloss)

    if e == 1 or e % 20 == 0:
        # plotGeneratedImages(e)
        saveModels(e)
#开始生成测试类别对应的样本
#迭代5次训练好生成器后，为各个类别生成200个样本，存入训练集合


pseudoTrainLabels=list(pseudoTrainLabels)
for i in range(5):
    print('-' * 15, 'Epoch %d' % i, '-' * 15)
    #生成了468次128批次的数据。
    #样本需要重新获取
    #测试类的类标签及对应的属性向量，输入属性向量及噪声
    #输入后得到生成的图像特征与对应的类别标签存储进伪训练集
    for _ in tqdm(range(5)):
        with sess.as_default():
            # 得到生成的噪声10000*50
            noise = K.random_normal(shape=(batchSize, n_z), mean=0., stddev=1.).eval()
        # noise = np.random.normal(0, 1, size=[batchSize, 50])
        num = np.random.randint(0, X_test.shape[0], size=batchSize)
        att_x = testLabelVectors[num]
        label_x = y_test[num]
        # print(label_x)
        #128
        att_noise = np.column_stack((att_x, noise))
        #生成相应的Discriminator输出结果
        yGen = np.ones(batchSize)
        generatedImages = generator.predict(att_noise)

        generatedImages = min_max_scaler.fit_transform(generatedImages)
        pseudoTrainData=np.row_stack((pseudoTrainData,generatedImages))
        tl = []
        for i in label_x:
            for j in i:
                tl.append(j)

        label_x =np.array(tl)
        # print(len(label_x))
        for kk in label_x:
            pseudoTrainLabels.append(kk)
        pseudoTrainAttr=np.row_stack((pseudoTrainAttr,att_x))

        # 将Discriminator设置为不可训练的状态

        discriminator.trainable=False
        # 训练整个GAN 网络即可训练出一个能生成真实样本的 Generator
        gloss = gan.train_on_batch(att_noise, yGen)
    gLosses.append(gloss)


#对生成的图像和测试的图像进行正则化，放到最后

# print(pseudoTrainLabels.shape)
pseudoTrainLabels=np.array(pseudoTrainLabels)
#正则化
pseudoTrainData = normalize(pseudoTrainData , axis =1)
testData = normalize(testData , axis=1)

#arrayname=np.concatenate(arrayname, axis=0)
print(pseudoTrainAttr)
print(pseudoTrainLabels)
print(pseudoTrainData)

print(testLabels)
# testData = min_max_scaler.fit_transform(testData)
print(testData)
###########################################################
# #训练分类器
# #已经准备好了训练样本（x,y）
# print('Training SVM-100')
# clf5 = svm.SVC(C=100)
# clf5.fit(pseudoTrainData, pseudoTrainLabels)
# print('Predicting...')
# #对于测试的数据得到测试的标签
# pred = clf5.predict(testData)
# #得到SVC支持向量分类的结果
# print(accuracy_score(testLabels , pred))

#得到KNN分类的结果
print('Training KNN-100')
knn_clf=KNeighborsClassifier()
knn_clf.fit(pseudoTrainData,pseudoTrainLabels)
print('Predicting...')
pred=knn_clf.predict(testData)
# print(accuracy_score(testLabels , pred))
print(knn_clf.score(testData,testLabels))

#得到逻辑回归的结果
print('Training LogisticRegression-100')
lr=LogisticRegression()
lr.fit(pseudoTrainData,pseudoTrainLabels)
print('Predicting...')
pred=lr.predict(testData)
print(accuracy_score(testLabels , pred))
# print(lr.score(testData,testLabels))

#集成多个分类器的过程
base_classifiers=[knn_clf,lr]
stacking_clf=SC(base_classifiers)
stacking_clf.fit(pseudoTrainData,pseudoTrainLabels)
print('Stacking classifier accuracy: %s' % stacking_clf.score(testData, testLabels))
allTestClasses = sorted(list(set(testLabels)))
dict_correct = {}
dict_total = {}

for ii in allTestClasses:
	dict_total[ii] = 0
	dict_correct[ii] = 0

for ii in range(0,len(testLabels)):
	if(testLabels[ii] == pred[ii]):
	    dict_correct[testLabels[ii]] = dict_correct[testLabels[ii]] + 1
	dict_total[testLabels[ii]] = dict_total[testLabels[ii]] + 1

avgAcc = 0.0
for ii in allTestClasses:
	avgAcc = avgAcc + (dict_correct[ii]*1.0)/(dict_total[ii])

avgAcc = avgAcc/len(allTestClasses)
print('Average Class Accuracy = ' + str(avgAcc))

#1、对生成的特征进行t-SNE可视化展示（之前做的现有的可视化，现在做生成的可视化）
#GAN的可视化、CVAE的可视化
#2、混淆矩阵的展示，选取10个类别
#3、AWA数据集继续