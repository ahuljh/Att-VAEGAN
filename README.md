# Att-VAEGAN
The paper's code of Att-VAEGAN

## 1.Download Dataset（878.98M） of Zero-shot Learning.
link is from https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip

## 2.Data preprocess
MATLAB:获取到对应数据集的mat文件

(1)getrighttxt.m---获得allclasses.txt、testclasses.txt文件【CUB中自带这几个txt文件，不需要运行，其他数据集需要运行】
(2)ReadTrainTest.m---【读取allclasses.txt、testclasses.txt】提取类别编号，获取可见类、未见类、全部类的编号，并存入trainANDtestClass.mat文件
(3)ExtractClassFeatureAndAttribute.m---【读取trainANDtestClass.mat、res101.mat、att_splits.mat】提取训练类与测试类样本及属性并存入XXX.mat文件
coding(4)提取词向量并存入对应文件

Python:
(1)MyDataRead.py，将从MATLAB得到的mat文件转换为Python支持的数据文件格式（CVAE需要的数据格式）

## 3.Data generation and classification
Python:
(1)train_CVAE-AttGAN.py，训练及分类，并将得到的数据特征存储
(2)真实特征与生成特征的t-SNE可视化展示。语义的展示
(3)分类的混淆矩阵
(4)广义零样本学习
(5)小样本学习
(6)零样本检索
