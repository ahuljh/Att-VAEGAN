import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib
matplotlib.use("agg")
import numpy as np
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
sys.path.append(r'F:\代码项目\1115ZeroShot_CVAE-master\Disjoint\tools')
########################################################################
#                              获取数据                                #
########################################################################
path = '../../Datasets/xlsa17/data/AWA1/'
testData = np.load(open(path+'out/testData','rb'))
testLabels = np.load(open(path+'out/testLabels','rb'))
#获取所有的属性

data = testData
tl=[]
for  i in testLabels:
    for j in i:
        tl.append(j)

testLabels=tl
labels = np.array(testLabels)

print(labels)
print(type(labels))
print(labels.shape)
print(data.shape)
n_samples,n_features = data.shape
tsne = TSNE(n_components=2,init='pca',random_state=0)
t0 = time()
result = tsne.fit_transform(data)
result

x_min,x_max = np.min(result,0),np.max(result,0)
data = (result-x_min)/(x_max-x_min)

fig = plt.figure()
# 表示画布整个输出，不分割成小块区域，图形直接输出在整块画布上
ax = plt.subplot(111)
# print(data.shape[0])

#[7,9,23,24,30,31,34,41,47,50]
for i in range(data.shape[0]):
    print(labels[i])
    print(data[i, 0])
    print(data[i, 1])
    print('\n')
    if labels[i]==7:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set1(1),fontdict={'weight':'bold','size':9})
    if labels[i]==9:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set1(2),fontdict={'weight':'bold','size':9})
    if labels[i]==23:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set1(3),fontdict={'weight':'bold','size':9})
    if labels[i]==24:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set1(4),fontdict={'weight':'bold','size':9})
    if labels[i]==30:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set1(5),fontdict={'weight':'bold','size':9})
    if labels[i]==31:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set2(1),fontdict={'weight':'bold','size':9})
    if labels[i]==34:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set2(2),fontdict={'weight':'bold','size':9})
    if labels[i]==41:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set2(3),fontdict={'weight':'bold','size':9})
    if labels[i]==47:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set2(4),fontdict={'weight':'bold','size':9})
    if labels[i]==50:
        plt.text(data[i,0],data[i,1],str(labels[i]),color=plt.cm.Set2(5),fontdict={'weight':'bold','size':9})
plt.xticks([])
plt.yticks([])
plt.title('t-SNE embedding of the digits')
plt.show()

###########################################################

