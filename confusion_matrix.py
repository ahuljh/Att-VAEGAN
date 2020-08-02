import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix of classifier prediction',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from scipy.io import loadmat

m = loadmat("confusion.mat")
ls=m["ls"]
ps=m["ps"]
classes=m["classes"]
ls=np.array(ls).transpose()
ps=np.array(ps).transpose()
classes=np.array(classes).transpose()

tl = []
for i in classes:
    for j in i:
        tl.append(j)
classes =np.array(tl)


print(ls.shape)
print(ps.shape)
print(classes.shape)
mat = confusion_matrix(ls, ps)
print(mat)
# sns.heatmap(mat.T, square=True, annot=True, fnt='d', cbar=False, xticklabels=classes, yticklabels=classes)
plt.figure()
plot_confusion_matrix(mat, classes=classes, title='Confusion matrix of classifier prediction')
plt.show()