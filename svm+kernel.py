import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm,datasets,metrics
import random


#load data iris and remove target 0
iris = datasets.load_iris()
num_0 = list(i for i in iris["target"]).index(1)
data = iris['data'][num_0:]
data = data[:,:2]
target = iris['target'][num_0:]
#split data and target into training data (50%) and testing data (50%).
n_sample = len(data)
training_data = np.vstack((data[:n_sample//4,:],data[n_sample//4*2:n_sample//4*3,:]))
training_target = np.hstack((target[:n_sample//4],target[n_sample//4*2:n_sample//4*3]))
testing_data = np.vstack((data[n_sample//4:n_sample//4*2,:],data[n_sample//4*3:,:]))
testing_target = np.hstack((target[n_sample//4:n_sample//4*2],target[n_sample//4*3:]))

fignum = 1
# 3 different kernel SVM
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel,gamma=10)
    clf.fit(training_data, training_target)
    expected = testing_target
    predicted = clf.predict(testing_data)
    #print(metrics.classification_report(expected, predicted))
    # plot the line, the points
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.scatter(training_data[:, 0], training_data[:, 1], c=training_target, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k',label='training')
    plt.scatter(testing_data[:,0],testing_data[:,1],c=testing_target,marker='D',zorder=10, cmap=plt.cm.Paired,
                edgecolors='k',label='testing')
    plt.axis('tight')
    plt.legend()
    x_min = 4.8
    x_max = 8
    y_min = 1.9
    y_max = 3.9
    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.title("Kernel: "+kernel)
    fignum = fignum + 1

plt.show()
