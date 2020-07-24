import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

# Load dataset
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# separate first half of the digits as learning data
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# use the second half of the digits as testing data
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

# show results of classification report
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

#show first 10 images of prediction
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:10]):
    plt.subplot(2, 5, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()