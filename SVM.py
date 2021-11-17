__author__ = "Amir Mallaei"
__email__ = "amirmallaei@gmail.com"

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Read the Original Image
img = np.array(Image.open('Images/1.jpg'))
size_img = img.shape

# Read the Masks
water = np.array(Image.open('Images/water.jpg'))
green = np.array(Image.open('Images/green.jpg'))
urban = np.array(Image.open('Images/urban.jpg'))

# Seperating 10000 Sample of Water Section vs Other Samples to Label
# and Features and againg Green Section vs other Samples
Labels = []
Features = []
Labels1 = []
Features1 = []
new_img = []
counter2 = 0
counter4 = 0
counter21 = 0
counter41 = 0

for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        x = np.zeros((1, 3))
        x[0, 0] = img[i, j, 0]
        x[0, 1] = img[i, j, 1]
        x[0, 2] = img[i, j, 2]
        new_img.append([x[0, 0], x[0, 1], x[0, 2]])
        if water[i, j] > 0:
            if counter2 <= 10000:
                counter2 += 1
                Labels.append(2)
                Features.append([x[0, 0], x[0, 1], x[0, 2]])
        else:
            if counter4 <= 10000:
                counter4 += 1
                Labels.append(4)
                Features.append([x[0, 0], x[0, 1], x[0, 2]])

        if green[i, j] > 0:
            if counter21 <= 10000:
                counter21 += 1
                Labels1.append(2)
                Features1.append([x[0, 0], x[0, 1], x[0, 2]])
        else:
            if counter41 <= 10000:
                counter41 += 1
                Labels1.append(4)
                Features1.append([x[0, 0], x[0, 1], x[0, 2]])

# Convert Features to numpy Array and split it to Train and Test sample
# for Water Section
X = np.asarray(Features)
y = np.asarray(Labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

# Apply SVM Classifier with linear kernel for Water Section
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

# Test the Classifier and Display Result For Water Section
y_predict = classifier.predict(X_test)
print(classification_report(y_test, y_predict))

# Apply the Classifier's Prediction to whole picture
img_predict = classifier.predict(new_img)

# Convert Prediction to 3D image and filling the Blue channel
# Based on Water Prediction
pic = np.zeros(size_img)
my_img = img_predict.reshape((size_img[0], size_img[1], 1))
for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        if my_img[i, j] == 2:
            pic[i, j, 2] = 255

# Convert Features to numpy Array and split it to Train and Test sample
# for Green Section
X1 = np.asarray(Features1)
y1 = np.asarray(Labels1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33,
                                                        random_state=42)

# Apply SVM Classifier with linear kernel for Green Section
classifier1 = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier1.fit(X_train1, y_train1)

print('--------------------------------------------------')
# Test the Classifier and Display Result For Green Section
y_predict1 = classifier.predict(X_test1)
print(classification_report(y_test1, y_predict1))

# Apply the Classifier's Prediction to whole picture
img_predict1 = classifier1.predict(new_img)

# Convert Prediction to 3D image and filling the Green and Red channels
# Based on green Prediction
my_img1 = img_predict1.reshape((size_img[0], size_img[1], 1))
for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        if my_img1[i, j] == 2:
            pic[i, j, 1] = 255
        else:
            pic[i, j, 0] = 255

# Display the Whole Prediction Result
plt.imshow(pic)
plt.show()


# Display Results
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
ax1.title.set_text('Water Section')
ax1.imshow(pic[:, :, 2], cmap='gray')
ax2.title.set_text('Green Section')
ax2.imshow(pic[:, :, 1], cmap='gray')
ax3.title.set_text('Urban Section')
ax3.imshow(pic[:, :, 0], cmap='gray')
ax4.imshow(pic)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
plt.show()
