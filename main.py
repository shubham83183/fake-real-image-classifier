"""
Name : Shubham Shivaji Suryavanshi
Matriculation No: 1537745
Task Description:
1. To classify the real and fake images, where the fake images have been generated using SNGAN
 using different up sampling techniques:
   * Bilinear interpolation
   * Bicubic interpolation
   * Pixel shuffle
2. Is the model working for other up sampling techniques?
3. Can we get better results by using vertical and horizontal projections instead of
radial ones, when images are transformed in frequency domain?

"""


import torchvision.transforms as transforms
import os
from torchvision.datasets import ImageFolder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from data_preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
#  Before running code mention the directory if you want to run on other datasets
# data_dir_train = 'E:\\Program Files\\sem 6\\raml\\exp'
# data_dir_test = 'E:\\Program Files\\sem 6\\raml\\exp'

# train_dataset = ImageFolder(os.path.join(data_dir_train, 'train'), transform=transform)
# test_dataset = ImageFolder(os.path.join(data_dir_test, 'test'), transform=transform)

train_dataset = ImageFolder('train', transform=transform)
test_dataset = ImageFolder('test', transform=transform)
#  preprocessing data,x1,x2,x3 are tuples containing real and fake image
#  x1=(imagewoof, bicubic), x2=(imagewoof, bilinear), x3=(imagewoof, pixelshuffle)
#  xt1, xt2, xt3 are testing tuple
print(f'Labels based on folders {train_dataset.class_to_idx}', '\n')
print(f'Labels based on folders {test_dataset.class_to_idx}', '\n')

# Mention the method= "unrolled_xy", method= "average_xy", method= "azimuthal_avg"
# Default method is method= "unrolled_xy"
x1, x2, x3 = preprocess(train_dataset, method="unrolled_xy")
xt1, xt2, xt3 = preprocess(test_dataset, method="unrolled_xy")

print('Size of the image ' + str(train_dataset[0][0].shape))
print('Size of preprocessed training input' + str(x1[0].shape))
print('Size of preprocessed training output' + str(x1[1].shape), '\n')
print('Size of preprocessed test input ' + str(xt1[0].shape))
print('Size of preprocessed test output' + str(xt1[1].shape), '\n')
#################################################################################

# C=2 for azimuthal_avg and unrolled_xy, C=6.5 for average_xy method, other hyper parameters
# remains same
#  SVM classifier for imagewoof and bicubic
svc_classifier = SVC(C=2, kernel='rbf', gamma='scale')
svc_classifier.fit(x1[0], x1[1])
svm_score = svc_classifier.score(xt1[0], xt1[1])
#  logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=1500)
log_reg.fit(x1[0], x1[1])
LR_score = log_reg.score(xt1[0], xt1[1])
print("SVM score for bicubic  : "+str(svm_score))
print(f"Logistic Regression score for bicubic : "+str(LR_score), '\n')
################################################################################

#  SVM classifier for imagewoof and bilinear
svc_classifier = SVC(C=2, kernel='rbf', gamma='scale')
svc_classifier.fit(x2[0], x2[1])
svm_score = svc_classifier.score(xt2[0], xt2[1])
#  logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=1500)
log_reg.fit(x2[0], x2[1])
LR_score = log_reg.score(xt2[0], xt2[1])
print("SVM score for bilinear  : "+str(svm_score))
print(f"Logistic Regression score for bilinear : "+str(LR_score), '\n')
##################################################################################

#  SVM classifier for imagewoof and pixel_shuffle
svc_classifier = SVC(C=2, kernel='rbf', gamma='scale')
svc_classifier.fit(x3[0], x3[1])
svm_score = svc_classifier.score(xt3[0], xt3[1])
#  logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=1500)
log_reg.fit(x3[0], x3[1])
LR_score = log_reg.score(xt3[0], xt3[1])
print("SVM score for pixel  : "+str(svm_score))
print(f"Logistic Regression score for pixel : "+str(LR_score), '\n')
#################################################################################

#  SVM classifier for combined data, C=10.
X = np.vstack((x1[0][0:599, :], x1[0][800:999, :], x2[0][800:999, :], x3[0][800:999, :]))
Y = (np.hstack((x1[1][0:599], x1[1][800:999], x2[1][800:999], x3[1][800:999]))).ravel()
Xt = np.vstack((xt1[0][0:199, :], xt1[0][200:399, :], xt2[0][0:199, :], xt3[0][0:199, :]))
Yt = (np.hstack((xt1[1][0:199], xt1[1][200:399], xt2[1][0:199], xt3[1][0:199]))).ravel()

svc_classifier = SVC(C=10, kernel='rbf', gamma='scale')
svc_classifier.fit(X, Y)
svm_score = svc_classifier.score(Xt, Yt)
#  logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=1500, C=2)
log_reg.fit(X, Y)
LR_score = log_reg.score(Xt, Yt)
print("SVM score for combined data  : "+str(svm_score))
print(f"Logistic Regression score for combined data : "+str(LR_score), '\n')

# plotting mean plot of extracted features of image
plt.plot(np.mean(x1[0][0:799, :], axis=0), color='blue', label='Real')
plt.plot(np.mean(x1[0][800:1599, :], axis=0), color='red', label='Bicubic up sampling',)
plt.plot(np.mean(x2[0][800:1599, :], axis=0), color='black', label='Bilinear up sampling',)
plt.plot(np.mean(x3[0][800:1599, :], axis=0), color='brown', label='pixel shuffle up sampling')
plt.xlabel("Spatial Frequency", fontsize=10)
plt.ylabel("Power Spectrum", fontsize=10)
plt.legend()
plt.show()

'''
# plotting label zero for real image and 1 for fake image
plt.plot(x3[1])
plt.title('Training examples label, zero for real image, 1 for fake image')
plt.xlabel('Number of training samples')
plt.ylabel('label value')
plt.figure()
plt.plot(xt3[1])
plt.title('Test examples label, zero for real image, 1 for fake image')
plt.xlabel('Number of test samples')
plt.ylabel('label value')
plt.show()
'''
