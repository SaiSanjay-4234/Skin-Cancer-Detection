import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import pandas as pd
import seaborn as sns
import cv2
import scipy.misc
import scipy.ndimage
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report
import warnings
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", UserWarning)
#from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array


#data = r'D:\MANIKANDAN\MARAN-ELYSIAN\2021\PRODUCT\PROJ10\CMP\Source code\dataset'
benign = glob.glob(r'dataset\benign\*.jpg')
malignant = glob.glob(r'dataset\malignant\*.jpg')
print('Number of images with benign : {}'.format(len(benign)))
print('Number of images with malignant : {}'.format(len(malignant)))

from tkinter import filedialog
from tkinter import *
global fileNo
import os
#Image Selection
#Image Selection
root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'RGB/'

options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("JPEG files","*.jpg"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')
Image = cv2.imread(head_tail[0]+'/'+fileNo[0]+'.jpg')
Image1=Image[:,:,0]
img=cv2.resize(Image,(512,512))
plt.title('Original Image')
plt.imshow(img)


#Threshold segmentation
print('\n')
print('Threshold Segmentation')
image=img
imga = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
ret, TB = cv2.threshold(imga, 120, 255, cv2.THRESH_BINARY)
ret, TBI = cv2.threshold(imga, 120, 255, cv2.THRESH_BINARY_INV)
ret, TT = cv2.threshold(imga, 120, 255, cv2.THRESH_TRUNC)
ret, T_T = cv2.threshold(imga, 120, 255, cv2.THRESH_TOZERO)
ret, TTI = cv2.threshold(imga, 120, 255, cv2.THRESH_TOZERO_INV)

 

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, TB, TBI, TT, T_T, TTI]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


benign_ = img
benign_ = 255 - benign_
median_filtered = scipy.ndimage.median_filter(benign_, size=3)


fig, ax = plt.subplots(1, 3, figsize=(10, 8));
plt.suptitle('SAMPLE PROCESSED IMAGE', x=0.5, y=0.8)
plt.tight_layout()

ax[0].set_title('ORG.', fontsize=12)
ax[1].set_title('BENIGN', fontsize=12)
ax[2].set_title('MEADIAN_FILTER', fontsize=12)

ax[0].imshow(255-benign_, cmap='gray');
ax[1].imshow(benign_, cmap='gray');
ax[2].imshow(median_filtered, cmap='gray');
plt.show()


min_YCrCb = np.array([0,153,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

# Get pointer to video frames from primary device
image = img
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
plt.title('CANCER')
plt.imshow(skinYCrCb)
plt.show()
# Reading same image in another
# variable and converting to gray scale.
img1 = img
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
# Find contours
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True) 
# Select long perimeters only
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
numcards=len(listindex)
# Show image
imgcont = img.copy()
[cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 5) for i in listindex]

plt.title('SKIN CANCER')
plt.imshow(imgcont)
plt.show()

from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
#img = io.imread('image_sample.jpg')

gray = color.rgb2gray(image)
image = img_as_ubyte(gray)

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
inds = np.digitize(image, bins)

max_value = inds.max()+1
matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

# GLCM properties
def contrast_feature(matrix_coocurrence):
	contrast = greycoprops(matrix_coocurrence, 'contrast')
	return "Contrast = ", contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
	return "Dissimilarity = ", dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return "ASM = ", asm

print(contrast_feature(matrix_coocurrence))
print(dissimilarity_feature(matrix_coocurrence))
print(homogeneity_feature(matrix_coocurrence))
print(energy_feature(matrix_coocurrence))
print(correlation_feature(matrix_coocurrence))
print(asm_feature(matrix_coocurrence))

lst_benign = []
for x in benign:
  lst_benign.append([x,1])
lst_malignant = []
for x in malignant:
  lst_malignant.append([x,0])
lst_complete = lst_benign + lst_malignant
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])
df.head(10)
filepath_img ="dataset/malignant/*.png"
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
df.shape

plt.figure(figsize = (10,10))
sns.countplot(x = "target",data = df)
plt.title("BENING and MALIGNANT") 
plt.show()

def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(60,60))  # resize
  img = img / 255 #scale
  return img

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)
X, y = create_format_dataset(df)
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,stratify = y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


'''CNN'''
CNN = Sequential()

CNN.add(Conv2D(32,(2,2),input_shape = (60,60,3),activation='relu'))
CNN.add(Conv2D(64,(2,2),activation='relu'))
CNN.add(MaxPooling2D())
CNN.add(Conv2D(32,(2,2),activation='relu'))
CNN.add(MaxPooling2D())

CNN.add(Flatten())
CNN.add(Dense(32))
CNN.add(Dense(1,activation= "sigmoid"))
CNN.summary()
CNN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
CNN.fit(X_train,y_train,epochs = 10,batch_size = 5)
print("Accuracy of the CNN is:",CNN.evaluate(X_train,y_train)[1]*100, "%")
history = CNN.history.history

#Plotting the accuracy
train_loss = history['loss']

train_acc = history['acc']

    
# performance
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(train_acc, label='Training Accuracy')
plt.title('Performance Plot')
plt.legend()
plt.show()
    

y_pred = CNN.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')
y_pred
print('\n')
classification=classification_report(y_test,y_pred)
print(classification)
print('\n')
plt.figure(figsize = (5,4.5))
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()