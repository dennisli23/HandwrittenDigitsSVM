
# coding: utf-8

# In[568]:


from scipy.misc import imread # using scipy's imread
import cv2
import numpy as np

def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)

def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

# Example usage
big_img = imread("a.png", flatten = True)# flatten = True converts to grayscale
cv2.imshow("a",big_img/255)
cv2.waitKey(1000)
cv2.destroyAllWindows()

imgs = separate(big_img) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)
for img in imgs:
    cv2.imshow("a",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[569]:


columnA = imread("a.png", flatten = True)
imgsA = separate(columnA)
for img in imgsA:
    cv2.imshow("a",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[570]:


columnB = imread("b.png", flatten = True)
imgsB = separate(columnB)
for img in imgsB:
    cv2.imshow("a",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[571]:


columnC = imread("c.png", flatten = True)
imgsC = separate(columnC)
for img in imgsC:
    cv2.imshow("a",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[572]:


def fun(imgs, data):
    for img in imgs:
        scaled = cv2.resize(img, (5,5), interpolation = cv2.INTER_CUBIC)
        img = np.array(scaled)
        data.append(img.flatten())
dataA = []
dataB = []
dataC = []
fun(imgsA, dataA)
fun(imgsB, dataB)
fun(imgsC, dataC)


# In[573]:


dataA = np.array(dataA)
dataB = np.array(dataB)
dataC = np.array(dataC)


# In[574]:


data = np.concatenate((dataA, dataB, dataC))
data.shape


# In[575]:


targetA = [0]*23
targetB = [1]*23
targetC = [2]*23
target = np.concatenate((targetA, targetB, targetC))
target.shape


# In[576]:


#my own handwriting
columnE = imread("e.JPG", flatten = True)
imgsE = separate(columnE)
for img in imgsE:
    cv2.imshow("e",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[577]:


columnD = imread("d.JPG", flatten = True)
imgsD = separate(columnD)
for img in imgsD:
    cv2.imshow("d",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[578]:


columnH = imread("h.JPG", flatten = True)
imgsH = separate(columnH)
for img in imgsH:
    cv2.imshow("h",img) 
    cv2.waitKey(250)
cv2.destroyAllWindows()


# In[579]:


dataD = []
dataE = []
dataH = []
fun(imgsD, dataD)
fun(imgsE, dataE)
fun(imgsH, dataH)


# In[580]:


dataD = np.array(dataD)
dataE = np.array(dataE)
dataH = np.array(dataH)


# In[581]:


data = np.concatenate((dataD, dataE, dataH))
data.shape


# In[582]:


targetD = [0]*6
targetE = [1]*6
targetH = [2]*6
target = np.concatenate((targetD, targetE, targetH))
target.shape


# In[583]:


def partition(data, target, p):
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    train_size = int(len(target)*p)
    test_size = len(target)-train_size
    duplicates = []
    
    index = np.random.randint(len(target))
    for i in range(train_size):
        while index in duplicates:
            index = np.random.randint(len(target))
        duplicates.append(index)
        train_data.append(data[index])
        train_target.append(target[index])
    
       
    index = np.random.randint(len(target))
    for i in range(test_size):
        while index in duplicates:
            index = np.random.randint(len(target))
        duplicates.append(index)
        test_data.append(data[index])
        test_target.append(target[index])
    
    return train_data, train_target, test_data, test_target, test_size
    


# In[584]:


train_data, train_target, test_data, test_target, test_size = partition(data, target, 0.5)


# In[585]:


import sklearn.svm as svm
linearModel = svm.LinearSVC()
linearModel.fit(train_data, train_target)
predicted = linearModel.predict(test_data)
matches = 0
for i in range(test_size):
    if predicted[i] == test_target[i]:
        matches = matches + 1
        
accuracy = float(matches)/test_size * 100


# In[586]:


print "Predicted: " + str(predicted) + "\nTruth: " + str(test_target) + "\nAccuracy: " + str(accuracy) + "%"

