#!/usr/bin/env python
# coding: utf-8

# # Dyslexia Detection using Eye-Tracking

# Dyslexia is a neurodevelopmental reading disability that adversely affects the speed and accuracy of word recognition, and as a consequence, impedes reading fluency and text comprehension.
# It affects 5–10% of the population. While there is yet no full understanding of the cause of dyslexia, or agreement on its
# precise definition, it is certain that many individuals suffer persistent problems in learning to
# read for no apparent reason. 
# 
# 
# Eye tracking is a more natural screening procedure than oral or written examinations, as it doesn't require a verbal answer from the individual and provides a means to objectively evaluate the reading process as it occurs in real time. While dyslexia is primarily a language-based learning issue, our findings indicate that eye movements during reading can accurately predict individual reading abilities. Eye tracking can also effectively identify children at risk of long-term reading difficulties.
# 
# 
# Our study is
# based on a sample of 97 high-risk subjects with early identified word decoding difficulties
# and a control group of 88 low-risk subjects.
Dataset:
    
    https://drive.google.com/drive/folders/1-lyxnq6IZHEJSAeOY-cG2nEeZ4jckO5K?usp=drive_link
# Our aim here was to let the system use it's computational power to determine the the 2 types of candidates appart. We intended on using a non-supervised learning method to allow the data to separate itself rather than provide it with final results which it would use to immprove the accuracy. We have mainly focused on K-Means for classification purposes.
# 
# It has data of:
# 
# 98 Dyslexic Candidates
# 88 Control Candidates
# Structure of the data:
# 
# The data describes the exact location of focus for the individual. For each candidate we have the position of the left and right eye in the x-y coordinate plane. Data vectors for each candidate:
# 
# X coordinate of left eye -> LX
# Y coordinate of left eye -> LY
# X coordinate of right eye -> RX
# Y coordinate of right eye -> RY

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Data Gathering & Manipulation

# In[154]:


# Importing libraries for Data Gathering & Manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure
from scipy.spatial.distance import euclidean as eu
import math
from scipy import signal
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing

import glob




# Importing Libraries for Binning
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing
from scipy.fftpack import fft, ifft




# Importing the necessary libraries for building neural network and k-means clustering
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


# In[155]:


def get_data():
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Converts the eye-tracking data of the Dyslexic and Control candidates present in the data folder into lists of datafromes
#Each data frame represents the data of 1 candidate
#The entire data is converted into 2 lists:
# 1. C_data for control candidates 
# 2. D_data for dyslexic candidates
#Structure of the dataframes:
#        LX    LY    RX    RY
#    0   ..    ..    ..    .. 
#    1   ..    ..    ..    .. 
#    2   ..    ..    ..    .. 
#   ..   ..    ..    ..    .. 
#    n   ..    ..    ..    .. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    D_path = glob.glob(r'C:\Users\aswin\Downloads\Recording Data\Dyslexic' + "\*")
    C_path = glob.glob(r'C:\Users\aswin\Downloads\Recording Data\Control' + "\*")

    C_data = []
    for path in C_path:
        temp = pd.read_csv(path)
        temp = temp.drop('Unnamed: 0',axis = 1)
        C_data.append(temp)

    D_data = []
    for path in D_path:
        temp = pd.read_csv(path)
        temp = temp.drop('Unnamed: 0',axis = 1)
        D_data.append(temp)

    return C_data, D_data


# # Print_statements
Here we have defined a function plot_entire_candidate() to plot the different features of candidate's eyes in a 2x2 grid.
We try to plot the Features of both the left and right eye in 1 graph and see some indiscrepencies.

Finally we calculate the distance between the elements in the 1st set and 2nd set. We represent them with shades of grey between white (max distance) and black (no distance)
# In[156]:


# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import euclidean as eu


# In[157]:


def plot_entire_candidate(category, num, data):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To plot the different features of a candidate in a 2x2 grid.
#INPUT: 
#     category: Dyslexic or Control, 0 for Control and 1 for Dyslexic
#     num: serial number of the entry which is to be printed
#     data: entire dataset

#Structure of plot: 
# LX  LY
# RX  RY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=2)
    gs.update(wspace = 0.2, hspace = 0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(data[category][num]['LX'], linewidth = 0.7)
    ax0.set_title('LX')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(data[category][num]['RX'], linewidth = 0.7)
    ax1.set_title('RX')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data[category][num]['LY'], linewidth = 0.7)
    ax2.set_title('LY')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data[category][num]['RY'], linewidth = 0.7)
    ax3.set_title('RY')


# In[158]:


def plot_left_right(category, num, data, axis):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is to plot left and right eye readings of one axis for one candidate. This is to mainly check the overlap betweeen the 2 sides. 
#INPUT: 
#     category: Dyslexic or Control, 0 for Control and 1 for Dyslexic
#     num: serial number of the entry which is to be printed
#     data: entire dataset
#     axis: Which axis to be printed - x or y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.plot(data[category][num]['R'+axis], 'r', label = 'Right', alpha=1)
    plt.plot(data[category][num]['L'+axis], 'y', label = 'Left', alpha=0.7)
    plt.legend()


# In[159]:


def return_sq_im(set1, set2):    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculated the distances of element of the first set to each in the second set and represented them with shades of grey between white(max distance) and black(no distance).
#Each intersection/pixel represents the distance between the corresponding vectors.
#Eg. The grade of grey of the pixel at (4,1) will represent the distance between the 4th element of the first set to the 1st element of the second set
#INPUT:
#     set1: First set of vectors
#     set2: Second set of vectors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    matrix = np.empty([len(set1),len(set2)])    
    for a in range(len(set1)):
        for b in range(len(set2)):
            m = set1[a]
            n = set2[b]

            dis = eu([m], [n])
            matrix[a][b] = dis 

    mx = np.amax(matrix)
    img = matrix/mx  
    img = img*256
    
    plt.imshow(img, cmap='gray', vmin=0, vmax=256)
    plt.show()


# In[ ]:





# # Exploring Dataset
The dataset we have collected is Screening for Dyslexia using Eye Tracking during reading. 

Our aim here is to let the system use it's computational power to determine the the 2 types of candidates appart.

It has data of:

98 Dyslexic Candidates
88 Control Candidates

Structure of the data:

The data describes the exact location of focus for the individual. For each candidate we have the position of the left and right eye in the x-y coordinate plane. Data vectors for each candidate:

X coordinate of left eye -> LX
Y coordinate of left eye -> LY
X coordinate of right eye -> RX
Y coordinate of right eye -> RY
# In[160]:


# Importing Data
feature_list=['LX', 'LY', 'RX', 'RY']
C_data, D_data = get_data()
Full_data = [C_data, D_data]


# In[161]:


#print entire data of one candidate
#you can plot of any candidate using the following function:
plot_entire_candidate(1, 45, Full_data) #(0-Control 1-Dyslexic, number of the candidate)


# In[162]:


# looking at Data

# (0-Control 1-Dyslexic, number of the candidate)

#printing x-axis data for a few random candidates:
plt.plot(C_data[4]['RX'])


# In[163]:


# There are many dips that are observed here - They indicate the ends of lines 


# In[164]:


# Plotting a few more points
plt.plot(C_data[60]['RX'])


# In[165]:


plt.plot(D_data[49]['RX'])


# In[166]:


# Although Dips are apparent, it was difficult to seperate the lines, especially the 4th and 5th lines. All the candidates were asked to test the same number of lines


# However there were some inconsistencies between the left and the right eye

# In[167]:


#ploting few left and right eye readings:
plot_left_right(1, 89, Full_data, 'X')

In this case, the Right Eye has been marked with Red lines while the Left Eye is marked with Yellow Lines. Orange part is the part where both the left and right eye intersect.
However, there are several instances of red or yellow lines i.e, no intersection b/w Left and Right

Solution:  We try to average the values of Left and Right so that system is not biased

# In[ ]:





# In[ ]:





# # Data_Load
Converts the eye-tracking data of the Dyslexic and Control candidates present in the data folder into lists of datafromes
Each data frame represents the data of 1 candidate

The entire data is converted into 2 lists:
 1. C_data for control candidates 
 2. D_data for dyslexic candidates 
# In[168]:


#Get Control and Dyslexic data as required for the STFT operations 
def get_stft_data(C_data, D_data): 
    C_new = []
    for data in C_data:
        X =data[['LX','RX']]
        Y =data[['LY','RY']]
        Xm = X.mean(axis=1)
        Ym = Y.mean(axis=1)
        f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
        f = f.transpose()
        f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
        C_new.append(f)

    D_new = []
    for data in D_data:
        X =data[['LX','RX']]
        Y =data[['LY','RY']]
        Xm = X.mean(axis=1)
        Ym = Y.mean(axis=1)
        f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
        f = f.transpose()
        f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
        D_new.append(f)
    
    C_new,D_new = normalise_data(C_new,D_new)
    
    C_cmx = []
    C_real= []
    C_img=[]
    for j in range(len(C_new)):
        dat = C_new[j]
        x = dat['X']
        y = dat['Y']
        t = dat['T']

        z=[]
        x_in=[]
        y_in=[]
        for i in range(0,x.size):
            z.append(complex(x[i],y[i]))
            x_in.append(x[i])
            y_in.append(y[i])


        C_cmx.append(z)
        C_real.append(x_in)
        C_img.append(y_in)

    D_cmx = []
    D_real= []
    D_img=[]
    for j in range(len(D_new)):
        dat = D_new[j]
        x = dat['X']
        y = dat['Y']
        t = dat['T']

        z=[]
        x_in=[]
        y_in=[]
        for i in range(0,x.size):
            z.append(complex(x[i],y[i]))
            x_in.append(x[i])
            y_in.append(y[i])
        D_cmx.append(z)
        D_real.append(x_in)
        D_img.append(y_in)
    
    return C_cmx, C_real, C_img, D_cmx, D_real, D_img, C_new, D_new


# In[169]:


# Function to normalise data

def normalise_data(C_new,D_new): 
    for i in range(len(C_new)):
        C_tempx = np.abs(C_new[i]['X'])
        mx = max(C_tempx)
        C_tempy = np.abs(C_new[i]['Y'])
        my= max(C_tempy)
        C_new[i]['X'] = C_new[i]['X']/np.abs(mx)
        C_new[i]['Y'] = C_new[i]['Y']/np.abs(my)
    for i in range(len(D_new)):
        D_tempx = np.abs(D_new[i]['X'])
        mx = max(D_tempx)
        D_tempy = np.abs(D_new[i]['Y'])
        my= max(D_tempy)
        D_new[i]['X'] = D_new[i]['X']/np.abs(mx)
        D_new[i]['Y'] = D_new[i]['Y']/np.abs(my)  
    return C_new,D_new


# In[170]:


# Function to find the average of the Left & Right Eyes. This is used in the cases of intersection b/w the data points for the Left and Right Eye.

def average_l_r(data):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the average values of the left and righty eye readings.
# INPUT:
#     data   : dataframe of a single candidate 

#x   : the average of the x coordinates of the left and right eye readings
#y   : the average of the y coordinates of the left and right eye readings

#OUTPUT:
#     x_y_data: combines the average of the x and y coordinates into a dictionary of form: 
#     X: x,
#     Y: y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    x = [sum(x)/2 for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x)/2 for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    x_y_data = {'X':x, 'Y':y}
    
    return x_y_data 


# In[171]:


def data_lens():
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Returns a representation of lengths of the entries in C_data and D_data combined. 
#OUTPUT: 
#     lens: contains the representation of lengths of each entry in C_data and D_data in this order.
#           value : length represented
#             0   :        999
#             1   :        1249
#             2   :        1499
#             3   :        1749
#             5   :        1999
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    C_data, D_data = get_data()
    lens = []
    for dSet in [C_data, D_data]:
        for data in dSet:
            lens.append(int(((len(data['LX']) + 1)/250) - 4))
    return lens


# In[ ]:





# In[ ]:





# In[172]:


# Now to fix the issue of averaging,


# In[173]:


x_y_data = average_l_r(D_data[80])


# In[174]:


plt.plot(x_y_data['X'])


# In[ ]:





# In[ ]:





# # Data Manipulation:
Now the data that we have currently needs to be run through some filters to get our desired result.
For this purpose, we use inter-padding, padding at the end to add zeroes and other values to the initial dataset. This helps in preserving the spatial size of the data as it passes through filters.

We also perform Interpolation - Adds datapoints at regular intervals by taking average of the adjacent values to match the longest length of feature vector

Exterpolation - Removes data points at regular intervals to match the shortest length of a single fearture vector.
# In[175]:


# To define the original dataset

def original_vector(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#appends the 'X' and 'Y' values of the dataset one after the other 

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values
#  data = xxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyyyyy
#        |________________________||________________________|       
#                    x                        y

#OUTPUT:
#     data : Appended avg of x and y values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += data_conv[f]

    return data


# In[176]:


def get_padded(a,ml):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Adds 0 padding at the end to make vector length equal to 'ml'
#INPUT:
#     a = vector to be padded
#     ml = target length after padding

#   a     =   ////////////////////////

#padded a =   ////////////////////////0000000
#             |<-------------ml------------->|
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    a = np.pad(a, (0,abs(ml - len(a))), 'mean')
    
    return a 


# In[177]:


def padding_at_last(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combines the data vectors (X, Y) one after the other into a single vector. Added 0s towards the end to match the length of the longest vector if just x and y values are combined without any padding. 
#In our dataset the longest vector is of length 3998.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

##  data = xxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyyyyy0000000000000000
#          |_______________________||_______________________||______________|       
#                     x                        y                    0s
#          |<-------------------length of longest vector------------------->|

#OUTPUT:
#     data: x, y combined and padded with 0s
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += data_conv[f]
    
    return get_padded(data,3998)


# In[178]:


def padding_in_bw(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Pads each feature (X and Y) separately to match the longest length of a single fearture vector. In our case it's 1999. Then appends them together.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

# padded x =  xxxxxxxxxxxxxxxxxxxxxxxxx00000000
#             |_______________________||______|
#                         x               0s          

# padded y =  yyyyyyyyyyyyyyyyyyyyyyyyy00000000
#             |_______________________||______|
#                         y               0s    

#             |<----------------------------->|
#          length of longest single fearture vector

# data =  xxxxxxxxxxxxxxxxxxxxxxxxx00000000yyyyyyyyyyyyyyyyyyyyyyyyy00000000
#         |_______________________||______||_______________________||______|
#                     x               0s               y               0s     

#OUTPUT:
#     data: combined padded x and padded y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += get_padded(data_conv[f],1999).tolist()
        
    return data


# In[179]:


def positions(secf, dif, fact):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Gives indexes of equally spaced positions in a vector to add or remove values to match a certain length. In our case it's 1999.
#INPUT: 
#     secf:  Length of each equal space
#     dif:  Number of positions to be added/removed
#     fact:  

#OUTPUT:
#     arr: vector containing 'dif' number of equally spaced positions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    arr = []
    cn = 0
    for a in range(dif-1):
        cn = cn+secf + 1 if (a+1)%fact == 0 else cn+secf 
        arr.append(cn)
    
    return arr


# In[180]:


def interpolating(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Adds data points at regular intervals by taking the average of the adjacent values to match the longest length of a single fearture vector. In our case it's 1999.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

# interpolated x =  xxxxxXxxxxxXxxxxxXxxxxxXxxxxx
#                  |_____^_____^_____^_____^___|
#           x with values inserted at regualar intervals                 

# interpolated y =  yyyyyYyyyyyYyyyyyYyyyyyYyyyyy
#                  |_____^_____^_____^_____^___|
#           y with values inserted at regualar intervals    

#             |<----------------------------->|
#          length of longest single fearture vector

# combined_data =   xxxxxXxxxxxXxxxxxXxxxxxXxxxxxyyyyyYyyyyyYyyyyyYyyyyyYyyyyy
#                  |____________________________||___________________________|
#                          interpolated x               interpolated y     

#OUTPUT:
#     combined_data: Appended interpolated x and inpterpolated y together
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    
    combined_data = []
    ln = len(data_conv['X'])
    dif = 2000-ln
    sec = math.floor(ln/dif)

    pos_arr = positions(sec, dif, 1 if ln == 1499 else 2)

    for f in feature_list:
        data = []         
        curr = 0 
        for pos in range(1999):
            if curr < ln:
                if pos in pos_arr:
                    data.append((data_conv[f][curr]+data_conv[f][curr+1])/2)                        
                    data.append(data_conv[f][curr])
                    curr += 1
                else:
                    data.append(data_conv[f][curr])
                    curr += 1
        while len(data)<1999:
            last_val = data[-1]
            data.append(last_val)
            
        combined_data += data
    return combined_data


# In[181]:


def exterpolation(data_set):  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Removes data points at regular intervals to match the shortest length of a single fearture vector. In our case it's 999.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

#                       x   x   x   x
# exterpolated x =  xxxxxxxxxxxxxxxxxxxx
#                  |___^___^___^___^___|
#           x with values removed at regualar intervals                 

#                       y   y   y   y
# exterpolated y =  yyyyyyyyyyyyyyyyyyyy
#                  |___^___^___^___^___|
#           y with values removed at regualar intervals    

#                  |<----------------->|
#          length of shortest single fearture vector

# combined_data =   xxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyy
#                  |___________________||__________________|
#                      exterpolated x      exterpolated y     

#OUTPUT:
#     combined_data: Appended exterpolated x and exterpolated y together
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    
    combined_data = []
    ln = len(data_conv['X'])
    dif = ln-998
    sec = math.floor(ln/dif)

    pos_arr = positions(sec, dif, 4 if ln == 1749 else 2)


    for f in feature_list:
        data = []  
        for pos in range(ln):
            y = 0
            if pos not in pos_arr:
                data.append(data_conv[f][pos])

        combined_data += data
        
    return combined_data


# In[ ]:





# In[182]:


full_data = original_vector(C_data[2])


# In[183]:


len(C_data[2]['LX'])


# In[184]:


plt.plot(full_data)
plt.axvline(1249, linewidth = 0.7)

Unfortunately the lengths of the lines are not equal. To resolve this, we try to equalize the lengths of the vectors and then apply K-Means Classification.
We shall perform padding which is the addition of layers of zeroes or other values as an additional layer over the input layer.
This leads to preserving the spatial size of the matrix even after passing through a filter.
# In[ ]:




length_confusion_matrixThis is used to divide the data into groups of different lengthed vectors

INPUT:
     act_lab : set of actual labels obtained from the dataset
     pred_lab : set of labels predicted by the classifier

 conf_lens: set of confusion matrix for different data lengths
 
 Structure:
        [[[/,/],       -> for 999 length vectors
          [/,/]],      

         [[/,/],       -> for 999 length vectors
          [/,/]],

         [[/,/],       -> for 999 length vectors
          [/,/]],

         [[/,/],       -> for 999 length vectors
          [/,/]],

         [[/,/],       -> for 999 length vectors
          [/,/]]]

OUTPUT:  
     conf_len: Confustion matrix of each section of the dataset, segregated based on the number of readings. This helps to access the performance of the classifier for each such section.
# In[185]:


# Now we shall split the data into different length vectors. We will use k-means classification on the new data and compare the confusion matrix of different groups
# We compare the performance on different sub-groups


def conf_mat(act_lab, pred_lab):
    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    data_lens1 = data_lens()
    for a in range(len(act_lab)):
        conf_len[data_lens1[a]][int(act_lab[a])][int(pred_lab[a])] += 1
    
    return np.array(conf_len)
    


# In[ ]:





# # We shall now try different methods to equalize length of the vectors

# # 1. Padding at the end: 
Combined all the data vectors (X, Y) one after the other into a single vector. Added 0s or avg of all values towards the end to match the length of the longest vector.
# In[186]:


padding = padding_at_last(C_data[2])  # From Data Manipulations


# In[187]:


plt.plot(padding)
plt.axvline(1249, linewidth = 0.7, c = 'k')
plt.axvline(1249*2, linewidth = 0.7, c = 'k')
plt.axvline(0, linewidth = 0.7, c = 'k')
plt.axvline(len(padding), linewidth = 0.7, c = 'k')


# In[188]:


# Applying K-means after performing padding to see performance


# In[189]:


all_candidates = []
for categ in [D_data, C_data]:
    for data_set in categ:
        all_candidates.append(padding_at_last(data_set))


# In[190]:


kmeans = KMeans(n_clusters = 2, random_state=0).fit(all_candidates)

predicted_labels = kmeans.labels_
actual_labels = np.concatenate((np.ones(97), np.zeros(88)))


# In[191]:


acc = accuracy_score(actual_labels,predicted_labels)*100
acc


# In[192]:


#confusion matrix of sets of different lengths of candidates
conf_mat(actual_labels, predicted_labels)

Problem that happens here is, since the vectors are of different lengths, the reading at a particular point in time may not be the same across all the columns.

For instance, take 2 vectors for LX & RX with length of vectors 2149, 2500. The LX of the first will end at 2149th position in the final vector and the 2150th position will be filled by the 1st value of the LY vector. But for the second one, the 2150th position will still be filled by a reading from the LX vector. When applying k-means the algorithm will compare the 2150th vector of each candidate irrespectively. This will be a problem as different features will get compared with each other for different candidates.


In the confusion matrix we can see that most of the shorter lenghted vectors are getting categorized as control and longer ones as dyslexic. Here the algorithm is relying on the length of the vector rather than the vector itself and is not the best algorithm.
# In[ ]:





# # 2. Padding in between
Padding in between is a method in which each of the vectors is padded with 0's or additional layers individually and then combined
# In[193]:


padding = padding_in_bw(C_data[2])


# In[194]:


plt1 = plt.figure(figsize=(12,6))
plt.plot(padding)
plt.axvline(1249, linewidth = 0.7, c = 'k')
plt.axvline(len(padding)/2, linewidth = 0.7, c = 'k')
plt.axvline(1249 + len(padding)/2, linewidth = 0.7, c = 'k')
plt.axvline(len(padding), linewidth = 0.7, c = 'k')
plt.axvline(0, linewidth = 0.7, c = 'k')


# In[195]:


all_candidates = []
for categ in [D_data, C_data]:
    for data_set in categ:
        all_candidates.append(padding_in_bw(data_set))


# In[196]:


kmeans = KMeans(n_clusters = 2, random_state=0).fit(all_candidates)

predicted_labels = kmeans.labels_
actual_labels = np.concatenate((np.ones(97), np.zeros(88)))


# In[197]:


acc = accuracy_score(actual_labels,predicted_labels)*100
acc


# In[198]:


conf_mat(actual_labels, predicted_labels)

Problem:
    Here the candidates get segregated almost randomly

# In[ ]:





# # 3. Interpolation
In Interpolation, we add data points at regular intervals by taking the averages of the adjacent values. Suppose we want to
make 1000 length vector into a 2000 length vector, for this we add data points after each data point after averaging the adjacent values.
# In[199]:


padding = interpolating(C_data[2])


# In[200]:


plt1 = plt.figure(figsize=(12,6))
plt.plot(padding)
plt.axvline(1999, linewidth = 0.7, c = 'k')
plt.axvline(1999*2, linewidth = 0.7, c = 'k')
plt.axvline(0, linewidth = 0.7, c = 'k')


# In[201]:


all_candidates = []
for categ in [D_data, C_data]:
    for data_set in categ:
        all_candidates.append(interpolating(data_set))


# In[202]:


kmeans = KMeans(n_clusters = 2, random_state=0).fit(all_candidates)

predicted_labels = kmeans.labels_
actual_labels = np.concatenate((np.ones(97), np.zeros(88)))


# In[203]:


acc = accuracy_score(actual_labels,predicted_labels)*100
acc


# In[204]:


conf_mat(actual_labels, predicted_labels)

Problem:
    However here we have added a lot of data and in an attempt to balance all the vectors, the candidates who read quickly got 
    distorted.
# In[ ]:





# # 4.Exterpolation
Exterpolation is done to reduce the length of all vectors to the length of the shortest vector. For instance if we have a 
2000 length vector and need to convert it to 1000 length, we remove every alternate data point.
# In[205]:


padding = exterpolation(C_data[2])


# In[206]:


plt1 = plt.figure(figsize=(12,6))
plt.plot(padding)
plt.axvline(2000, linewidth = 0.7, c = 'k')
plt.axvline(1000, linewidth = 0.7, c = 'k')
plt.axvline(0, linewidth = 0.7, c = 'k')


# In[207]:


all_candidates = []
for categ in [D_data, C_data]:
    for data_set in categ:
        all_candidates.append(exterpolation(data_set))


# In[208]:


kmeans = KMeans(n_clusters = 2, random_state=0).fit(all_candidates)

predicted_labels = kmeans.labels_
actual_labels = np.concatenate((np.ones(97), np.zeros(88)))


# In[209]:


acc = accuracy_score(actual_labels,predicted_labels)*100
100  - acc


# In[210]:


conf_mat(actual_labels, predicted_labels)

Problem:
    Here we have removed a lot of information and this might be vital information. Also the reality of candidates who read faster also gets distorted.
# In[ ]:





# # 5. Black & White Distance Visualizations
In this we will be using a matrix of different shades of grey to visualize the distance between the data points.
2 candidates are selected and the distance between each point in candidate 1 and each point in candidate 2 is calculated. This distance is represented in the form of a matrix with White (max distance) & Black (no distance)
# In[211]:


#selecting candidates:
set1 = [D_data[6], D_data[3], D_data[23], D_data[43], D_data[64]]
set2 = [C_data[63], C_data[23], C_data[43], C_data[3], C_data[14]]

#using padding at the end to equalize lengths (any method can be used):
new_set1 = np.array([])
new_set2 = np.array([])

for i in set1:
    new_set1 = np.append(new_set1,padding_in_bw(i))
    
for i in set2:
    new_set2 = np.append(new_set2,padding_in_bw(i))   


# In[212]:


new_set1


# In[60]:


return_sq_im(new_set1, new_set2)


# In[ ]:





# In[ ]:





# In[ ]:





# # Binning
Maps a vector any length to a vector of a fixed length(bins) as needed. 
The input elements are:
    - Bins: Number of Elements in the target vector
    - FFT: Set of all Fast Fourier Transforfm
    - overlap_per: Percentage of overlap between successive entries
        
Each entry of the resulting vector is a sum of fixed number of elements of the input vector. Few of these elements are considered common for successive entries into the resulting vector. This is the overlapping factor.
So the fixed number of elements considered for the each entry = (lenght of input vector/lenght of output vector) + overlap

Output:
    - binned: list of all ffts after binning
Problem: 
The problem that we have faced so far is due to the varying lengths of the vectors in dataset. Now one of the approaches to solve this is to extract n-features from the dataset with each dimension representing a feature. This however is not possible in case we do not have very good idea about the dataset. 
If we consider raw signal for processing then we have to resort to padding, interpolation, exterpolation to equalize the vector length. This can lead to changes in the temporal and spectral nature of the data.

For this we convert the time series data to frequency domian so that we can add/delete data points in frequency domain without affecting the temporal nature of the data

# # Time Domain --->  Frequency Domain
A frequency transform of the time series data is performed, which in most cases preserves the important information in the data set. However, the signal lengths differ in the spectral domain. We can now adjust the signal's frequency without changing its temporal features. To avoid adding points to the signal, try using a sliding window to establish frequency groupings or bins. If we create the same number of bins across the signal while preserving a fraction of frequencies common between consecutive bins, we can capture defining properties of the temporal domain without having to decide whether to include or exclude certain features.Fourier transform is used to convert a signal in the Time domain to the Frequency Domain. Here we are using a variant of Fourier Transform called as Fast Fourier Transform (FFT). For example, Shazam and other Music identification services use Fourier Transform to identify songs. Similarly JPEG Compression uses a form of Fourier Transform to remove the high frequency components in an image.
# In[ ]:





# In[213]:


def binning(bins, fft, overlap_per):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# maps a vector any lenght to a vector of a fixed length(bins) as needed. 

#INPUT: 
#     bins: Number of elements in the target vector
#     fft: Set of all FFTs
#     overlap_per: Percentage of overlap between successive entries

#Each entry of the resulting vector is a sum of fixed number of elements of the input vector. Few of these elements are considered common for successive entries into the resulting vector. This is the overlapping factor.

# So the fixed number of elements considered for the each entry = (lenght of input vector/lenght of output vector) + overlap

#OUTPUT: 
#     binned: list of all ffts after binning
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    div_size = len(fft)/bins
    bin_size = div_size*(1+(overlap_per/100))
    half_bin = bin_size/2
    
    binned = []
    
    current_step = bin_size
    for a in range(bins):
        
        pos = np.ceil(half_bin + a*(div_size))
        start = 0 if a == 0 else int(np.ceil(pos - half_bin))
        end = -1 if a == (bins-1) else int(np.ceil(pos + half_bin))
        #print([start, end])
        
        binned = np.append(binned, sum(np.abs(fft[start : end]))) 
        
    return binned


# In[214]:


def kmeans_binning(D_data,C_data, bins, overlap_per):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# does the binning process and then applies kmeans to the binnned vectors. 
#INPUT: 
#     D_data: The data of all Dyslexic candidates
#     C_data: The data of all Control candidates
#     fft: Set of all FFTs
#     overlap_per: Percentage of overlap between successive entries

# OUTPUT:
#     1. conf_len: divides the entire data into groups of different lenghted vectors. Gives the confusion matrix based on the predictions for each separate group
#     2. conf_m: gives confusion matrix based on prediction for the entire dataset 
#     3. acc: gives accuracy of predictions
#     4. buckets: gives the binned data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    data_sets = [D_data,C_data]
    all_buckets = []
    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    for dataset in data_sets:
        for no in range(1,len(dataset)):
            d = average_l_r(dataset[no])
            # Convert the dictionary dataset into a 2D array
            X = np.array([[d['X'], d['Y']]], dtype=object)
            Y = np.array([])
            Y = np.append(Y,X)
            Y_fft = fft(Y)   

            binned = binning(bins, Y_fft, overlap_per)
            all_buckets.append(binned)
    buckets = np.asarray(all_buckets)
            
    kmeans = KMeans(n_clusters = 2, random_state=0).fit(buckets)

    predicted_labels = kmeans.labels_
    actual_labels = np.concatenate((np.ones(95), np.zeros(88)))
    
    #for a in range(len(buckets)):
     #   conf_len[data_lens[a]][int(actual_labels[a])][int(predicted_labels[a])] += 1
    
    conf_m = confusion_matrix(actual_labels,predicted_labels)[:2]
    acc = accuracy_score(actual_labels,predicted_labels)*100
    
    return conf_m, acc, buckets


# In[ ]:





# In[215]:


# Importing Data

feature_list=['LX', 'LY', 'RX', 'RY']
C_data, D_data = get_data()


# In[ ]:




Getting the Binning Results:
# In[216]:


conf, a, binned = kmeans_binning(D_data,C_data, 1000, 10)


# In[217]:


# Confusion Matrix
conf


# In[218]:


# Accuracy
a


# In[219]:


# Binned Data
binned


# In[ ]:





# In[ ]:





# In[ ]:





# # Perceptron Model Building
Finally we move onto the actual model building for our project. We will be creating a Neural Network with the use of Perceptrons to do cluster analysis on our dataset. The previous approaches we have looked at  dealt with the problem using Supervised Learning algorithms like SVM, RandomForest etc.
Through our approach, we try to create an Unsupervised Learning Model. This is done so that the model does not depend on the nature of the data.
 A Perceptron is an algorithm for supervised learning of binary classifiers. This algorithm enables neurons to learn and processes elements in the training set one at a time.
 
The basic components of a perceptron are:
1. Input Layer: The input layer consists of one or more input neurons, which receive input signals from the external world or from other layers of the neural network.
2. Weights: Each input neuron is associated with a weight, which represents the strength of the connection between the input neuron and the output neuron.
3. Bias: A bias term is added to the input layer to provide the perceptron with additional flexibility in modeling complex patterns in the input data.
4. Activation Function: The activation function determines the output of the perceptron based on the weighted sum of the inputs and the bias term. Common activation functions used in perceptrons include the step function, sigmoid function, and ReLU function.
5. Output: The output of the perceptron is a single binary value, either 0 or 1, which indicates the class or category to which the input data belongs.
6. Training Algorithm: The perceptron is typically trained using a supervised learning algorithm such as the perceptron learning algorithm or backpropagation. During training, the weights and biases of the perceptron are adjusted to minimize the error between the predicted output and the true output.
# ![general-diagram-of-perceptron-for-supervised-learning_4%20%281%29.avif](attachment:general-diagram-of-perceptron-for-supervised-learning_4%20%281%29.avif)

# In[ ]:





# In[220]:


# Taking the data that has been pre-processed and using it for the model


# In[221]:


#Converts the eye-tracking data of the Dyslexic and Control candidates present in the data folder into lists of datafromes
#Each data frame represents the data of 1 candidate
#The entire data is converted into 2 lists:
# 1. C_data for control candidates 
# 2. D_data for dyslexic candidates
#Structure of the dataframes:
#        LX    LY    RX    RY
#    0   ..    ..    ..    .. 
#    1   ..    ..    ..    .. 
#    2   ..    ..    ..    .. 
#   ..   ..    ..    ..    .. 
#    n   ..    ..    ..    .. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get_data()
D_path = glob.glob(r'C:\Users\aswin\Downloads\Recording Data\Dyslexic' + "\*")
C_path = glob.glob(r'C:\Users\aswin\Downloads\Recording Data\Control' + "\*")

C_data = []                                   # Reading individual Control csv files
for path in C_path:
    temp = pd.read_csv(path)
    temp = temp.drop('Unnamed: 0',axis = 1)
    C_data.append(temp)

D_data = []                                   # Reading individual Dyslexia csv files
for path in D_path:
    temp = pd.read_csv(path)
    temp = temp.drop('Unnamed: 0',axis = 1)
    D_data.append(temp)

    
#Get Control and Dyslexic data as required for the STFT operations
# get_stft_data()
C_new = []                                    # Creating a dataframe with 'LX','LY','RX','RY' and removing 0th column
for data in C_data:                           # Control
    X =data[['LX','RX']]
    Y =data[['LY','RY']]
    Xm = X.mean(axis=1)
    Ym = Y.mean(axis=1)
    f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
    f = f.transpose()
    f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
    C_new.append(f)

D_new = []                                    # Creating a dataframe with 'LX','LY','RX','RY' and removing 0th column
for data in D_data:                           # Dyslexic
    X =data[['LX','RX']]
    Y =data[['LY','RY']]
    Xm = X.mean(axis=1)
    Ym = Y.mean(axis=1)
    f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
    f = f.transpose()
    f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
    D_new.append(f)


# In[222]:


# We take absolute value of the reading along X and Y axes

for i in range(len(C_new)):
    C_tempx = np.abs(C_new[i]['X'])
    mx = max(C_tempx)
    C_tempy = np.abs(C_new[i]['Y'])
    my= max(C_tempy)
    C_new[i]['X'] = C_new[i]['X']/np.abs(mx)
    C_new[i]['Y'] = C_new[i]['Y']/np.abs(my)
for i in range(len(D_new)):
    D_tempx = np.abs(D_new[i]['X'])
    mx = max(D_tempx)
    D_tempy = np.abs(D_new[i]['Y'])
    my= max(D_tempy)
    D_new[i]['X'] = D_new[i]['X']/np.abs(mx)
    D_new[i]['Y'] = D_new[i]['Y']/np.abs(my)   


# In[223]:


# Spliiting the Control and Dyslexic data into complex, real and imaginary parts.

C_cmx = []
C_real= []
C_img=[]
for j in range(len(C_new)):
    dat = C_new[j]
    x = dat['X']
    y = dat['Y']
    t = dat['T']
    
    z=[]
    x_in=[]
    y_in=[]
    for i in range(0,x.size):
        z.append(complex(x[i],y[i]))
        x_in.append(x[i])
        y_in.append(y[i])
    
    
    C_cmx.append(z)
    C_real.append(x_in)
    C_img.append(y_in)

    
    
D_cmx = []
D_real= []
D_img=[]
for j in range(len(D_new)):
    dat = D_new[j]
    x = dat['X']
    y = dat['Y']
    t = dat['T']
    
    z=[]
    x_in=[]
    y_in=[]
    for i in range(0,x.size):
        z.append(complex(x[i],y[i]))
        x_in.append(x[i])
        y_in.append(y[i])
    D_cmx.append(z)
    D_real.append(x_in)
    D_img.append(y_in)


# In[ ]:





# # STFT

# # 1. Standard STFT run

# In[224]:


def stft_run(n_ratio,o_ratio):

# Standard STFT Run: Returns Full Flattened Vector
# Input:
#     tmat: Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. n_ratio = Length (L)/Bin width (B)
#     o_ratio: Ratio of Bin_width to Overlap


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) Shape: (t*f)x1 : Flattened C_spec vector. 



    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        
        
        C_spec.append(np.abs(Zxx)**2)
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    print(C_spec[i].shape)
    
    for i in range(len(C_spec)):
        vec[i]=vec[i].flatten()
    
    

    return vec


# # 2. STFT Half Run

# In[225]:


def stft_run_half(n_ratio,o_ratio,lim1,lim2):

# Calculates STFT of selected temporal bins. 
# ------------------------------------------
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# ------------------------------------------
#           lim1            lim2

# Input:
#     tmat   : Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. n_ratio = Length (L)/Bin width (B)
#     o_ratio: Ratio of Bin_width to Overlap
#     lim1   : Lower limit of selected bins
#     lim2   : Upper limit of selected bins


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) : Flattened C_spec vector sliced by bins 




    lim1= int(lim1)
    lim2= int(lim2)
    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        half_im=[]
        for i in range(len(tot)):
            half_im.append(tot[i][lim1:lim2])
            
        
        C_spec.append(np.asarray(half_im))
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    #print(C_spec[i].shape)
    
    for i in range(len(C_spec)):
        vec[i]=vec[i].flatten()
    
    

    return vec


# # 3. STFT Frequency

# In[226]:


# Flattened Vector depending on frequency

def stft_run_freq(n_ratio,o_ratio,lim1,lim2):

# Calculates STFT of selected temporal bins. 
# ------------------------------------------
# |========================================| lim2
# |========================================|
# |========================================| lim1
# |                                        |
# |                                        |
# |                                        |
# ------------------------------------------


# Input:
#     tmat   : Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. n_ratio = Length (L)/Bin width (B)
#     o_ratio: Ratio of Bin_width to Overlap
#     lim1   : Lower limit of selected frequency range
#     lim2   : Upper limit of selected frequency range


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) : Flattened C_spec vector sliced by frequency 




    lim1= int(lim1)
    lim2= int(lim2)
    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        
        
        C_spec.append(np.abs(Zxx)**2)
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    print(C_spec[i].shape)
    
    factor = len(C_spec[i][3])
    l1= factor*lim1
    l2 = factor*lim2
    print(l1,l2)
    lfvec =[]
    hfvec =[]
    fvec=[]
    for i in range(len(C_spec)):
        x=vec[i].flatten()
        #print(len(x))
        #lfvec.append(x[:lim])
        #hfvec.append(x[lim:])
        fvec.append(x[l1:l2])
    
    

    return fvec


# In[ ]:





# # Preparing Data before passing onto STFT ()

# In[227]:


tmat=[]
for i in range(87):
    tmat.append(C_cmx[i])
for i in range(98):
    tmat.append(D_cmx[i])


# In[228]:


#Labelling signals of different lengths
ylen=[]
for s in tmat:
    if(len(s)>1750):
        ylen.append(4)
    elif(len(s)>1500):
        ylen.append(3)
    elif(len(s)>1250):
        ylen.append(2)
    elif(len(s)>1000):
        ylen.append(1)
    else:
        ylen.append(0)


# # Split Train and Test data for analysis

# In[229]:


#Training testing data based on fixed index defined signals
def create_train_test(X,index):
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    index_test =[]
    index_train=index
    for i in range(len(X)):
        if i in index_train:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
            index_test.append(i)
    return X_train, X_test,y_train,y_test,index_train,index_test


# In[230]:


def add_num(ind):
    r=random.randint(0,186)
    if r not in ind:
        ind.append(r)
    else:
        add_num(ind)


# # Getting Misclassified Points

# In[231]:


# Function to get misclassified points

# Get misclassified points form final result 
#INPUT:
#     Res: Distances from separating plane
#  
# OUTPUT:
#      WrongClass: Array of all wrongly classified points
#                 Every element is [index, Length of reading]


def get_misclassified(res):
    WrongClass=[]
    for i in range(88):
        if(res[i]<0):
            WrongClass.append([i,ylen[i]])
    for i in range(88,185):
        if(res[i]>0):
            WrongClass.append([i,ylen[i]])
    return WrongClass
    


# In[232]:


def misc_pts(misc):
    dp=[]
    for i in misc:
        dp.append(i[0])
    print(dp)


# In[ ]:





# # Output Function
- Prints the accuracy score
- Plots the distances of each point from the separating plane.
- Returns the list of all distances
# In[233]:


def final(X_train,y_train,X,y):

# Function to plot final perceptron results.

# scatter plot of all reading with heights from x-axis as distances from the separating plane.
# INPUT:
#      X_train, y_train: Training set readings and labels
# OUTPUT:
#      ht : Distances form separating plane ( +ve: Control side | -ve: Dyslexic side)
#      Perceptron Neural Network score: clf.score()

    clf.fit(X_train, y_train)
    ht = clf.decision_function(X)
    x = range(len(y))
    fig, ax = plt.subplots()
    ax.scatter(x,ht,c = ylen)
    ax.axvline(x=88, color='b', linestyle='-')
    ax.axhline(y=0, color='r', linestyle='-')
    print(clf.score(X,y))
    return ht


# In[ ]:





# # Perceptron Output
X axis: Index of data
Y axis: Distance from separating line
    
There are four quadrants in the output.
The RED LINE represents the output separation line:
If the point is above, it is classified as the control group. Below the red line, it is classified as Dyslexic
The distance of a point from the red line is the distance from the separating plane
The BLUE LINE is the actual label separation:
The points to the left are control group and to the right are dyslexic group.
This makes the TOP LEFT and BOTTOM RIGHT correctly classified as control and dyslexic respectively.
While the BOTTOM LEFT and TOP RIGHT are wrongly classified point.
# In[234]:


# Creating instance of Perceptron to be used for Neural Network
clf = Perceptron(tol=1e-3, random_state=0)
y= np.concatenate((np.ones(87), np.zeros(98)))


# In[235]:


# K-Means Clustering

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)


# In[236]:


import random
index=[]
for i in range(130):
    add_num(index)


# In[237]:


len(index)


# In[ ]:





# In[ ]:


# Standard STFT Run
# Returns Full Flattened Vector C_spec


# In[238]:


vec = stft_run(125,2)
X = np.asarray(vec)
# Train_Test_Split
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)

# Using K-Means Classifier
y_km = Kmean.fit_predict(X)


# In[239]:


# Accuracy Score for Standard STFT run with K-Means Classifier

accuracy_score(y, y_km)


# In[240]:


# Unsupervised learning method in which Perceptrons are used to build Neural Networks for prediction
# Distances from separating plane ( +ve: Control side | -ve: Dyslexic side)

res = final(X_train,y_train,X,y)

clf score is also given


X axis: Index of data
Y axis: Distance from separating line
    
There are four quadrants in the output.
The RED LINE represents the output separation line:
If the point is above, it is classified as the control group. Below the red line, it is classified as Dyslexic
The distance of a point from the red line is the distance from the separating plane
The BLUE LINE is the actual label separation:
The points to the left are control group and to the right are dyslexic group.
This makes the TOP LEFT and BOTTOM RIGHT correctly classified as control and dyslexic respectively.
While the BOTTOM LEFT and TOP RIGHT are wrongly classified point.
# In[241]:


# Misclassified Points
misc1 = get_misclassified(res)


# In[ ]:





# # Partial Bins
We split the data into different bins through a process called Binning.
# In[242]:


# Dataset is divided into groups/bins of 2

# First Half Bins        (0-124)

# K-Means Classifier
vec = stft_run_half(125,2,0,125)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
accuracy_score(y, y_km)


# In[243]:


# Perceptron 
# Distances from separating plane ( +ve: Control side | -ve: Dyslexic side)

res = final(X_train,y_train,X,y)


# In[244]:


misc2_half = get_misclassified(res)


# In[ ]:





# In[245]:


#Second Half Bins    (125-250)

# K-Means Classifier
vec = stft_run_half(125,2,125,251)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
accuracy_score(y, y_km)


# In[246]:


# Perceptron 
# Distances from separating plane ( +ve: Control side | -ve: Dyslexic side)

res = final(X_train,y_train,X,y)


# In[247]:


misc2_half_r = get_misclassified(res)


# In[ ]:





# # Quarter Bins

# In[248]:


#First 25% (Q1) Bin     (0-62)

# K-Means Classifier
vec = stft_run_half(125,2,0,63)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
accuracy_score(y, y_km)


# In[249]:


# Perceptron 
# Distances from separating plane ( +ve: Control side | -ve: Dyslexic side)

res = final(X_train,y_train,X,y)


# In[250]:


misc2_q1 = get_misclassified(res)


# In[ ]:





# In[251]:


#Q2 Bin (25-50%)          (63-124)

vec = stft_run_half(125,2,63,125)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
misc2_q2 = get_misclassified(res)


# In[ ]:





# In[252]:


#Q3 Bin (50-75%)          (125-187)

vec = stft_run_half(125,2,125,188)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
misc2_q3 = get_misclassified(res)


# In[ ]:





# In[253]:


#Q4 Bin (75-100%)          (188-250)   

vec = stft_run_half(125,2,188,251)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
misc2_q4 = get_misclassified(res)


# In[ ]:





# In[254]:


# Best Hyperparameters used on the model to get maximum score

# Standard STFT run
# Returns full flattened C_Spec vector
vec = stft_run(50,5)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
print(get_misclassified(res))


# In[ ]:





# In[255]:


# Best Hyperparameters used on the model to get maximum score

# STFT run half
vec = stft_run_half(50,5,0,64)
X = np.asarray(vec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
print(get_misclassified(res))


# In[ ]:





# In[256]:


# Partial frequencies
# Binnning based on frequency

#Freq Div 35/40
fvec = stft_run_freq(50,5,0,50)
X = np.asarray(fvec)
X_train, X_test,y_train,y_test,index_train,index_test= create_train_test(X,index)
y_km = Kmean.fit_predict(X)
print(accuracy_score(y, y_km))
res = final(X_train,y_train,X,y)
temp = get_misclassified(res)


# In[ ]:





# In[ ]:




