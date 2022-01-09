"Task 1"

import pandas as pd
import numpy as np

glass = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/KNN/glass.csv")


glass_1 = glass.iloc[:, 0:9] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass_1)
glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors # We Can do random partioning as well as sequenctial partioning like in r
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21) # 21 short distances - it should be always odd number
knn.fit(X_train, Y_train)


pred = knn.predict(X_test)
pred

#Performance Metrics On MultiClass Classification Problems
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 
# In training data set also has 11 FN - Not good

# creating empty list variable 
acc = []

# To identify best k value we use for loop
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2): # 3,5,7,9.....
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train) #we fit on training dataset
    train_acc = np.mean(neigh.predict(X_train) == Y_train) # training accuracy for different value of k
    test_acc = np.mean(neigh.predict(X_test) == Y_test) # test accuracy for different value of k
    acc.append([train_acc, test_acc]) # append training acuracy, test accuracy


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-") #i[0]

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-") #i[1]

# On the basis of graph we find that k=15 is the best k value(shortest distance) where, test accuracy = 67%, train accuracy = 66% they both are close enough.
#Note: Training accuracy and test accuracy should close like , 98%-97%, not 80%-60%, if equals then good


