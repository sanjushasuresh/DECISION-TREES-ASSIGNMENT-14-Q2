# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 20:06:15 2022

@author: LENOVO
"""

# DECISION TREE CLASSIFIER

import pandas as pd 
import numpy as np

df=pd.read_csv("Company_Data.csv")
df.head()
df.dtypes
df.info()
df.duplicated()
df[df.duplicated()]

df.corr()
df.corr().to_csv("Dtree.csv")

# Boxplots
df.boxplot("CompPrice",vert=False)
Q1=np.percentile(df["CompPrice"],25)
Q3=np.percentile(df["CompPrice"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["CompPrice"]<LW].shape
df[df["CompPrice"]>UW].shape
df["CompPrice"]=np.where(df["CompPrice"]>UW,UW,np.where(df["CompPrice"]<LW,LW,df["CompPrice"]))

df.boxplot("Income",vert=False)
Q1=np.percentile(df["Income"],25)
Q3=np.percentile(df["Income"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Income"]<LW].shape
df[df["Income"]>UW].shape
df["Income"]=np.where(df["Income"]>UW,UW,np.where(df["Income"]<LW,LW,df["Income"]))

df.boxplot("Advertising",vert=False)
df.boxplot("Population",vert=False)

df.boxplot("Price",vert=False)
Q1=np.percentile(df["Price"],25)
Q3=np.percentile(df["Price"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Price"]<LW].shape
df[df["Price"]>UW].shape
df["Price"]=np.where(df["Price"]>UW,UW,np.where(df["Price"]<LW,LW,df["Price"]))

df.boxplot("Age",vert=False)
df.boxplot("Education",vert=False)

# Converting Sales into categorical
df["Sales"] = pd.cut(df["Sales"], bins=[0,4.2,8.01,12.01,16.27],labels=["poor","good","very good","excellent"])
df

# Splitting the variables
Y=df["Sales"]
X=df.iloc[:,1:]
X.columns
X.dtypes

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["CompPrice"]=MM.fit_transform(X[["CompPrice"]])

X["Income"]=MM.fit_transform(X[["Income"]])

X["Advertising"]=MM.fit_transform(X[["Advertising"]])

X["Population"]=MM.fit_transform(X[["Population"]])

X["Price"]=MM.fit_transform(X[["Price"]])

X["Age"]=MM.fit_transform(X[["Age"]])

X["Education"]=MM.fit_transform(X[["Education"]])

X["ShelveLoc"]=LE.fit_transform(X["ShelveLoc"])
X["ShelveLoc"]=pd.DataFrame(X["ShelveLoc"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["US"]=LE.fit_transform(X["US"])
X["US"]=pd.DataFrame(X["US"])
X

Y=LE.fit_transform(df["Sales"])
Y=pd.DataFrame(Y)

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# Model fitting
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(max_depth=5,max_leaf_nodes=15)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)
Y_predtest=DT.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(Y_train,Y_predtrain)
cm2 = confusion_matrix(Y_test,Y_predtest)

ac1 = accuracy_score(Y_train,Y_predtrain) 
ac2 = accuracy_score(Y_test,Y_predtest) 

# If max_depth is 5, max_leaf_nodes is 15 and test_size is 0.2 then ac1=71% and ac2=62%
# The model is overfitting since the training accuracy significantly overpowers the test accuracy

# If max_depth is 5, max_leaf_nodes is 15 and test_size is 0.3 then ac1=68% and ac2=64%
# In this case, eventhough the accuracy is <70 the difference b/w training and test accuracy is <5%


DT.tree_.max_depth # number of levels = 5
DT.tree_.node_count # counting the number of nodes = 29

# Tree visualization
# pip install graphviz
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# To know which is the best max depth value and max leaf node value we are doing gridsearchcv 
from sklearn.model_selection import GridSearchCV

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=DecisionTreeClassifier(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_


# Bagging
from sklearn.ensemble import BaggingClassifier
DT=DecisionTreeClassifier(max_depth=5)
Bag=BaggingClassifier(base_estimator=DT,max_samples=0.9,n_estimators=100)
Bag.fit(X_train,Y_train)                     
Y_predtrain=Bag.predict(X_train)
Y_predtest=Bag.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
ac2=accuracy_score(Y_test,Y_predtest)

# If max_depth is 5 , max samples is 0.2 and n estimators is 100 then ac1=67% and ac2=62%
# If max_depth is 5 , max samples is 0.5 and n estimators is 100 then ac1=78% and ac2=65%
# If max_depth is 5 , max samples is 0.7 and n estimators is 100 then ac1=80% and ac2=66%
# If max_depth is 5 , max samples is 0.9 and n estimators is 100 then ac1=85% and ac2=68%

# Entropy method
Training_accuracy = []
Test_accuracy = []

for i in range(1,12):
    regressor = DecisionTreeClassifier(max_depth=i,criterion="entropy") 
    regressor.fit(X_train,Y_train)
    Y_pred_train = regressor.predict(X_train)
    Y_pred_test = regressor.predict(X_test)
    Training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    Test_accuracy.append(accuracy_score(Y_test,Y_pred_test))


pd.DataFrame(Training_accuracy)
pd.DataFrame(Test_accuracy)
pd.concat([pd.DataFrame(range(1,12)) ,pd.DataFrame(Training_accuracy),pd.DataFrame(Test_accuracy)],axis=1)    


#===================================================================================================#

# DECISION TREE REGRESSOR

import pandas as pd 
import numpy as np

df=pd.read_csv("Company_Data.csv")
df.head()
df.dtypes
df.info()
df.duplicated()
df[df.duplicated()]
df.corr()

# Splitting the variables
Y=df["Sales"]
X=df.iloc[:,1:]
X.columns
X.dtypes

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["CompPrice"]=MM.fit_transform(X[["CompPrice"]])

X["Income"]=MM.fit_transform(X[["Income"]])

X["Advertising"]=MM.fit_transform(X[["Advertising"]])

X["Population"]=MM.fit_transform(X[["Population"]])

X["Price"]=MM.fit_transform(X[["Price"]])

X["Age"]=MM.fit_transform(X[["Age"]])

X["Education"]=MM.fit_transform(X[["Education"]])

X["ShelveLoc"]=LE.fit_transform(X["ShelveLoc"])
X["ShelveLoc"]=pd.DataFrame(X["ShelveLoc"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["US"]=LE.fit_transform(X["US"])
X["US"]=pd.DataFrame(X["US"])
X

Y=LE.fit_transform(df["Sales"])
Y=pd.DataFrame(Y)

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

# Model fitting
from sklearn.tree import DecisionTreeRegressor
DT=DecisionTreeRegressor(max_depth=5,max_leaf_nodes=20)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)
Y_predtest=DT.predict(X_test)

from sklearn.metrics import r2_score
rs1=r2_score(Y_train,Y_predtrain)
# rs1=72%
rs2=r2_score(Y_test,Y_predtest)
#rs2=41%

# Tree visualization
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

#To know which is the best max depth value and max leaf node value we are doing gridsearchcv 
from sklearn.model_selection import GridSearchCV
import numpy as np

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=DecisionTreeRegressor(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_

# Bagging
from sklearn.ensemble import BaggingRegressor
DT=DecisionTreeRegressor(max_depth=5)
Bag=BaggingRegressor(base_estimator=DT,max_samples=0.4,n_estimators=100)
Bag.fit(X_train,Y_train)                     
Y_predtrain=Bag.predict(X_train)
Y_predtest=Bag.predict(X_test)

from sklearn.metrics import r2_score
rs1=r2_score(Y_train,Y_predtrain)
# rs1=81%
rs2=r2_score(Y_test,Y_predtest)
# rs2=68%

# Squared-error method
Training_accuracy = []
Test_accuracy = []

for i in range(1,12):
    regressor = DecisionTreeRegressor(max_depth=i,criterion="squared_error") 
    regressor.fit(X_train,Y_train)
    Y_pred_train = regressor.predict(X_train)
    Y_pred_test = regressor.predict(X_test)
    Training_accuracy.append(r2_score(Y_train,Y_pred_train))
    Test_accuracy.append(r2_score(Y_test,Y_pred_test))
    
    
pd.DataFrame(Training_accuracy)
pd.DataFrame(Test_accuracy)
pd.concat([pd.DataFrame(range(1,12)) ,pd.DataFrame(Training_accuracy),pd.DataFrame(Test_accuracy)],axis=1)    


# Therefore after applying bagging rs1 is 81% and rs2 is 68% 