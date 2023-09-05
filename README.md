# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.

2. Assigning hours to x and scores to y.
   
3. Plot the scatter plot.
 
4. Use mse,rmse,mae formula to find the values.

## Program:
```python

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vikash A R
RegisterNumber:  212222040179
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

To read csv file

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/9c4c9092-d20a-4bf4-aeb3-c0b5c3d5c913)

To Read Head and Tail Files

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/c1ac175f-b39c-4ed3-9236-d49eed3da214)

Compare Dataset

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/d6551083-db41-4e0e-8f48-52da085630e8)

Predicted Value

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/f01d961d-274e-47e5-96a1-6e32371fdaa0)

Graph For Training Set

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/e5aef6c8-dc0c-421b-8c90-8a1166715f8d)

Graph For Testing Set

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/876779cd-bc6c-4a3d-91fb-bdcc98602f53)

Error

![image](https://github.com/VIKASHAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405655/d257abd8-a7a5-4f1b-a4c2-0bce8db497b4)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
