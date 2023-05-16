# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NAVEEN KUMAR.M
RegisterNumber:  212221040113
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

## Placement Data
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/dccae15f-09d0-4815-be99-431a1ed4af68)

## Salary data
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/bc2bfd7f-7684-4b3d-aa2b-cca3e6e7ae61)

## Checking the null() function
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/b90a9287-7cb3-4d81-acc1-5648de99a090)

## Data Duplicate
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/a99c7d7c-8ca2-4cbf-94e3-82d57bb17bab)

## Print data
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/a716d1fe-4c14-4268-89e9-95f125bca031)

## Data-status
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/5bb8be21-f74d-4925-a298-45c34c725dc7)
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/feea8578-9e5c-4cc5-9596-a67efe537ded)

## y_prediction array
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/a3fd98bc-5df4-45a7-b1c6-595417f124f9)

## Accuracy value
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/fdd16af9-6c92-45ac-86da-52653d7bc2b8)

## Confusion array
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/d2e08171-202c-42af-93fa-45406b84a74a)

## Classification report
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/08350f7d-4db6-4b54-9165-302f6061a614)

## Prediction of LR
![image](https://github.com/NaveenKumar-008/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135244/ec762823-1fef-42dc-9380-0fedce82324d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
