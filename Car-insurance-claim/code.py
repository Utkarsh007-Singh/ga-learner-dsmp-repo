# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)

print(df.head())

print(df.info())

for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']:
    df[col] = df[col].str.replace("$", '')
    df[col] = df[col].str.replace(",", '')

print(df.head())

X = df.drop(columns='CLAIM_FLAG')
y = df['CLAIM_FLAG']

count = y.value_counts()
print(count)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=6)



# Code ends here


# --------------
# Code starts here
for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']:
    X_train[col] = X_train[col].astype(float)
    X_test[col] = X_test[col].astype(float)

print(X_train.info())
print(X_test.info())



# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'], inplace=True)

# check if data has dropped
print(X_train.info())
print(X_test.info())


# udpate y_train and y_test
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

# check the y shape
print(y_test.shape, X_test.shape)

# fill missing value 
for col in ['AGE', 'CAR_AGE', 'INCOME', 'HOME_VAL']:
    X_train[col].fillna(value=X_train[col].mean(), inplace=True)
    X_test[col].fillna(value=X_train[col].mean(), inplace=True)

# check updated values 
print(X_train.info())

# Code ends here


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for col in columns:
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print(X_train[columns].head())


# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)

# predict on test data
y_pred  = model.predict(X_test)

# check accuracy score
score = accuracy_score(y_test, y_pred)
print("Accuracy Score: {}".format(score))

# check precision 
precision = precision_score(y_test, y_pred)
print("Precision Socre: {}".format(precision))



# Code ends here



# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)
X_train, y_train = smote.fit_sample(X_train, y_train)
print(X_train.shape, y_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# Code ends here



# Code ends here


# --------------
# Code Starts here
# Code Starts here

model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)

# predict on test data
y_pred  = model.predict(X_test)

# check accuracy score
score = accuracy_score(y_test, y_pred)
print("Accuracy Score: {}".format(score))

# Code ends here


# Code ends here


