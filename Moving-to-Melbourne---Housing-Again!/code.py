# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
#load the dataframe
df = pd.read_csv(path)

#Indepenent varibles
X = df.drop('Price',axis=1)

# store dependent variable
y = df['Price']

# spliting the dataframe

X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3,random_state=6)

# check correlation
corr=X_train.corr()

# print correlation
print(corr)

#Code ends here



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
r2 = regressor.score (X_test,y_test)
print(r2)


# --------------
from sklearn.linear_model import Lasso


# Code starts here
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = lasso.score(X_test,y_test)
print(r2_lasso)


# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)
r2_ridge = ridge.score(X_test,y_test)
print(r2_ridge)
# Code ends here





# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor=LinearRegression()

# Initiate cross validation score
score= cross_val_score(regressor,X_train,y_train ,scoring= 'r2' ,cv=10)
print(score)
#calculate mean of the score
mean_score = np.mean(score)

# print mean score
print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

# Fit the model on training set
model.fit(X_train, y_train)

# predict the model performance
y_pred = model.predict(X_test)

# calculate r2 score
r2_poly= r2_score(y_test,y_pred)

# print r2 score
print(r2)


