# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
df.head(5)
X = df.iloc[:,[0,2,3,4,5,6,7,8,9]]
y = df.loc[:,'list_price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        

import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns

fig, axes = plt.subplots(nrows = 3, ncols = 3)
for i in range(3):
    for j in range(3):
        col = cols[i * 3 + j]        
        axes[i,j].scatter(X_train[col], y_train)
        

plt.show()       

# code ends here



# code ends here



# --------------
# Code starts here
# Code starts here
corr = round(X_train.corr(),2)
print("Correlation Table : \n",corr)
X_train.drop(['play_star_rating', 'val_star_rating'], axis = 1, inplace = True)
X_test.drop(['play_star_rating', 'val_star_rating'], axis = 1, inplace = True)


# Code ends here


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(mse)
r2 = r2_score(y_test,y_pred)
print(r2)


# Code ends here


# --------------
# Code starts here
residual = y_test - y_pred
plt.hist(residual)
plt.title("Residual Histogram")
plt.show()



# Code ends here



# Code ends here


