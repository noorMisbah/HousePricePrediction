import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read file and check for duplicated or missing values
df=pd.read_csv('/content/housepriceprediction.csv')
df.duplicated().sum()
df.isnull().sum()

#remove unnecessary columns
df=pd.DataFrame(df)
df=df.drop('city',axis=1)
df=df.drop('street',axis=1)
df=df.drop('country',axis=1)
df=df.drop('date',axis=1)
df=df.drop('statezip',axis=1)

#split into 20%training and 80% testing
X=df.iloc[:,1:-2]
Y=df.iloc[:,0:1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

#apply linear regression
lr=LinearRegression()
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)

#testing accuracy
print("MAE",mean_absolute_error(Y_test,y_pred))
print("MSE",mean_squared_error(Y_test,y_pred))
print("R2",r2_score(Y_test,y_pred))
lr.coef_
lr.intercept_
