from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_excel("./Credit_Card_Data.xlsx")
df.to_csv("./UCL_Credit_Card_data.csv", sep=",",header = ['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAYMENT_STATUS'])
#print(df.head)


# y= np.array(df.loc['PAYMENT_STATUS'])
# x=df.drop('PAYMENT_STATUS',axis =1)

# x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

x_train = train.iloc[:,1:6]
y_train = train.iloc[:,6:7]
x_test = train.iloc[:,1:6]
y_test = train.iloc[:,6:7]
print(y_train.head)

clf = LogisticRegression(C = 100)
clf.fit(x_train, y_train.values.ravel())

accuracy = clf.score(x_test, y_test)
print(accuracy)