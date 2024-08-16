import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

ogdata = pd.read_csv('house_buy.csv',comment='#')

data = pd.read_csv('house_buy.csv',comment='#')
data['SEX'].replace(['Male','Female'],[0,1],inplace=True)
data['JOB'].replace(['Government','Private'],[0,1],inplace=True)


x = data.drop(columns=['HOUSE_BUY'])
y = data['HOUSE_BUY']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# ################# WE CAN CHECK THE MODEL PREDICTION BELOW BY CUSTOM INPUT VALUES IN DATAFRAME #####################

# input_data = pd.DataFrame(
#     [[1,1,45000,0]],columns=['SEX','JOB','SALARY','LOAN']
# )
# print(input_data)

# pred_val = model.predict(input_data)
# # print(pred_val)


# if (pred_val==0):  
#     print('Ooops...He/She Cant Able TO Buy the House Due To Financial Situation ')
# else:
#     print('Great.. He/She Can Buy The Dream House ')


import pickle as pk
pk.dump(model,open('log_model.pkl','wb'))

