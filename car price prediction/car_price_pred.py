import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_csv('car.csv')
# print(data)

num_data =  pd.read_csv('car.csv')
num_data['name'].replace(['santro','ertiga','swift','omni','audi','xuv'],[1,2,3,4,5,6],inplace=True)
num_data['fuel'].replace(['petrol','diesel','cng'],[1,2,3],inplace=True)
# print(num_data)

input_data = num_data.drop(columns=['selling_price'])
output_data = num_data['selling_price']
# print(input_data)
# print(output_data)

(x_train,x_test,y_train,y_test)=train_test_split(input_data,output_data,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# print(y_pred)

input_data = pd.DataFrame(
    [[2,1,20,2021,7,800]],columns=['name','fuel','average','launch','seats','horsepower']
)
print(input_data)

print(model.predict(input_data))


import pickle as pk
pk.dump(model,open('lin_model.pkl','wb'))