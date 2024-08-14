import numpy as np
import pandas as pd
import streamlit as st
import pickle as pk
model = pk.load(open('lin_model.pkl','rb'))
st.header('Car Price Prediction ML Model')

num_data =  pd.read_csv('car.csv') 

# car_name = num_data['name']
name=st.selectbox('Select Car Brand',num_data['name'].unique())
fuel=st.selectbox('Select Car Fuel-Type',num_data['fuel'].unique())
average=st.slider('Aproximate Mileage Of Car',10,30)
launch=st.slider('Launching Year Of Car',2004,2024)
seats=st.slider('Number of Seats in Car',5,9)
horsepower=st.slider('Horsepower Of Car',100,900)

if st.button('Predict'):
    input_data = pd.DataFrame(
    [[name,fuel,average,launch,seats,horsepower]],
    columns=['name','fuel','average','launch','seats','horsepower']
)
    
    input_data['name'].replace(['santro','ertiga','swift','omni','audi','xuv'],[1,2,3,4,5,6],inplace=True)
    input_data['fuel'].replace(['petrol','diesel','cng'],[1,2,3],inplace=True)

    st.write(input_data)

    car_price = model.predict(input_data)
    st.markdown('Car Price Is going To Be: '+str(car_price[0]))