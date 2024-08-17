import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

iris = load_iris()

# ############ The loaded Dataset Can be shown as following ##########
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species'] = iris.target      # Adds the column 'species' to dataframe

x = df[iris.feature_names]

y = df['species']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred,y_test)
print("accuracy score is :",accuracy)

# input_data = pd.DataFrame(
#      [[6.2,3.4,5.4,2.3]],columns=iris.feature_names
# )
# pred_val =  model.predict(input_data)
# print(pred_val)


st.title('|----- IRIS SPECIES DETECTION------|')

sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.5)

if st.button('Predict'):
    input_data = pd.DataFrame(
        [[sepal_length,sepal_width,petal_length,petal_width]],columns=iris.feature_names
    )
    pred_val = model.predict(input_data)
    species = iris.target_names[pred_val][0]            # Convert numeric prediction to species name
    st.markdown('The Above Species is : ' + str(species))

