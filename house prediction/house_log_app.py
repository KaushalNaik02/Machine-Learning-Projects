import pandas as pd
import streamlit as st
import pickle as pk


model = pk.load(open('log_model.pkl','rb'))
# st.header('House Buying prediction According to Financial Situation')
st.markdown(
    """
    <h1 style='color:black; text-align: center;'>HOUSE BUY PREDICTOR 🏢</h1>
    """,
    unsafe_allow_html=True
)
   
data = pd.read_csv('house_buy.csv',comment='#')



st.markdown(
    """
    <style>
    .stApp {
        background-color:grey;
        padding:15px;
        
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""<h3 style='text-align:center;'>Select Your Gender : </h3>""", unsafe_allow_html=True) 
SEX = st.selectbox('',data['SEX'].unique())

st.markdown('<h3> </h3>', unsafe_allow_html=True)

st.markdown("""<h3 style='text-align:center;'>Select Your Job :</h3>""", unsafe_allow_html=True) 
JOB = st.radio('', data['JOB'].unique())

st.markdown('<h3> </h3>', unsafe_allow_html=True)

st.markdown("""<h3 style='text-align:center;'>Your Salary  :</h3>""",unsafe_allow_html=True)
SALARY= st.slider('',0,99000)

st.markdown('<h3> </h3>', unsafe_allow_html=True)

st.markdown("""<h3 style='text-align:center;'>Your Loan Amount:</h3>""",unsafe_allow_html=True)
LOAN= st.text_input('',)

st.markdown('<h3> </h3>', unsafe_allow_html=True)

if st.button('Predict'):
    input_data = pd.DataFrame(
    [[SEX,JOB,SALARY,LOAN]],
    columns=['SEX','JOB','SALARY','LOAN']
)
    
    st.write(input_data)
    input_data['SEX'].replace(['Male','Female'],[0,1],inplace=True)
    input_data['JOB'].replace(['Government','Private'],[0,1],inplace=True)

    house_pred = model.predict(input_data)
    if (house_pred==0):  
        st.markdown('<h5>Ooops...He/She Cant Able TO Buy the House Due To Financial Situation 🥲 </h5>',unsafe_allow_html=True)
    else:
        st.markdown('<h5>Great.. He/She Can Buy The Dream House 😍 </h5>',unsafe_allow_html=True)
