import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("CAR_DETAILS.csv")



lrr = pickle.load(open('lr1_modell.pkl','rb'))
dtt = pickle.load(open('dt1_modell.pkl','rb'))
rff = pickle.load(open('rf1_modell.pkl','rb'))



st.title('Car Price prediction Web App')
st.subheader('Fill the below Details to predict Car Price Charges')


model = st.sidebar.selectbox('Select the ML Model',['Lin_Reg'])


# age 	sex 	bmi 	children 	smoker 	charges 	
# region_northwest 	region_southeast 	region_southwest

name = st.selectbox('name',df['name'].unique())
year = st.slider('year',1990,2023)
km_driven= st.slider('km_driven',0,200000)
fuel= st.selectbox('fuel',['Petrol','Diesel','CNG','LPG','Electric'])
seller_type = st.selectbox('seller_type',['Individual','Dealer','Trustmark_Dealer'])
transmission = st.selectbox('transmission',['Manual','Automatic'])
owner = st.selectbox('owner',['First_Owner','Second_Owner','Third_Owner','Fourth_Above_Owner','Test_Drive_Car'])


if st.button('Predict Car Price Charges'):
    if fuel=='Petrol':
        fuel=1
    elif fuel == "Diesel": 
        fuel=1
    elif fuel == "CNG": 
        fuel=1
    elif fuel == "LPG": 
        fuel=1
    else: 
        fuel=1

    if seller_type == "Individual":
       seller_type = 1
    

    elif seller_type == "Dealer": 
        seller_type = 1

    else:
        seller_type = 1

    if owner=="First_Owner":
       owner=1

    elif owner == "Second_Owner":
        owner=1

    elif owner =="Third_Owner":
       owner=1

    elif owner =="Fourth_Above_Owner":
       owner=1

    else:
       owner=1

    if transmission == "Manual":
       transmission = 1

    else:
        transmission=1


    test = np.array([year,km_driven,fuel,seller_type,transmission,owner])
    test = test.reshape(1,6)
    if model == "Lin_Reg":
        st.success(lrr.predict(test)[0])
    

















# To run Streamlit Web App
# streamlit run app.py





