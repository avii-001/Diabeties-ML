#streamlit run "C:\Users\kusum\Diabeties-ML\Diabeties_Prediction.py"


import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open("C:/Users/kusum/Diabeties-ML/trained_model.sav",'rb'))

def diabeties_prediction(input_data):

    #changing input_data into numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabeties Prediction')

    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('BloodPressure Level')
    SkinThickness=st.text_input('SkinThickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction')
    Age=st.text_input('Age')

    diagonsis=''

    if st.button('Diabetes Test Result'):
        diagonsis=diabeties_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin,
               BMI, DiabetesPedigreeFunction, Age])

        
        st.success(diagonsis)

if __name__ =='__main__':
    main()