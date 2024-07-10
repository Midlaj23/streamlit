import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/diabetes.csv")


X = data.drop('Outcome',axis = 1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

st.title("Diabetes Prediciton")

form = st.form("User Information Form")
with form:
    age = st.number_input(label="Enter Age",min_value=data['Age'].min(),max_value=data['Age'].max())
    glucose = st.number_input(label="Enter Glucose Level",min_value=data['Glucose'].min(),max_value=data['Glucose'].max())
    pregnancies = st.number_input(label="Enter the number of pregnancies",min_value=data['Pregnancies'].min(),max_value=data['Pregnancies'].max())
    blood_pressure = st.number_input(label="Enter BP measurement",min_value=data['BloodPressure'].min(),max_value=data['BloodPressure'].max())
    skin_thickness = st.number_input(label="Enter Thickness of the skin",min_value=data['SkinThickness'].min(),max_value=data['SkinThickness'].max())
    insulin = st.number_input(label="Enter Insulin level",min_value=data['Insulin'].min(),max_value=data['Insulin'].max())
    bmi = st.number_input(label="Enter Body Mass Index",min_value=data['BMI'].min(),max_value=data['BMI'].max())
    diabetes_pedigree = st.number_input(label="Enter Diabetes Percentage",min_value=data['DiabetesPedigreeFunction'].min(),max_value=data['DiabetesPedigreeFunction'].max())
    submit = form.form_submit_button("Submit")

if submit:
    input =pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    Prediction = model.predict(input)
    if Prediction[0] == 0:
        st.write("### Predicted Diabetes Outcome: No (Non-Diabetic)")
    else:
       st.write("### Predicted Diabetes Outcome: Yes (Diabetic)")




