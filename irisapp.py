import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/iris.csv")

X = df.drop("species",axis = 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)

st.title("Flower Species Predictor")

form = st.form("iris classification form")
with form:
    sepal_length = st.number_input(label="enter sepal length",min_value=df["sepal_length"].min(),max_value=df["sepal_length"].max())
    sepal_width = st.number_input(label="enter sepal width",min_value=df["sepal_width"].min(),max_value=df["sepal_width"].max())
    petal_length = st.number_input(label="enter petal length",min_value=df["petal_length"].min(),max_value=df["petal_length"].max())
    petal_width = st.number_input(label="enter petal width",min_value=df["petal_width"].min(),max_value=df["petal_width"].max())
    submit = form.form_submit_button("Submit")

if submit:
    input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                         columns=["sepal_length","sepal_width","petal_length","petal_width"])
    prediction = model.predict(input)[0]

    st.balloons()
    st.write(f"The predicted species is:  {prediction}")
    st.toast("Prediction complete!")

