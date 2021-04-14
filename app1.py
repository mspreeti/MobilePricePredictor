import pickle

import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image

from app import model1

pickle_in = open("model1.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome"

def predict_pricerange(ram,px_height,battery_power,px_width):

        """Let's help in determining the cost of the mobile.
        This is using docstrings for specifications.
        ---
        parameters:
          - name: ram
            in: query
            type: number
            required: true
          - name: px_height
            in: query
            type: number
            required: true
          - name: battery_power
            in: query
            type: number
            required: true
          - name: px_width
            in: query
            type: number
            required: true
        responses:
            200:
                description: The prediction of price range is:

        """

        prediction = classifier.predict([[ram,px_height,battery_power,px_width]])
        print(prediction)
        return prediction


def main():
    st.title("Mobile Price Generator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Mobile Price Generator App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    ram = st.number_input("Ram")
    px_height = st.number_input("PX Height")
    battery_power = st.number_input("Battery Power")
    px_width = st.number_input("PX Width")
    result = ""
    if st.button("Predict"):
        result=predict_pricerange(ram,px_height,battery_power,px_width)

    st.success('The price range for the mobile phone is {}'.format(result))

if __name__=='__main__':
    main()


