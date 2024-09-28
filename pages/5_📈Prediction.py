from layouts.footer import footer
from layouts.header import header
from layouts.data import get_data
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from yellowbrick.regressor import PredictionError
from joblib import dump, load
import joblib
import matplotlib.pyplot as plt
import numpy as np
import time


# Pré-traitement de données
def data_preprocessing():

    df = get_data()

    df_cut = pd.get_dummies(df["cut"], dtype=int)
    df_color = pd.get_dummies(df["color"], dtype=int)
    df_clarity = pd.get_dummies(df["clarity"], dtype=int)

    data = pd.concat([df, df_cut, df_color, df_clarity], axis=1)
    data = data.drop(["cut", "color", "clarity"], axis=1)

    return data

# Prediction Sytem
def pred():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carat = float(st.number_input('Carat', min_value=0.0)) # ,step=1, format="%i"
        
    with col2:
        table = int(st.number_input('Table', min_value=0))
        
    with col3:
        x = float(st.number_input('x length in mm (0--10.74)', min_value=0.0, max_value=10.74))
        
    with col1:
         y = float(st.number_input('y width in mm (0--58.9)', min_value=0.0, max_value=58.9))
        
    with col2:
        z = float(st.number_input('z depth in mm (0--31.8)', min_value=0.0, max_value=31.8))
        
    with col3:
        fair = int(st.number_input('Fair (cut)', min_value=0, max_value=1))
        
    with col1:
        good = int(st.number_input('Good (cut)', min_value=0, max_value=1))
        
    with col2:
        ideal = int(st.number_input('Ideal (cut)', min_value=0, max_value=1))
        
    with col3:
        premium = int(st.number_input('Premium (cut)', min_value=0, max_value=1))
        
    with col1:
        very_good = int(st.number_input('Very good (cut)', min_value=0, max_value=1))
        
    with col2:
        d = int(st.number_input('D (color)', min_value=0, max_value=1))
        
    with col3:
        e = int(st.number_input('E (color)', min_value=0, max_value=1))
        
    with col1:
        f = int(st.number_input('F (color)', min_value=0, max_value=1))
    with col2:
        g = int(st.number_input('G (color)', min_value=0, max_value=1))
    with col3:
        h = int(st.number_input('H (color)', min_value=0, max_value=1))
    with col1:
        i = int(st.number_input('I (color)', min_value=0, max_value=1))
    with col2:
        j = int(st.number_input('J (color)', min_value=0, max_value=1))
    with col3:
        i1 = int(st.number_input('I1 (clarity)', min_value=0, max_value=1))
    with col1:
        IF = int(st.number_input('IF (clarity)', min_value=0, max_value=1))
    with col2:
        si1 = int(st.number_input('SI1 (clarity)', min_value=0, max_value=1))
    with col3:
        si2 = int(st.number_input('SI2 (clarity)', min_value=0, max_value=1))
    with col1:
        vs1 = int(st.number_input('VS1 (clarity)', min_value=0, max_value=1))
    with col2:
        vs2 = int(st.number_input('VS2 (clarity)', min_value=0, max_value=1))
    with col3:
        vvs1 = int(st.number_input('VVS1 (clarity)', min_value=0, max_value=1))
    with col1:
        vvs2 = int(st.number_input('VVS2 (clarity)', min_value=0, max_value=1))
    
    if st.button('Price Result'):

        depth =  ((2 * z ) / (x + y))
        model = load("resources/best_model.pkl")
        X_test = [[carat, depth, table, x, y, z, fair, good, ideal, premium, very_good, d, e, f, g, h, i, j, i1, IF, si1, si2, vs1, vs2, vvs1, vvs2]]
        price_prediction = round(model.predict(X_test).flatten()[0], 2)                      
        
        with st.spinner("In progress..."):
            time.sleep(5)
            response = f"The price of this diamond💎 is {price_prediction} $."
            st.success(response)
            st.balloons()

# Modélisation
def model():

    df = data_preprocessing()

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Model 
    # model = HistGradientBoostingRegressor(random_state=0)
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    # y_pred = model.predict(X_test)

    # model = load("resources/best_model.pkl")
    with open("resources/best_model.pkl", "rb") as f:
        model = joblib.load(f)

    y_pred = model.predict(X_test)
    data_comp = pd.DataFrame({"Actuel":y_test, "prediction":y_pred, "Residual":y_test - y_pred})



    tab1, tab2, tab3, tab4 = st.tabs(["Precessed data", "Models", "Make prediction", "Model visualization"])

    with tab1:
        st.dataframe(df)

    with tab2:
        st.write("Hist Gradient Boosting Regressor: ", model.score(X_test, y_test))
        st.write("", data_comp)

    with tab3:
        pred()

    with tab4:
        fig = plt.figure()
        visualizer = PredictionError(model)

        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()           
        st.plotly_chart(fig)
        
def main():
    header()
    model()
    footer()

if __name__ == "__main__":
    main()