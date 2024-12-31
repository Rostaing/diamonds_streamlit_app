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
import plotly.express as px


# Pré-traitement des données
def data_preprocessing():
    df = get_data()

    # Encodage des colonnes catégoriques
    df_cut = pd.get_dummies(df["cut"], dtype=int)
    df_color = pd.get_dummies(df["color"], dtype=int)
    df_clarity = pd.get_dummies(df["clarity"], dtype=int)

    # Concaténation des colonnes encodées
    data = pd.concat([df, df_cut, df_color, df_clarity], axis=1)
    data = data.drop(["cut", "color", "clarity"], axis=1)

    return data


# Système de prédiction
def pred():
    col1, col2, col3 = st.columns(3)

    # Inputs utilisateur
    with col1:
        carat = float(st.number_input('Carat', min_value=0.0))
    with col2:
        table = int(st.number_input('Table', min_value=0))
    with col3:
        x = float(st.number_input('x length in mm (0--10.74)', min_value=0.0, max_value=10.74))
    with col1:
        y = float(st.number_input('y width in mm (0--58.9)', min_value=0.0, max_value=58.9))
    with col2:
        z = float(st.number_input('z depth in mm (0--31.8)', min_value=0.0, max_value=31.8))

    # Inputs encodés
    inputs = {}
    for col, options in [("cut", ["Fair", "Good", "Ideal", "Premium", "Very Good"]),
                         ("color", ["D", "E", "F", "G", "H", "I", "J"]),
                         ("clarity", ["I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"])]:
        for option in options:
            key = f"{option.lower()} ({col})"
            with col1 if len(inputs) % 3 == 0 else col2 if len(inputs) % 3 == 1 else col3:
                inputs[option] = int(st.number_input(key, min_value=0, max_value=1))

    if st.button('Price Result'):
        depth = (2 * z) / (x + y)  # Calcul de la profondeur
        model = load("resources/best_model.pkl")

        # Préparer les données d'entrée pour le modèle
        X_test = [[carat, depth, table, x, y, z] + list(inputs.values())]
        price_prediction = round(model.predict(X_test).flatten()[0], 2)

        # Afficher le résultat
        with st.spinner("In progress..."):
            time.sleep(5)
            response = f"The price of this diamond💎 is {price_prediction} $."
            st.success(response)
            st.balloons()

# Modélisation et visualisation
def model():
    df = data_preprocessing()

    # Division des données
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Charger le meilleur modèle
    with open("resources/best_model.pkl", "rb") as f:
        model = joblib.load(f)

    # Prédictions
    y_pred = model.predict(X_test)
    data_comp = pd.DataFrame({"Actual": y_test, "Prediction": y_pred, "Residual": y_test - y_pred})

    # Onglets Streamlit
    tab1, tab2, tab3, tab4 = st.tabs(["Processed Data", "Model Evaluation", "Make Prediction", "Visualization"])

    with tab1:
        st.dataframe(df)

    with tab2:
        st.write("Hist Gradient Boosting Regressor Score: ", model.score(X_test, y_test))
        st.dataframe(data_comp)

    with tab3:
        pred()

    with tab4:
        # Graphique avec Matplotlib
        fig, ax = plt.subplots()
        visualizer = PredictionError(model, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        # Affichage du graphique dans Streamlit
        st.pyplot(fig)

        # Optionnel : Visualisation alternative avec Plotly
        fig_plotly = px.scatter(
            data_comp, x="Actual", y="Prediction",
            title="Prediction Error",
            labels={"Actual": "Valeurs réelles", "Prediction": "Prédictions"},
            trendline="ols"
        )
        fig_plotly.add_shape(
            type="line", x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dot")
        )
        # st.plotly_chart(fig_plotly)


def main():
    header()
    model()
    footer()


if __name__ == "__main__":
    main()