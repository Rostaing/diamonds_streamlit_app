import streamlit as st
import pandas as pd


def get_data():
    df = pd.read_csv("resources/diamonds.csv", index_col=0)
    return df

def main():
    get_data()

if __name__ == "__main__":
    main()