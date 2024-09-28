import streamlit as st
import pandas as pd
from layouts.footer import footer
from layouts.header import header
from layouts.data import get_data


def analyse():
    data = pd.DataFrame(get_data())

    select = st.sidebar.selectbox("Select", ["Head", "Tail", "Shape", "Type", "Isnull", "Describe"])
    
    if select == "Head":
         st.write("Head", data.head())
    elif select == "Tail":
        st.write("Tail: ", data.tail())
    elif select == "Shape":
        st.write("Shape:", data.shape)
    elif select == "Type":
        st.write("Type:", data.dtypes.to_frame("Types"))
    elif select == "Isnull":
        st.write("Isnull:", data.isnull().sum().to_frame("Null"))
    elif select == "Describe":
        st.write("Describe: ", data.describe())
   

def main():
    header()
    analyse()
    footer()

if __name__ == "__main__":
    main()