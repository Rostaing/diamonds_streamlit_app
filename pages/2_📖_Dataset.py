import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from layouts.footer import footer
from layouts.header import header
from layouts.data import get_data


def main():
    header()
    st.subheader("Dataset brute")
    st.dataframe(get_data())
    footer()



if __name__ == "__main__":
    main()