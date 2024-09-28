from layouts.footer import footer
from layouts.header import header
import streamlit as st
from layouts.data import get_data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def viz():

    df = get_data()
    v = st.sidebar.selectbox("Chart Type numericals", ("Bar", "Correlation", "Line", "Histogram", "Violin", "Density", "Heatmaps", "Scatter", "Area", "Funnel", "Pie", "Box", "Ecdf"))
    
    numeric_cols = df.select_dtypes(exclude=['object']).columns.to_list()
    categories = df.select_dtypes(include=['object']).columns.to_list()
    var_x = st.sidebar.selectbox("X axis", numeric_cols)
    var_y = st.sidebar.selectbox("Y axis", numeric_cols)
    var_color = st.sidebar.selectbox("Marker", categories)

    if v == "Bar":
        fig = px.bar(df, x=var_x, y=var_y, color=var_color)
        st.plotly_chart(fig)
        
    elif v == "Correlation":
        data = df.select_dtypes(exclude=["object"])
        fig = plt.figure(figsize=(22, 9))
        sns.heatmap(data.corr(), annot=True, center=True) 
        st.pyplot(fig)
        
    elif v == "Line":
        fig = px.line(df, x=var_x, color=var_color)
        st.plotly_chart(fig)
        
    elif v == "Histogram":
        fig = px.histogram(df, x=var_x, y=var_y, color=var_color, marginal="rug", hover_data=df.columns)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
        
    elif v == "Violin":
        fig = px.violin(df, x=var_x, y=var_y, color=var_color, box=True, points="all", hover_data=df.columns)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
    
    elif v == "Density":
        fig = px.density_contour(df, x=var_x, y=var_y)
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
        
    elif v == "Heatmaps":
        fig = px.density_heatmap(df, x=var_x, y=var_y, marginal_x="rug", marginal_y="histogram")
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        st.plotly_chart(fig)
        
                
    elif v == "Scatter":
        fig = px.scatter(data_frame=df, x=var_x, y=var_y, color=var_color, size_max=60) # size=df.size, animation_frame=df.size,
        st.plotly_chart(fig)
        
    elif v == "Area":
        fig = px.area(df, x=var_x, y=var_y, color=var_color, line_group=var_color)
        st.plotly_chart(fig)
        
    elif v == "Funnel":
        fig = px.funnel(df, x=var_x, y=var_y, color=var_color)
        st.plotly_chart(fig)
        
    elif v == "Pie":
        var = st.sidebar.selectbox("Chart type Categories", categories)
        fig = px.pie(values=df[var].value_counts(), names=df[var].value_counts().index)
        st.plotly_chart(fig)
        
    elif v == "Box":
        fig = px.box(df, x=var_x, color=var_color, notched=True)
        st.plotly_chart(fig)
    
    elif v == "Ecdf":
        fig = px.ecdf(df, x=var_x, color=var_color)
        st.plotly_chart(fig)

def main():

    header()
    viz()
    footer()

if __name__ == "__main__":
    main()