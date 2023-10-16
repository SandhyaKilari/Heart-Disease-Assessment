import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import hiplot as hip

st.title('Heart Disease Dataset')
st.caption('Presented by Sandhya Kilari')
st.divider()

# Load the Dataframe
url = "https://raw.githubusercontent.com/SandhyaKilari/Heart-Disease-Assessment/main/heart.csv"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df_heart = pd.read_csv(url, names=column_names)
df_heart = df_heart.drop(0)

# Define a function to be executed when the checkbox is selected
def my_function():
    st.caption("Heart Disease Data Visualization")
    selected_variable = st.selectbox("Select the desired variable", df_heart.columns)
    scatter_plot = alt.Chart(df_heart).mark_circle(size=60, opacity=0.7).encode(
        x=selected_variable,
        y='target:N',
        tooltip=[selected_variable, 'target']
    ).properties(
        width=600,
        height=400
    )
    st.write(scatter_plot)

# Create a checkbox with a label and associated function
if st.checkbox('Dataset'):
    st.caption("View the first 10 rows of the dataset")
    st.table(df_heart.head(10))

if st.checkbox('Relation between "Target" and other variable'):
    my_function()

# Visualization with HiPlot
st.write("Heart Disease Dataset Visualization with HiPlot")
selected_columns = st.multiselect("Select columns to visualize", df_heart.columns)

# Create a DataFrame with the selected columns
selected_data = df_heart[selected_columns]
st.write(selected_data)

# Create a HiPlot experiment with the selected data
experiment = hip.Experiment.from_dataframe(selected_data)
st.write(experiment)
