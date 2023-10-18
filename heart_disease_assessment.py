import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import hiplot as hip
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

st.markdown("<h1 style='text-align: center; font-size: 35px;'>Heart Disease Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 15px;'>Presented by Sandhya Kilari</p>",
    unsafe_allow_html=True
)

# Load the Dataframe
url = "https://raw.githubusercontent.com/SandhyaKilari/Heart-Disease-Assessment/main/heart.csv"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df_heart = pd.read_csv(url, names=column_names)
df_heart = df_heart.drop(0)
df_heart.dropna()

# Information about the App
st.sidebar.subheader("About the Application")
st.sidebar.info("This web application will enable users to input their health attributes (e.g., age, sex, cholesterol levels, blood pressure, blood sugar level and more) and receive a risk assessment for heart disease. The app will provide a clear prediction that's easy to understand. It will also explain why each detail is important. This tool helps people understand their health and assists doctors when talking to patients.")
sidebar_placeholder = st.sidebar.empty()

tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Data and its Statistics", "Data Visualization", "Model Prediction"])

with tab1:
    st.markdown("<div style='text-align: justify'>This project is very important because it deals with a serious health problem that affects many people. Heart disease is one of the top reasons why people die worldwide, and finding it early is crucial for helping patients get better and reducing the cost of healthcare. By using data science and machine learning, we have a chance to create a tool that can save lives and make people more aware of their health. This project helps individuals, doctors, and society as a whole by giving them a useful way to understand and manage the risk of heart disease.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("<div style='text-align: justify'>It aims to develop a predictive model capable of assessing an individual's risk of developing heart disease by analyzing relevant health attributes. This undertaking holds considerable importance for healthcare professionals, policymakers, and individuals invested in heart disease prevention. It has the potential to enhance early intervention, leading to life-saving outcomes and reduced healthcare expenditures.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Dataset**")
    st.markdown("<div style='text-align: justify'>The Cleveland dataset which is widely used in heart disease research comprises 303 instances and 14 attributes, encompassing variables such as age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol level (chol), fasting blood sugar (fbs), maximum heart rate achieved (thalach), oldpeak, thal, and the target variable indicating presence of heart disease in the patient (0 = no disease, 1 = disease)</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Attributes Information**")
    st.markdown("1. Age")
    st.markdown("2. Sex")
    st.markdown('3. Chest Pain Type ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic")')
    st.markdown("4. Resting Blood Pressure in mm Hg")
    st.markdown("5. Serum Cholesterol in mg/dL")
    st.markdown("6. Fasting Blood Sugar > 120 mg/dL")
    st.markdown('7. Resting Electrocardiographic Results ("Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy")')
    st.markdown("8. Maximum Heart Rate Achieved")
    st.markdown("9. Exercise Induced Angina")
    st.markdown("10. oldpeak = ST depression induced by exercise relative to rest")
    st.markdown("11. The slope of the peak exercise ST segment")
    st.markdown("12. Number of Major Vessels Colored by Fluoroscopy: 0-3")
    st.markdown("13. Thalium Stress Result (0 = normal; 1 = fixed defect; 2 = reversable defect)")

    # if st.checkbox('View the dataset'):
    #    st.write('First 10 rows of dataset')
    #    st.table(df_heart.head(10))

    st.markdown("**Reference Link:**")
    st.markdown("https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")

with tab2:
    if st.checkbox('Dataset'):
        row = st.selectbox("Number of rows", range(1, 1025, 1))
        st.write(f"View the first {row} rows of the dataset")
        st.table(df_heart.head(row))
    if st.checkbox('Descriptive Statistics'):
        st.write(df_heart.describe())
        st.write('Information about the data:')
        st.write(f'Total Number of Samples: {df_heart.shape[0]}')
        st.write(f'Number of Features: {df_heart.shape[1]}')

with tab3:
    # Check for user's input to visualize the data
    if st.checkbox('Select desired X and Y Variables and visualize the data'):
        st.markdown("Data Visualization")

        x_variable = st.selectbox("X Variable", df_heart.columns)
        y_variable = st.selectbox("Y Variable", df_heart.columns)

        data_button = st.selectbox('Please choose preferred visualization', ['Scatter Plot', 'Heatmap', 'Histogram Plot', 'Line Plot', 'Boxplot', 'Relational Plot', 'Distribution Plot'])

        if data_button == 'Scatter Plot':
            scatter_plot = sns.scatterplot(data=df_heart, x=x_variable, y=y_variable)
            st.pyplot(scatter_plot.figure)

        elif data_button == 'Heatmap':
            plt.figure(figsize=(10,10))
            heatmap = sns.heatmap(df_heart.corr(numeric_only=True), annot=True, cmap="crest")
            st.pyplot(heatmap.figure)

        elif data_button == 'Histogram Plot':
            histplot = sns.histplot(data=df_heart, x=x_variable, binwidth=5)
            st.pyplot(histplot.figure)

        elif data_button == 'Line Plot':
            lineplot = sns.lmplot(x=x_variable, y=y_variable, hue="target", data=df_heart)
            st.pyplot(lineplot.figure)

        elif data_button == 'Boxplot':
            boxplot = sns.boxplot(x=x_variable, hue='target', data=df_heart)
            st.pyplot(boxplot.figure)

        elif data_button == 'Relational Plot':
            relplot = sns.relplot(df_heart, x=x_variable, y=y_variable, hue="target", kind="line")
            st.pyplot(relplot.figure)

        elif data_button == 'Distribution Plot':
            distplot = sns.displot(df_heart, x=x_variable, hue="target", col="sex", kind="kde", rug=True)
            st.pyplot(distplot.figure)

    def my_function():
        df = df_heart.drop('target', axis=1)
        selected_variable = st.selectbox("Select the desired variable", df.columns)

        altair_plot = alt.Chart(df_heart).mark_circle(size=60, opacity=0.7).encode(
            x=selected_variable,
            y='target',
            color='sex',
            tooltip=[selected_variable, 'target', 'sex']
            ).properties(
                width=600,
                height=400
            ).interactive()
        
        st.write(altair_plot)

    if st.checkbox('Relation between "Target" and other variable'):
        my_function()

    
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
        
	# HiPlot Visualization
    if st.checkbox('Heart Disease Dataset Visualization with HiPlot'):
        selected_columns = st.multiselect("Select columns to visualize", df_heart.columns)
        selected_data = df_heart[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
        else:
            st.write("No data selected. Please choose at least one column to visualize.")

with tab4:
    st.write("Enter the Required fields to check whether you have a healthy heart")
    
    age = st.selectbox("Age", range(1, 121, 1))
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type', ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"))
    trestbps = st.selectbox('Resting Blood Pressure (in mm Hg)', range(1, 500, 1))
    chol = st.selectbox('Serum Cholestoral (in mg/dl)', range(1, 1000, 1))
    restecg = st.selectbox('Resting Electrocardiographic Results', ("Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
    fbs = st.radio("Fasting Blood Sugar (> 120 mg/dl)", ['Yes', 'No'])
    thalach = st.selectbox('Maximum Heart Rate Achieved', range(1, 300, 1))
    exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.number_input('Oldpeak (ST depression induced by exercise relative to rest)')
    slope = st.selectbox('Heart Rate Slope', ("Upsloping: better heart rate with exercise (uncommon)", "Flatsloping: minimal change (typical healthy heart)", "Downsloping: signs of an unhealthy heart"))
    ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', range(0, 5, 1))
    thal = st.selectbox('Thalium Stress Result', range(1, 8, 1))
    
    if st.button("Estimate"):
        x = df_heart.drop('target', axis=1)
        y = df_heart['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(x_train, y_train)
        y_pred = dt_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown("**Accuracy of the Model:** {:.2f}%".format(accuracy * 100))

        # Visualize the decision tree
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(dt_classifier, filled=True, feature_names=x.columns.tolist(), class_names=["No Heart Disease", "Heart Disease"])
        st.pyplot(fig)
