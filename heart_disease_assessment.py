import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import hiplot as hip

st.markdown("<h1 style='text-align: center; font-size: 35px;'>Heart Disease Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 15px;'>Presented by Sandhya Kilari</p>",
    unsafe_allow_html=True
)

# Load the Dataframe
url = "https://raw.githubusercontent.com/SandhyaKilari/Heart-Disease-Assessment/main/heart.csv"
df_heart = pd.read_csv(url)
df_heart = df_heart.drop(0)


# Information about the App
st.sidebar.subheader("About the Application")
st.sidebar.info("This web application will enable users to input their health attributes (e.g., age, sex, cholesterol levels, blood pressure, blood sugar level and more) and receive a risk assessment for heart disease. The app will provide a clear prediction that's easy to understand. It will also explain why each detail is important. This tool helps people understand their health and assists doctors when talking to patients.")
sidebar_placeholder = st.sidebar.empty()

tab1, tab2, tab3, tab4, tab5= st.tabs(["Introduction", "Statistical Analysis", "Data Visualization", 'Exploring Relationships', "Model Prediction"])

with tab1:
    st.markdown("<div style='text-align: justify'>Heart disease is one of the top reasons why people die worldwide, and finding it early is crucial for helping patients get better and reducing the cost of healthcare. By using data science and machine learning, we have a chance to create a tool that can save lives and make people more aware of their health. This project helps individuals, doctors, and society as a whole by giving them a useful way to understand and manage the risk of heart disease.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("<div style='text-align: justify'>It aims to develop a predictive model capable of assessing an individual's risk of developing heart disease by analyzing relevant health attributes. This undertaking holds considerable importance for healthcare professionals, policymakers, and individuals invested in heart disease prevention. It has the potential to enhance early intervention, leading to life-saving outcomes and reduced healthcare expenditures.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Dataset**")
    st.markdown("<div style='text-align: justify'>The Cleveland dataset which is widely used in heart disease research comprises 303 instances and 14 attributes, encompassing variables such as age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol level (chol), fasting blood sugar (fbs), maximum heart rate achieved (thalach), oldpeak, thal, and the target variable indicating presence of heart disease in the patient (0 = no disease, 1 = disease)</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Attributes Information**")
    st.markdown("1. Age")
    st.markdown("2. Sex")
    st.markdown('3. Chest Pain Type (cp: "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic")')
    st.markdown("4. Resting Blood Pressure (trestbps) in mm Hg")
    st.markdown("5. Serum Cholesterol (chol) in mg/dL")
    st.markdown("6. Fasting Blood Sugar (fbs) > 120 mg/dL")
    st.markdown('7. Resting Electrocardiographic Results (restecg: "Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy")')
    st.markdown("8. Maximum Heart Rate Achieved (thalach)")
    st.markdown("9. Exercise Induced Angina (exang)")
    st.markdown("10. oldpeak = ST depression induced by exercise relative to rest")
    st.markdown("11. The slope of the peak exercise ST segment")
    st.markdown("12. Number of Major Vessels Colored by Fluoroscopy (ca): 0-3")
    st.markdown("13. Thalium Stress Result (thal: 0 = normal; 1 = fixed defect; 2 = reversable defect)")
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
    if st.checkbox('Correlation Matrix'):
        st.write('Pairwise correlation of columns, excluding NA/null values')
        st.write(df_heart.corr())

with tab3:
    st.markdown('**Which critical risk factors substantially contribute to the occurrence of Heart Disease?**')

    if st.checkbox('Individual Variable Distribution'):
        st.markdown("This dataset contains many attributes that contributes to the presence or absence of heart disease")
        variable = st.selectbox('Please choose preferred variable', ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
        distplot = sns.displot(data=df_heart, x=variable)
        st.pyplot(distplot)
        if variable == 'age':
            st.markdown("*The age distribution appears to be slightly positively skewed, with a tail extending to the right. The mode of the distribution is around 50-60 years, indicating that this is the most common age in the dataset. However, there are several outliers on the higher age side, suggesting a few individuals with significantly older ages. The distribution is relatively wide, indicating a fair amount of variability in ages. There is also a secondary, smaller peak around 30-40 years, which could suggest the presence of a younger subgroup in the dataset*")
        if variable == 'sex':
            st.markdown("*The distribution of the 'sex' variable shows that there are two categories: 'female' and 'male.' In this dataset, 'male' is the dominant category, representing a larger proportion of the individuals. This indicates an imbalance in the dataset, with a higher number of males compared to females. The specific proportions of each category would be helpful to understand the exact magnitude of this imbalance*")
        if variable == 'cp':
            st.markdown("*The distribution of the 'cp' variable shows that it comprises several categories corresponding to different types of chest pain. Type 'X' is the most common type of chest pain, representing the largest proportion of individuals in the dataset. This suggests that type 'X' is the dominant category. The other types of chest pain, 'Y,' 'Z,' and 'W,' have smaller proportions, indicating less common occurrences. This distribution provides insights into the prevalence of various chest pain types within the dataset*")        
        if variable == 'trestbps':
            st.markdown("*The distribution of 'trestbps' (resting blood pressure) appears to be slightly positively skewed, with a tail extending to the right. The mode of the distribution is around 120 mm Hg, indicating that this is the most common resting blood pressure level in the dataset. However, there are several outliers on the higher blood pressure side, suggesting a few individuals with significantly elevated resting blood pressure. The distribution is relatively wide, which implies a significant variability in resting blood pressure levels within the dataset. This information is crucial for understanding the overall range and characteristics of blood pressure in the context of heart health*")
        if variable == 'chol':
            st.markdown("*The distribution of the 'chol' variable appears to be slightly right-skewed, with a tail extending to the higher cholesterol levels. The central tendency of cholesterol levels in this dataset is around the mean value, indicating that this is the typical cholesterol level. There are a few outliers on the higher cholesterol side, suggesting the presence of individuals with significantly elevated cholesterol levels. The distribution is moderately wide, indicating some variability in cholesterol levels. However, it does not appear to have distinct modes or multiple peaks*")
        if variable == 'fbs':
            st.markdown("*The distribution of the 'fbs' variable reveals two categories: 'normal blood sugar' and 'elevated blood sugar.' In this dataset, it appears that 'normal blood sugar' is the dominant category, representing a larger proportion of individuals. This suggests that there are more individuals with normal blood sugar levels in the dataset*")
        if variable == 'restecg':
            st.markdown("*The distribution of the 'restecg' variable reveals that it consists of multiple categories, including 'normal,' 'ST-T wave abnormality,' and 'probable or definite left ventricular hypertrophy.' The most common category appears to be 'ST-T wave abnormality,' indicating that this particular electrocardiographic finding is the predominant result in the dataset. It's important to consult the clinical context to understand the significance of this electrocardiographic result, as 'ST-T wave abnormality' may have implications for heart health*")
        if variable == 'talach':
            st.markdown("*The distribution of the 'thalach' variable appears to be slightly positively skewed, with a tail extending to the right. The mode of the distribution is around a maximum heart rate of approximately 160 beats per minute, indicating that this is the most common maximum heart rate achieved during exercise in the dataset. The distribution is moderately wide, suggesting some variability in maximum heart rates. There are a few outliers on the higher end, indicating individuals who achieved notably higher maximum heart rates during exercise*")
        if variable == 'exang':
            st.markdown("*The distribution of the 'exang' variable indicates two categories: 'no exercise-induced angina' and 'exercise-induced angina.' Without detailed proportions, we can't assess the exact balance or imbalance, but this variable's distribution could be important in the context of a heart disease dataset. If 'exercise-induced angina' is prevalent, it might suggest a significant occurrence of angina during exercise among the individuals in the dataset*")
        if variable == 'oldpeak':
            st.markdown("*The distribution of the 'oldpeak' variable appears to be positively skewed, with a tail extending to the right. The peak of the distribution is around 1.5, indicating that this is the most common 'oldpeak' value in the dataset. The distribution is relatively wide, suggesting a notable variability in 'oldpeak' values. There are a few outliers on the higher 'oldpeak' side, which might represent individuals with exceptionally high stress on the heart during exercise*")
        if variable == 'slope':
            st.markdown("*The distribution of the 'slope' variable indicates that it has three distinct categories: 'upsloping,' 'horizontal,' and 'downsloping.' In this dataset, 'horizontal' appears to be the most common slope pattern, followed by 'downsloping' and 'upsloping.' These findings suggest that 'horizontal' is the predominant ST-segment slope pattern within the dataset*")
        if variable == 'ca':
            st.markdown("*The distribution of the 'ca' variable shows that it can take several values. Most notably, there is a significant presence of '0' values, indicating a substantial proportion of individuals with no major vessels colored by fluoroscopy. The distribution is right-skewed, suggesting that the majority of individuals in the dataset have fewer major vessels, but there are a few outliers with a higher number of major vessels*")
        if variable == 'thal':
            st.markdown("*The distribution of the 'thal' variable shows that the 'reversible defect' category is the most common, indicating that a significant proportion of the individuals in the dataset have this type of thallium stress test result. 'Normal' and 'fixed defect' are less prevalent. This distribution may have clinical implications, as the 'reversible defect' category might be associated with specific cardiac conditions that are more common in this dataset. Further analysis could explore the relationship between 'thal' types and health outcomes*")
        if variable == 'target':
            st.markdown("*The distribution of the 'target' variable indicates a binary outcome with two categories: '0' and '1.' In this dataset, it appears that '0' is the dominant category, suggesting a higher proportion of individuals with the absence of the target condition, while '1' represents the presence of the condition. This indicates an imbalance in the dataset, with a larger number of individuals not having the target condition*")

    if st.checkbox('Relation between "Target" and other variable'):
        df = df_heart.drop('target', axis=1)
        selected_variable = st.selectbox("Select the desired variable", df.columns)
        data_button = st.selectbox('Please choose preferred visualization', ['Scatter Plot', 'Histogram Plot', 'Distribution Plot'])

        if data_button == 'Scatter Plot':
            scatter_plot = sns.scatterplot(data=df_heart, x=selected_variable, y='target', hue='sex')
            st.pyplot(scatter_plot.figure)

        elif data_button == 'Histogram Plot':
            histplot = sns.histplot(data=df_heart, x=selected_variable, y='target', binwidth=5, hue='sex')
            st.pyplot(histplot.figure)

        elif data_button == 'Distribution Plot':
            distplot = sns.displot(data=df_heart, x=selected_variable, y='target', hue='sex')
            st.pyplot(distplot)

        if selected_variable == 'age':
            st.write('*In this plot, you can see a trend that suggests that as age increases, there appears to be a higher concentration of data points with a "1" (indicating the presence of a heart disease)*')
            st.write('*This suggests a positive correlation between age and the likelihood of having a heart disease. In other words, as individuals get older, they are more likely to have a heart disease, as indicated by the "target" variable*')
            st.write('*This observation aligns with the common understanding that age is a significant risk factor for heart disease*')

        if selected_variable == 'sex':
            st.write("*From the plot, there is a visible pattern where one gender has a higher concentration of '1' (indicating the presence of heart disease) while the other has a higher concentration of '0' (indicating no heart disease), it suggests that there may be a relationship between gender ('sex') and the likelihood of heart disease*")
        
        if selected_variable == 'cp':
            st.write("*We can see from the plot, certain values of 'cp' are associated with a higher concentration of '1' (indicating the presence of heart disease) and other values of 'cp' are associated with a higher concentration of '0' (indicating no heart disease), it suggests that the 'cp' variable is related to the likelihood of heart disease*")

        if selected_variable == 'trestbps':
            st.write("*Most data points are concentrated at lower resting blood pressure values for '0' (no heart disease), it suggest a negative correlation, indicating that lower resting blood pressure is associated with a lower likelihood of heart disease*")

        if selected_variable == 'chol':
            st.write("*Most data points are concentrated at lower cholesterol levels for '0' (no heart disease), it might suggest a negative correlation, indicating that lower cholesterol levels are associated with a lower likelihood of heart disease*")        
        
        if selected_variable == 'fbs':
            st.write('*Here data points are divided into two groups based on fasting blood sugar levels (e.g., high and low)*')
            st.write("*Pattern suggest that one group has a higher concentration of '1' (indicating the presence of heart disease) while the other group has a higher concentration of '0' (indicating no heart disease) which implies that high fasting blood sugar levels may be associated with a higher likelihood of heart disease*")
    
        if selected_variable == 'restecg':
            st.write('One cluster of data points (representing specific "restecg" values) is predominantly associated with a "target" value of "1" (indicating heart disease presence), while another cluster is primarily associated with a "target" value of "0" (indicating no heart disease), it suggests that "restecg" may be related to the likelihood of heart disease.')
        
        if selected_variable == 'thalach':
            st.write('*If you see that as heart rate increases, there is a higher concentration of "1" (indicating the presence of heart disease), it suggests a positive correlation. In other words, a higher heart rate might be associated with a higher likelihood of heart disease.*')
            st.write('*Conversely, if you observe that a lower heart rate is associated with a higher concentration of "1", it suggests a negative correlation. In this case, lower heart rate might be related to a higher likelihood of heart disease.*')

        if selected_variable == 'exang':
            st.write('*Most data points fall into two distinct clusters or patterns, it suggests a possible relationship between exercise-induced angina and the presence of heart disease*')
        if selected_variable == 'oldpeak':
            st.write('*Data points are concentrated at lower "oldpeak" values for a "target" value of 0, it suggests that lower "oldpeak" values are associated with a lower likelihood of heart disease.*')
        
        if selected_variable == 'slope':
            st.write('*Points appear scattered with no clear trend, it suggests that there may not be a strong relationship between the "slope" variable and the "target" variable*')
        
        if selected_variable == 'ca':
            st.write('*Individuals with a higher number of major vessels (higher "ca" values) tend to have a "0" for the "target" variable (indicating no heart disease), it may suggest a negative correlation. In this case, a higher number of major vessels could indicate a lower likelihood of heart disease*')
        
        if selected_variable == 'thal':
            st.write("*It suggests that individuals with that specific thal result are more likely to have heart disease and conversely*")
	
    # HiPlot Visualization
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
        
    if st.checkbox('Heart Disease Dataset Visualization with HiPlot'):
        st.write('*This plot allows user to select required columns and visualize them using HiPlot. By systematically exploring the dataset, we can uncover relationships into how attributes may be correlated with the presence or absence of heart disease within specific age groups and clinical attribute ranges.*')
        selected_columns = st.multiselect("Select columns to visualize", df_heart.columns)
        selected_data = df_heart[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
        else:
            st.write("No data selected. Please choose at least one column to visualize.")
    
    if st.checkbox("Visualization Techniques"):
        st.subheader('Correlation Heatmap')
        correlation_matrix = df_heart.corr()
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(heatmap.figure)
        st.markdown("*This heatmap will provide a visual representation of the correlations between all pairs of numerical variables in the dataset, helping you quickly identify which variables are strongly correlated with each other*")

        st.subheader("Pairplot")
        sns.set(style="ticks")
        pairplot = sns.pairplot(df_heart, hue="target", diag_kind="kde", markers=["o", "s"])
        st.pyplot(pairplot.figure)
        plt.clf()  # Clear the figure
        st.markdown("*The pair plot will show scatter plots for all pairs of numerical variables in the dataset, with color differentiation for the 'target' variable. This visualization can help you quickly identify patterns and relationships between different features, especially in the context of heart disease diagnosis*")

with tab4:
    # Correlation based analysis
    correlation = df_heart['cp'].corr(df_heart['target'])
    st.subheader(f'Chest Pain Types vs. Heart Disease\nCorrelation: {correlation:.2f}')
    plt.figure(figsize=(8, 6))  # Set a larger figure size
    plot1 = sns.boxplot(data=df_heart, x='cp', y='target', color='blue')
    st.pyplot(plot1.figure)
    plt.clf()  # Clear the figure
    st.write("*Each box in the plot represents a different chest pain type (probably categorized into types like 0, 1, 2, or 3)*")
    st.write("*The box plot helps us to understand how chest pain types are related to the presence of heart disease. For example, we can observe whether a particular chest pain type is more common in individuals with or without heart disease based on the median and the distribution of data points.*")
    
    # Feature Relationship
    correlation = df_heart['age'].corr(df_heart['trestbps'])
    st.subheader(f'Age vs. Blood Pressure\nCorrelation: {correlation:.2f}')
    plt.figure(figsize=(10, 10))  # Set a larger figure size
    plot2 = sns.scatterplot(data=df_heart, x='age', y='trestbps', color='green', label= "age vs trestbps")
    st.pyplot(plot2.figure)
    plt.clf()  # Clear the figure
    st.write("*Age and blood pressure are positively correlated, meaning that as people get older, their blood pressure tends to increase, which can be a risk factor for heart disease*")

    correlation = df_heart['thalach'].corr(df_heart['target'])
    st.subheader(f'Heart Rate vs. Heart Disease\nCorrelation: {correlation:.2f}')
    plt.figure(figsize=(8, 6))  # Set a larger figure size
    plot3 = sns.scatterplot(data=df_heart, x='thalach', y='target', color='red', alpha=0.5, label= "thalach vs target")
    st.pyplot(plot3.figure)
    plt.clf()  # Clear the figure
    st.write("*The scatter plot show that individuals with higher heart rate tend to be more likelihood of heart disease, as the points cluster in the direction of increasing rate and heart disease presence*")

with tab5:
    st.markdown("Can we establish a dependable predictive tool for early detection?")
    st.write(" ")
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
        st.write("Need to create a model based on the attributes to predict the risk of getting a heart disease!!")
