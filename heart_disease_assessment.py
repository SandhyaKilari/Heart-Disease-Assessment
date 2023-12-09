# Importing Necessary Libraries
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import hiplot as hip
import numpy as np
import plotly.express as px

import pickle as pkl

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the Saved Models
with open("rf_model.pkl", "rb") as file:
    rf_model = pkl.load(file)
with open("gnb_model.pkl", "rb") as file:
    gnb_model = pkl.load(file)
with open("lr_model.pkl", "rb") as file:
    lr_model = pkl.load(file)
with open("svm_model.pkl", "rb") as file:
    svm_model = pkl.load(file)
with open("dt_model.pkl", "rb") as file:
    dt_model = pkl.load(file)
with open("qda_model.pkl", "rb") as file:
    qda_model = pkl.load(file)
with open("knn_model.pkl", "rb") as file:
    knn_model = pkl.load(file)


# Project    
st.markdown("<h1 style='text-align: center; font-size: 35px;'> Heart Disease Assessment </h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 15px;'>Presented by Sandhya Kilari</p>",
    unsafe_allow_html=True
)

# Dataset
url = "https://raw.githubusercontent.com/SandhyaKilari/Heart-Disease-Assessment/main/heart.csv"
df_heart = pd.read_csv(url)
df_model = df_heart

# Target Variable
y = df_heart["target"]

# Data Analysis and Cleaning
df_heart = df_heart.drop(0)
df_new = df_heart
df_new = df_new.drop(['slope', 'ca', 'thal'], axis=1)

# Label Encoding
data_vis = df_new
data_vis.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'target']

# Information about the App
st.sidebar.write(
    "Welcome to a journey of self-discovery through your heart's story! 🌟\n\n"
    "Have you ever wondered what your heart health says about you?"
)
image_url = "heart.jpeg" 
st.sidebar.image(image_url, use_column_width=True)
st.sidebar.info("This web application unravels the tale hidden within your health attributes (e.g., age, sex, cholesterol levels, blood pressure, blood sugar level and more), offering insights into. The app will provide a clear prediction that's easy to understand. It will also explain why each detail is important. This tool helps people understand their health and assists doctors when talking to patients.")
st.sidebar.text("") 
st.sidebar.write("Start uncovering your heart's tale by interacting with various features of the app📈")

#sidebar_placeholder = st.sidebar.empty()

# Pre-processing of the user-input
def preprocess(sex, cp, exang,fbs,restecg):   
 
    if sex=="male":
        sex=1 
    else: 
        sex=0
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
        
    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2

    return sex, cp, exang,fbs,restecg

# Sections of the Web-Application
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Introduction", "Statistical Analysis", "Data Visualization", 'Exploring Relationships', "Classifiers Comparision", "Model Prediction", "Bio"])

# Introduction
with tab1:
    st.markdown("<div style='text-align: justify'>Heart disease is one of the top reasons why people die worldwide, and finding it early is crucial for helping patients get better and reducing the cost of healthcare. By using data science and machine learning, we have a chance to create a tool that can save lives and make people more aware of their health. This project helps individuals, doctors, and society as a whole by giving them a useful way to understand and manage the risk of heart disease.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("<div style='text-align: justify'>It aims to develop a predictive model capable of assessing an individual's risk of developing heart disease by analyzing relevant health attributes. This undertaking holds considerable importance for healthcare professionals, policymakers, and individuals invested in heart disease prevention. It has the potential to enhance early intervention, leading to life-saving outcomes and reduced healthcare expenditures.</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("**Dataset**")
    st.markdown("<div style='text-align: justify'>The Cleveland dataset which is widely used in heart disease research comprises 303 instances and 14 attributes, encompassing variables such as age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol level (chol), fasting blood sugar (fbs), maximum heart rate achieved (thalach), oldpeak, thal, and the target variable indicating presence of heart disease in the patient (0 = no disease, 1 = disease)</div>", unsafe_allow_html=True)
    st.markdown(" ")
    if st.button("Understand the Attributes/Features present in the dataset"):
        # Display the information when the button is clicked
        st.markdown("1. Age: Represents the age of persons in years")
        st.markdown("2. Sex: (1 = male, 0 = female)")
        st.markdown('3. Chest Pain Type (cp): [ 0: asymptomatic, 1: atypical angina, 2: non-anginal pain, 3: typical angina]')
        st.markdown("4. Resting Blood Pressure (trestbps) in mm Hg")
        st.markdown("5. Serum Cholesterol (chol) in mg/dL")
        st.markdown("6. Fasting Blood Sugar (fbs) > 120 mg/dL: [0 = no, 1 = yes]")
        st.markdown('7. Resting Electrocardiographic Results (restecg): [0: Nothing to note, 1: ST-T Wave abnormality, 2: Possible or definite left ventricular hypertrophy]')
        st.markdown("8. Maximum Heart Rate Achieved (thalach)")
        st.markdown("9. Exercise Induced Angina (exang): (1 = yes, 0 = no)")
        st.markdown("10. ST depression induced by exercise relative to rest (oldpeak)")
        st.markdown("11. Slope of the peak exercise ST segment (slope): [0: downsloping; 1: flat; 2: upsloping]")
        st.markdown("12. Number of Major Vessels Colored by Fluoroscopy (ca): 0-3")
        st.markdown("13. Thalium Stress Result (thal): (0 = normal; 1 = fixed defect; 2 = reversable defect)")
    st.markdown("**Reference Link:**")
    st.markdown("https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")

# Statistical Analysis
with tab2:
    df=df_heart
    if st.checkbox('Raw Data'):
        st.write(df)
    if st.checkbox('Descriptive Statistics'):
        st.write(df.describe())
        st.write('Information about the data:')
        st.write(f'Total Number of Samples: {df.shape[0]}')
        st.write(f'Number of Features: {df.shape[1]}')
    if st.checkbox('Correlation Heatmap'):
        st.write('Pairwise correlation of columns, excluding NA/null values')
        st.write(df.corr())
    if st.checkbox('Missing Values'):
        missing_values = df.isnull().sum()
        st.write(missing_values)

# Data Visualization
with tab3:
    categorical_columns = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina']
    data_vis[categorical_columns] = data_vis[categorical_columns].astype('category')
    st.markdown('**Which critical risk factors substantially contribute to the occurrence of Heart Disease?**')

    if st.checkbox('Distribution of Categorical Variables'):
        st.markdown("Explore distribution of categorical variables that contributes to the presence or absence of heart disease")
        variable = st.selectbox('Please choose preferred variable', categorical_columns)
        countplot = sns.countplot(x=variable,hue='target', data=data_vis)
        st.pyplot(countplot.figure)
        plt.clf()
        if variable == 'sex':
            st.markdown("*The distribution of the 'sex' variable shows that there are two categories: 'female' and 'male.' In this dataset, 'male' is the dominant category, representing a larger proportion of the individuals. This indicates an imbalance in the dataset, with a higher number of males compared to females. The specific proportions of each category would be helpful to understand the exact magnitude of this imbalance*")
        if variable == 'chest_pain_type':
            st.markdown("*The distribution of the 'cp' variable shows that it comprises several categories corresponding to different types of chest pain. Type 'X' is the most common type of chest pain, representing the largest proportion of individuals in the dataset. This suggests that type 'X' is the dominant category. The other types of chest pain, 'Y,' 'Z,' and 'W,' have smaller proportions, indicating less common occurrences. This distribution provides insights into the prevalence of various chest pain types within the dataset*")        
        if variable == 'fasting_blood_sugar':
            st.markdown("*The distribution of the 'fbs' variable reveals two categories: 'normal blood sugar' and 'elevated blood sugar.' In this dataset, it appears that 'normal blood sugar' is the dominant category, representing a larger proportion of individuals. This suggests that there are more individuals with normal blood sugar levels in the dataset*")
        if variable == 'rest_ecg':
            st.markdown("*The distribution of the 'restecg' variable reveals that it consists of multiple categories, including 'normal,' 'ST-T wave abnormality,' and 'probable or definite left ventricular hypertrophy.' The most common category appears to be 'ST-T wave abnormality,' indicating that this particular electrocardiographic finding is the predominant result in the dataset. It's important to consult the clinical context to understand the significance of this electrocardiographic result, as 'ST-T wave abnormality' may have implications for heart health*")
        if variable == 'exercise_induced_angina':
            st.markdown("*The distribution of the 'exang' variable indicates two categories: 'no exercise-induced angina' and 'exercise-induced angina.' Without detailed proportions, we can't assess the exact balance or imbalance, but this variable's distribution could be important in the context of a heart disease dataset. If 'exercise-induced angina' is prevalent, it might suggest a significant occurrence of angina during exercise among the individuals in the dataset*")

    if st.checkbox('Relation between "Target" variable and the features'):
        df = df_new.drop('target', axis=1)
        selected_variable = st.selectbox("Select the desired variable", df.columns)
        data_button = st.selectbox('Please choose preferred visualization', ['Scatter Plot', 'Box Plot', 'Distribution Plot'])

        if data_button == 'Scatter Plot':
            scatter_plot = sns.scatterplot(data=data_vis, x=selected_variable, y='target', hue='sex')
            st.pyplot(scatter_plot.figure)
            
        elif data_button == 'Box Plot':
            box_plot = sns.boxplot(x='target', y=selected_variable, data=data_vis,palette='rainbow')
            st.pyplot(box_plot.figure)

        #elif data_button == 'Histogram Plot':
            #histplot = sns.histplot(data=data_vis, x=selected_variable, y='target', binwidth=5, hue='sex')
            #st.pyplot(histplot.figure)

        elif data_button == 'Distribution Plot':
            distplot = sns.displot(data=data_vis, x=selected_variable, y='target', hue='sex')
            st.pyplot(distplot)

        if selected_variable == 'age':
            st.write('*In this plot, you can see a trend that suggests that as age increases, there appears to be a higher concentration of data points with a "1" (indicating the presence of a heart disease)*')
            st.write('*This suggests a positive correlation between age and the likelihood of having a heart disease. In other words, as individuals get older, they are more likely to have a heart disease, as indicated by the "target" variable*')
            st.write('*This observation aligns with the common understanding that age is a significant risk factor for heart disease*')

        if selected_variable == 'sex':
            st.write("*From the plot, there is a visible pattern where one gender has a higher concentration of '1' (indicating the presence of heart disease) while the other has a higher concentration of '0' (indicating no heart disease), it suggests that there may be a relationship between gender ('sex') and the likelihood of heart disease*")
        
        if selected_variable == 'chest_pain_type':
            st.write("*We can see from the plot, certain values of 'cp' are associated with a higher concentration of '1' (indicating the presence of heart disease) and other values of 'cp' are associated with a higher concentration of '0' (indicating no heart disease), it suggests that the 'cp' variable is related to the likelihood of heart disease*")
            
        if selected_variable == 'resting_blood_pressure':
            st.write("*Most data points are concentrated at lower resting blood pressure values for '0' (no heart disease), it suggest a negative correlation, indicating that lower resting blood pressure is associated with a lower likelihood of heart disease*")

        if selected_variable == 'cholesterol':
            st.write("*Most data points are concentrated at lower cholesterol levels for '0' (no heart disease), it might suggest a negative correlation, indicating that lower cholesterol levels are associated with a lower likelihood of heart disease*")        
        
        if selected_variable == 'fasting_blood_sugar':
            st.write('*Here data points are divided into two groups based on fasting blood sugar levels (e.g., high and low)*')
            st.write("*Pattern suggest that one group has a higher concentration of '1' (indicating the presence of heart disease) while the other group has a higher concentration of '0' (indicating no heart disease) which implies that high fasting blood sugar levels may be associated with a higher likelihood of heart disease*")
    
        if selected_variable == 'rest_ecg':
            st.write('One cluster of data points (representing specific "restecg" values) is predominantly associated with a "target" value of "1" (indicating heart disease presence), while another cluster is primarily associated with a "target" value of "0" (indicating no heart disease), it suggests that "restecg" may be related to the likelihood of heart disease.')
        
        if selected_variable == 'max_heart_rate_achieved':
            st.write('*If you see that as heart rate increases, there is a higher concentration of "1" (indicating the presence of heart disease), it suggests a positive correlation. In other words, a higher heart rate might be associated with a higher likelihood of heart disease.*')
            st.write('*Conversely, if you observe that a lower heart rate is associated with a higher concentration of "1", it suggests a negative correlation. In this case, lower heart rate might be related to a higher likelihood of heart disease.*')

        if selected_variable == 'exercise_induced_angina':
            st.write('*Most data points fall into two distinct clusters or patterns, it suggests a possible relationship between exercise-induced angina and the presence of heart disease*')
        
        if selected_variable == 'st_depression(oldpeak)':
            st.write('*Data points are concentrated at lower "oldpeak" values for a "target" value of 0, it suggests that lower "oldpeak" values are associated with a lower likelihood of heart disease.*')
            	
    # HiPlot Visualization
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
        
    if st.checkbox('Interactive HiPlot Visualization'):
        st.write('*This plot allows user to select required columns and visualize them using HiPlot. By systematically exploring the dataset, we can uncover relationships into how attributes may be correlated with the presence or absence of heart disease within specific age groups and clinical attribute ranges.*')
        selected_columns = st.multiselect("Select columns to visualize", data_vis.columns)
        selected_data = data_vis[selected_columns]
        if not selected_data.empty:
            experiment = hip.Experiment.from_dataframe(selected_data)
            hiplot_html_file = save_hiplot_to_html(experiment)
            st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
        else:
            st.write("No data selected. Please choose at least one column to visualize.")
    
    if st.checkbox("Visualization Techniques"):
        st.subheader('Correlation Heatmap')
        correlation_matrix = data_vis.corr()
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(heatmap.figure)
        st.markdown("*This heatmap will provide a visual representation of the correlations between all pairs of numerical variables in the dataset, helping you quickly identify which variables are strongly correlated with each other*")

        st.subheader("Pairplot")
        sns.set(style="ticks")
        data_cont = data_vis.drop(columns=categorical_columns)
        # Pair-Plot of Continuous Variables
        pairplot = sns.pairplot(data_cont, hue="target", diag_kind="kde", markers=["o", "s"])
        st.pyplot(pairplot.figure)
        plt.clf()  # Clear the figure
        st.markdown("*The pair plot will show scatter plots for all pairs of numerical variables in the dataset, with color differentiation for the 'target' variable. This visualization can help you quickly identify patterns and relationships between different features, especially in the context of heart disease diagnosis*")

# Exploring Relationships
with tab4:
    
    df_heart = data_vis

    # Age vs. Maximum Heart Rate Achieved
    scatter = alt.Chart(df_heart).mark_circle().encode(
        x='age:Q',
        y='max_heart_rate_achieved:Q',
        color='target:N'
        ).properties(
            width=750,
            height=400,
            title='Age vs. Maximum Heart Rate Achieved'
        ).interactive()
    st.altair_chart(scatter)
    plt.clf()
    st.write("*People who have detected maximum heart rate have a high risk of getting heart disease and fall under the age group between 40 and 60*")

    # Chest-Pain Type vs. Target
    fig = px.histogram(df_heart, x='chest_pain_type', color='target', barmode='group', title='Chest Pain Types and Heart Disease')
    st.plotly_chart(fig)
    plt.clf()
    st.write("*Each box in the plot represents a different chest pain type (probably categorized into types like 0, 1, 2, or 3)*")
    st.write("*The box plot helps us to understand how chest pain types are related to the presence of heart disease. For example, we can observe whether a particular chest pain type is more common in individuals with or without heart disease based on the median and the distribution of data points.*")
    
    # Age vs. Resting Blood Pressure
    scatter = alt.Chart(df_heart).mark_circle().encode(
        x='age:Q',
        y='resting_blood_pressure:Q',
        color='target:N'
        ).properties(
            width=750,
            height=400,
            title='Age vs. Resting Blood Pressure'
        ).interactive()
    st.altair_chart(scatter)
    plt.clf()
    st.write("*Age and blood pressure are positively correlated, meaning that as people get older, their blood pressure tends to increase, which can be a risk factor for heart disease*")

    # Heart Rate vs. Heart Disease
    strip_plot = px.strip(
        df_heart, x='target', y='max_heart_rate_achieved',
        color='target', width=750, height=400
        )
    
    strip_plot.update_layout(
        title="Heart Rate vs. Heart Disease",
        xaxis_title="0= Heart Disease, 1= No disease",
        yaxis_title="Max Heart Rate Achieved"
        )
    st.plotly_chart(strip_plot)
    plt.clf()
    st.write("*The scatter plot show that individuals with higher heart rate tend to be more likelihood of heart disease, as the points cluster in the direction of increasing rate and heart disease presence*")

# Model Performance Metrices
with tab5:
    data = df_model.drop(['slope', 'ca', 'thal'], axis=1)
    X = data.drop('target', axis=1) 
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    my_scaler = StandardScaler()
    my_scaler.fit(X_train)  
    
    X_train = my_scaler.transform(X_train)
    X_test = my_scaler.transform(X_test)
        
    st.write("Model Performance Metrices")
        
    algorithms = [
        "Support Vector Machine",
        "Logistic Regression",
        "Naive Bayes",
        "K-Nearest Neighbors",
        "Quadratic Discriminant Analysis",
        "Decision Tree",
        "Random Forest"
    ]
    
    model = st.selectbox("Select a Model", algorithms)
    if model == "Support Vector Machine":
        C = st.slider("C (Regularization parameter)", 0.1, 10.0, step=0.1, value=1.0)
        gamma = st.slider("Gamma", 0.1, 10.0, step=0.1, value=1.0)
        
        svm_model = svm.SVC(C=C, gamma=gamma)
        svm_model.fit(X_train, y_train)
        
        y_pred = svm_model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        st.write("Confusion matrix:\n", conf_mat)
        st.write("Accuracy:", svm_model.score(X_test, y_test))
        st.write("Precision:", metrics.precision_score(y_test, y_pred))
        st.write("F1:", metrics.f1_score(y_test, y_pred))
        with st.expander("Understand what Support Vector Machine is and how it works"):
            st.write("The Support Vector Machine (SVM) is a type of supervised machine learning algorithm used for classification and regression tasks. It targets to finds the best possible way to divide data points into different categories by drawing a line, plane, or hyperplane in a higher-dimensional space, maximizing the margin between the different categories while allowing for some errors.  It's widely used in various fields like image classification, text classification, biological classification problems, and more.")
        
    if model == "Logistic Regression":
        C = st.slider("C (Regularization parameter)", 0.1, 10.0, step=0.1, value=1.0)
        random_state = st.number_input("Random State", min_value=0, max_value=1000, value=0, step=1, format="%d")
        lr_model = LogisticRegression(C=C, random_state=random_state)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:\n", cnf_matrix)
        st.write("Accuracy:", lr_model.score(X_test, y_test))
        st.write("Precision:",metrics.precision_score(y_test, y_pred))
        st.write("F1:",metrics.f1_score(y_test, y_pred)) 
        with st.expander("Understand what Logistic Regression is and how it works"):
            st.write("Logistic Regression is a statistical technique used for binary classification. It estimates the probability of an observation belonging to one of two classes. It uses a sigmoid function to transform input features into probabilities and draws a decision boundary to separate classes. The model learns from labeled data, assigns importance to features, and is evaluated based on its predictive performance using metrics like accuracy, precision, recall, and F1-score.")                                                    
            
    if model == "Naive Bayes":
        col = X.columns
        selected_features = st.multiselect("Select Features", list(col))
        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            X_train_selected = pd.DataFrame(X_train, columns=col)[selected_features]
            X_test_selected = pd.DataFrame(X_test, columns=col)[selected_features]
            
            gnb_model = GaussianNB()
            gnb_model.fit(X_train_selected, y_train)
            y_pred = gnb_model.predict(X_test_selected)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            st.write("Confusion matrix:\n", cnf_matrix)
            st.write("Accuracy:", gnb_model.score(X_test_selected, y_test))
            st.write("Precision:", metrics.precision_score(y_test, y_pred))
            st.write("F1:", metrics.f1_score(y_test, y_pred))
        with st.expander("Understand what Gaussian Naive Bayes is and how it works"):
            st.write("Gaussian Naive Bayes is a classification algorithm based on Bayes' theorem. It assumes features are normally distributed and independent. It calculates the probability that a data point belongs to a particular class using Gaussian (normal) distributions for numeric features. This method is commonly used in text classification, medical diagnosis, and similar tasks where feature independence and Gaussian distribution hold. Despite its simplicity, it's often effective and computationally efficient, especially with smaller datasets.")       
                              
    if model == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", 1, 20, value=5)
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)        
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:\n", cnf_matrix)
        st.write("Accuracy:", knn_model.score(X_test, y_test)) 
        st.write("Precision:",metrics.precision_score(y_test, y_pred))
        st.write("F1:",metrics.f1_score(y_test, y_pred)) 
        with st.expander("Understand what K-Nearest Neighbors is and how it works"):
            st.write("K-Nearest Neighbors (KNN) is a machine learning algorithm used for classification and regression. It predicts the class or value of a new data point by considering the majority (for classification) or averaging (for regression) the 'K' nearest data points in the training set based on a chosen distance measure, typically Euclidean distance. Its simplicity makes it easy to understand, but it can be computationally intensive for large datasets during the prediction phase. The choice of 'K' influences the model's performance.")
                             
    if model == "Quadratic Discriminant Analysis":
        reg_param = st.slider("Regularization Parameter (reg_param)", 0.0, 1.0, value=0.0)
        qda_model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        qda_model.fit(X_train, y_train)
        y_pred = qda_model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:\n", cnf_matrix)
        st.write("Accuracy:", qda_model.score(X_test, y_test))
        st.write("Precision:",metrics.precision_score(y_test, y_pred))
        st.write("F1:",metrics.f1_score(y_test, y_pred)) 
        with st.expander("Understand what Quadratic Discriminant Analysis is and how it works"):
            st.write("Quadratic Discriminant Analysis (QDA) is a classification technique used for predicting categories based on input features. It calculates distinct mean vectors and covariance matrices for each class from the training data. These parameters are used to create a model that predicts the class to which new observations belong. QDA's flexibility in allowing separate covariance matrices for each class enables it to handle non-linear relationships between features and classes, making it useful for scenarios with complex or non-linear data patterns.")
                              
    if model == "Decision Tree":
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=0, step=1, format="%d")
        max_depth = st.slider("Maximum Depth", 1, 20, value=5)
        dt_model = DecisionTreeClassifier(criterion=criterion, random_state=random_state, max_depth=max_depth)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:\n", cnf_matrix)
        st.write("Accuracy:", dt_model.score(X_test, y_test))
        st.write("Precision:",metrics.precision_score(y_test, y_pred))
        st.write("F1:",metrics.f1_score(y_test, y_pred)) 
        with st.expander("Understand what Decision Tree is and how it works"):      
            st.write("Decision tree modeling is a machine learning technique that creates a tree-like structure to make decisions based on input data. It selects features to split the data into subsets, aiming to make the subsets as homogeneous as possible regarding the target variable. This process continues recursively until a stopping criterion is met. When new data is given, the model traverses the tree to predict the outcome based on the input features. Advantages include interpretability and the ability to capture non-linear relationships.")
            st.write(" ")
            st.write("However, decision trees can also suffer from certain limitations like overfitting (creating overly complex trees that perform well on training data but poorly on unseen data), instability with small variations in data, and sometimes not achieving the highest predictive accuracy compared to other algorithms.")                                    
                        
    if model == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 1, 100, value=2, step=1)
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=0, step=1, format="%d")
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:\n", cnf_matrix)
        st.write("Accuracy:", rf_model.score(X_test, y_test))
        st.write("Precision:",metrics.precision_score(y_test, y_pred))
        st.write("F1:",metrics.f1_score(y_test, y_pred)) 
        with st.expander("Understand what Random Forest is and how it works"):
            st.write("Random Forest is a powerful machine learning technique that builds numerous decision trees using random subsets of data and features. These individual trees work together by voting (for classification problems) or averaging (for regression tasks) to produce predictions. It's highly effective, particularly with large datasets, as it mitigates overfitting issues commonly found in single decision trees. Due to its ability to create diverse models and combine their outputs, Random Forest is known for its accuracy and robustness across different applications in machine learning.")                          
        
    with st.expander("Summarization and visualization of accuracy scores for different machine learning algorithms"):
        accuracy = []
        classifiers = ['Support Vector Machine','KNN', 'Decision Tree', 'Logistic Regression', 'Naive Bayes', 'Quadratic Discriminant Analysis', 'Random Forest']
        models = [svm.SVC(gamma=0.001),neighbors.KNeighborsClassifier(n_neighbors=2), DecisionTreeClassifier(criterion='gini', random_state=0), LogisticRegression(), GaussianNB(), QuadraticDiscriminantAnalysis(), RandomForestClassifier(n_estimators=8, random_state=0)]
        summary = pd.DataFrame(columns=['accuracy'], index=classifiers)
        for model, clf_name in zip(models, classifiers):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            accuracy.append(score)
            summary.loc[clf_name] = score
        
        st.write(summary)
        st.write(" ")
        scores = [accuracy[0], accuracy[3], accuracy[4], accuracy[1], accuracy[5], accuracy[2], accuracy[6]]
        algorithms = ["Support Vector Machine","Logistic Regression","Naive Bayes","K-Nearest Neighbors","Quadratic Discriminant Analysis","Decision Tree","Random Forest"]
        sns.set(rc={'figure.figsize': (15, 8)})
        fig, ax = plt.subplots()
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Accuracy score")
        ax.set_title("Accuracy scores of different machine learning algorithms")
        sns.barplot(x=algorithms, y=scores, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write('*The provided code creates a bar plot using Seaborn to visualize the accuracy scores of different machine learning algorithms. It uses the scores list containing accuracy values and algorithms list with algorithm names to plot a bar graph with algorithm names on the x-axis and accuracy scores on the y-axis. This visualization helps compare the performance of various algorithms in terms of accuracy.*')
        st.write(" ")
        st.write("Among the model trained and tested, RandomForestClassifier has overall the best accuracy")

### User Input for Predictions
def user_input_features():
    age = st.number_input('Age of persons (29 - 77): ', min_value=29, max_value=77, value=29, step=1)
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type', ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"))
    trtbps = st.number_input('Resting blood pressure (94 - 200): ', min_value=94, max_value=200, value=94, step=1)
    chol = st.number_input('Serum cholestrol in mg/dl (126 - 564): ', min_value=126, max_value=564, value=126, step=1)
    fbs = st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes', 'No'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ("Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
    thalachh = st.number_input('Maximum heart rate achieved thalach (71 - 202): ', min_value=71, max_value=202, value=71, step=1)
    exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.number_input(' ST depression induced by exercise relative to rest (oldpeak) (0 - 6.2): ')

    sex, cp, exang, fbs, restecg = preprocess(sex, cp, exang,fbs,restecg)

    data= {'age':age, 'sex':sex, 'cp':cp, 'trestbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalach':thalachh,
       'exang':exang, 'oldpeak':oldpeak
                }
    features = pd.DataFrame(data, index=[0])
    return features

# Model Prediction
with tab6:
    data = df_model.drop(['slope', 'ca', 'thal'], axis=1)
    X_test = data.drop('target', axis=1) 
    y_test = data['target']

    my_scaler = StandardScaler()
    my_scaler.fit(X_test)
    X_test_scaled = my_scaler.transform(X_test)
    
    def predict(data):
        data_scaled = my_scaler.transform(data)
        return rf_model.predict(data_scaled)
    
    st.markdown("Can we establish a dependable predictive tool for early detection?")
    st.write(" ")
    st.write("Enter the Required fields to check whether you have a healthy heart")
    input_df = user_input_features()    
    
    if st.button("Estimate"):
        result = predict(input_df)
        accuracy = rf_model.score(X_test_scaled, y_test)
                
        if (result[0]== 0):
            st.subheader('You are predicted to have no Heart Risk')
        else:
            st.subheader('You are predicted to likely have a Heart Risk')
    
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Bio      
with tab7:
    st.write("Hi there! I am Sandhya Kilari, currently pursuing Master's in Data Science. I'm an avid data scientist, passionate about extracting insights from data using various analytical tools and techniques. My expertise includes machine learning, statistical analysis, and data visualization.")
    st.write(" ")
    st.write("Thriving on challenges, I engage in impactful endeavors that matter. When I'm not diving into data, I love spending time in nature, capturing moments through photography, and honing my culinary skills by experimenting with different cuisines.")
    st.write(" ")
    st.write("I'm excited to be a part of this project because it aligns perfectly with my passion for leveraging data to create meaningful solutions. I believe that by applying data science principles, we can solve real-world problems and make a positive difference.")
    st.write(" ")
    st.write("Feel free to reach out if you have any questions or just want to discuss data science, philosophy, or anything else that piques your curiosity!")
    st.write(" ")
    st.image('profile.jpeg', width=300)
