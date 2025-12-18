import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to clean data
def clean_data(data, remove_duplicates=False, convert_dtypes=False):
    # Handle missing values
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].mean())
        else:
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    # Remove duplicates
    if remove_duplicates:
        data = data.drop_duplicates(keep='first')
    
    # Convert data types
    if convert_dtypes:
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='raise')
                except:
                    try:
                        data[col] = pd.to_datetime(data[col], errors='raise')
                    except:
                        continue
    
    return data

# Initialize the app
st.title('Comprehensive Data Analysis Dashboard')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset in CSV format:", type='csv')

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.success('File successfully uploaded.')

    # Data cleaning options
    st.sidebar.header("Data Cleaning Options")
    remove_duplicates = st.sidebar.checkbox("Remove duplicates")
    convert_dtypes = st.sidebar.checkbox("Convert data types")
    if st.sidebar.button("Clean Data"):
        data = clean_data(data, remove_duplicates, convert_dtypes)
        st.success('Data cleaned successfully.')

    # Options for what to do with the data
    task = st.sidebar.selectbox("Choose a task:", [
        "Show Data", "Descriptive Statistics", "Visualizations", "Correlation Analysis", "Machine Learning"
    ])

    if task == "Machine Learning":
        st.sidebar.subheader("Machine Learning Settings")
        target_col = st.sidebar.selectbox("Select Target Column", data.columns)
        test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)

        if st.sidebar.button("Train Model"):
            X = data.drop(target_col, axis=1)
            y = data[target_col]

            # Encoding categorical variables if necessary
            if X.select_dtypes(include=['object']).any().any():
                for column in X.columns:
                    if X[column].dtype == 'object':
                        le = LabelEncoder()
                        X[column] = le.fit_transform(X[column])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            st.write("Accuracy:", accuracy)
            st.text("Classification Report:\n" + classification_report(y_test, predictions))

    elif task == "Show Data":
        st.write(data)

    elif task == "Descriptive Statistics":
        st.write(data.describe())

    elif task == "Visualizations":
        plot_type = st.sidebar.radio("Select plot type:", [
            "Line Plot", "Bar Chart", "Histogram", "Box Plot", "Scatter Plot",
            "Heatmap", "Pie Chart", "3D Plot"
        ])
        columns = data.columns.tolist()
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if plot_type == "Line Plot":
            x_col = st.sidebar.selectbox("X-Axis", columns)
            y_col = st.sidebar.selectbox("Y-Axis", columns)
            fig = px.line(data, x=x_col, y=y_col)
            st.plotly_chart(fig)

        elif plot_type == "Bar Chart":
            x_col = st.sidebar.selectbox("X-Axis", columns)
            fig = px.bar(data, x=x_col)
            st.plotly_chart(fig)

        elif plot_type == "Histogram":
            x_col = st.sidebar.selectbox("Variable", columns)
            fig = px.histogram(data, x=x_col)
            st.plotly_chart(fig)

        elif plot_type == "Box Plot":
            y_col = st.sidebar.selectbox("Variable", columns)
            fig = px.box(data, y=y_col)
            st.plotly_chart(fig)

        elif plot_type == "Scatter Plot":
            x_col = st.sidebar.selectbox("X-Axis", numeric_columns)
            y_col = st.sidebar.selectbox("Y-Axis", numeric_columns)
            fig = px.scatter(data, x=x_col, y=y_col)
            st.plotly_chart(fig)

        elif plot_type == "Heatmap":
            if len(numeric_columns) > 1:
                fig = px.imshow(data[numeric_columns].corr())
                st.plotly_chart(fig)
            else:
                st.error("Not enough numeric columns for heatmap.")

        elif plot_type == "Pie Chart":
            if categorical_columns:
                col = st.sidebar.selectbox("Category", categorical_columns)
                fig = px.pie(data, names=col)
                st.plotly_chart(fig)
            else:
                st.error("No categorical data for a pie chart.")

        elif plot_type == "3D Plot":
            if len(numeric_columns) >= 3:
                x_col = st.sidebar.selectbox("X-Axis", numeric_columns)
                y_col = st.sidebar.selectbox("Y-Axis", numeric_columns)
                z_col = st.sidebar.selectbox("Z-Axis", numeric_columns)
                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col)
                st.plotly_chart(fig)

    elif task == "Correlation Analysis":
        if data.select_dtypes(include=[np.number]).empty:
            st.error("No numeric columns available for correlation matrix.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    

else:
    st.info('Awaiting the upload of a CSV file.')
