import streamlit as st 
import seaborn as sns 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# making containers
header = st.container()
data_sets = st.container()
model_training = st.container()
features = st.container()

with header:
    st.title("Titanic Chronicles: A Captivating Mobile App")
    st.text("Join me for a thrilling dive into the Titanic dataset!")
    
with data_sets:
    st.header("Thip, now ruling the Rubber Ducky empire!🛁🦆 #BathBoss") 
    st.text("Let's dive into the Titanic dataset for a quick yet insightful exploration!")
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader("Bar Chart according to the column 'Sex'")
    st.bar_chart(df['sex'].value_counts())
    
    
    # other plots
    st.subheader("Another example of a Bar chart according to the column 'pClass'")
    st.bar_chart(df['pclass'].value_counts())
    
    # barplot
    st.subheader("An example of a Bar plot according to the column 'Age'")
    st.bar_chart(df['age'].sample(10))
    
    
with model_training:
    st.header("Model training")
    st.markdown("We have created a Bar chart based on the column 'age', we have made a slider to incorporate new parameters or adjust existing ones.")
    # making new columns
    input, display= st.columns(2)
    
    # columns selection points 
    max_depth= input.slider("**Ages of people**", min_value=10, max_value=100, value=20, step=5)
    
    # n_estimator
    n_estimators = input.selectbox("**How many individuals boarded the Titanic based on their passenger class?**", options=[1, 2, 3])
        
    # adding list of features 
    input.write(df.columns)       
        
  # Input features from users
input_features = input.text_input("Which feature should we use According to you??")

# Check if the entered feature is not an empty string and is in the DataFrame columns
if input_features.strip() and input_features in df.columns:
    # Machine learning model
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)



    # define x and y values
    X = df[[input_features]]
    y = df['fare']  # Target variable

    try:
        # Fitting our model
        model.fit(X, y)
        pred = model.predict(X)

        # Display metrics
        display.subheader("Mean absolute error of the Model: ")
        display.write(mean_absolute_error(y, pred))
        display.subheader("Mean squared error of the Model: ")
        display.write(mean_squared_error(y, pred))
        display.subheader("R squared score of the Model: ")
        display.write(r2_score(y, pred))

    except Exception as e:
        st.error(f"An error occurred during model training: {str(e)}")

else:
    st.warning("Please enter a valid non-empty feature that exists in the DataFrame.")

    
    
    
with features: 
    st.subheader("Below are many features that are stored inside of the Titanic dataset!")  
    st.markdown("> * ### Let's incorporate a few additional features!")
    st.markdown("1. ##### **PassengerId:** A unique identifier for each passenger.")
    st.markdown("2. ##### **Survived:** Indicates whether the passenger survived (1) or did not survive (0).")
    st.markdown("3. ##### **Pclass:** Ticket class, representing the socio-economic status of the passenger (1 = 1st class, 2 = 2nd class, 3 = 3rd class).")
    st.markdown("4. ##### **Name:** Name of the passenger.")
    st.markdown("5. ##### **Sex:** Gender of the passenger (male or female).")
    st.markdown("6. ##### **Age:** Age of the passenger. (Some entries may be missing and denoted as NaN)")
    st.markdown("7. ##### **SibSp:** Number of siblings/spouses aboard the Titanic.")
    st.markdown("8. ##### **Parch:** Number of parents/children aboard the Titanic.")
    st.markdown("9. ##### **Ticket:** Ticket number.")
    st.markdown("10. ##### **Fare:** Fare paid for the ticket.")
    st.markdown("11. ##### **Cabin:** Cabin number where the passenger stayed. (Some entries may be missing)")
    st.markdown("12. ##### **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).")
    
    
