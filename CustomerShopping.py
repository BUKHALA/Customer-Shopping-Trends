#importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import pickle

import warnings
warnings.filterwarnings('ignore')

#App title
st.title('Customer Shopping Trends')

#add an upload button for data upload
uploaded_file = st.file_uploader('Upload your input csv file',type=['csv'])

#create a dataframe from the uploaded file
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  #get input from user on the number of rows to display		
  num_rows = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=25, value=5)
  #show the top 10 rows of the dataframe
  st.header('Data Sample')
  st.dataframe(data.head(num_rows))

#Get a list of all the columns in the dataframe
columns = data.columns.tolist()

#Which gender has the highest number of purchase?
data.Gender.value_counts()
sns.barplot(data, x = 'Gender' , y = 'Purchase Amount (USD)')
plt.xlabel('Gender')
plt.ylabel('Purchase Amount (USD)')
plt.title('Purchasing Power')
plt.show()
st.pyplot(fig)

#Create a function to plot categorical variables
def plot_cat(data,cat_var):
    st.header('Plot of' + cat_var)
    fig,ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=data, x=cat_var)
    

# create a function to encode categorical variables
def encode_cat(data, cat_var):
    encoder = OrdinalEncoder()
    data[cat_var] = encoder.fit_transform(data[[cat_var]])
    return data

# for loop to encode all categorical variables
for i in data.columns:
    if data[i].dtypes == 'object':
        encode_cat(data, i)

#Show the top 3 rows of our updated dataframe
st.header('Data Encoded Dataframe Sample')
st.dataframe(data.head(3))

#Split the data uploaded into features and target
#Create our target and features
X = data.drop(columns=['Gender'])

model = pickle.load(open('model.pkl', 'rb'))

#Make predictions using the model
prediction = model.predict(X)

#Add the predictions to the dataframe
data['Gender_prediction'] = prediction

#get user input on the number of rows to display
num_rows_pred = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=50, value=5)

#Show the top 5 rows of a dataframe
st.header('Predictions')
st.dataframe(data.head(num_rows_pred))

#Print the classification report
st.header('Classification Report')
st.text("0=Will be male, 1=Will be female")

class_report = classification_report(data['Gender'],
                                     data['Gender_prediction'])
st.text(class_report)