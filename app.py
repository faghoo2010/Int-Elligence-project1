from autots import AutoTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from seaborn import regression
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')


import streamlit as st
st.title(("Future Forex Currency Price Prediction Model"))

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN$': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON$': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN$': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

#function to make predictions, we'll use the code from analysis.ipynb file and make a function which would return forecasts
#def make_forecast(selected_option,forecast):
#   data = pd.read_csv("‪Foreign_Exchange_Rates.xls")
#   print(data.head())
#   data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
#   model = AutoTS(forecast_length=int(forecast), frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
#   model = model.fit(data, date_col = 'Time Serie', value_col=options[selected_option], id_col=None)
#   prediction = model.predict()
#   forecast = prediction.forecast
# return forecast

#currently the model is trained on every submit action from streamlit, find a solution to this problem so that on every submit action, a pretrained model for each currecncy is loaded and inferenced.
    
import joblib  # Or your preferred model loading library
def Sequential_Input_XGB(df, input_sequence):# function to split data into X and y values
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)

data = pd.read_csv("Foreign_Exchange_Rates.xls", index_col=0)# load and clean the data
data.replace('ND', np.nan, inplace=True)
data.reset_index(drop=True)
data.head()
data.drop('Unnamed: 24', axis=1, inplace=True)
data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
data = data.dropna()
n_input = 10

# Load the pretrained models
models = ["C:/Users/User/Desktop/Int-Elligence/models/AUSTRALIAN DOLLAR", 
"C:/Users/User/Desktop/Int-Elligence/models/MEXICAN PESO",
"C:/Users/User/Desktop/Int-Elligence/models/BRAZILIAN REAL",
"C:/Users/User/Desktop/Int-Elligence/models/CANADIAN DOLLAR",
"C:/Users/User/Desktop/Int-Elligence/models/CHINESE YUAN$",
"C:/Users/User/Desktop/Int-Elligence/models/DANISH KRONE",
"C:/Users/User/Desktop/Int-Elligence/models/EURO",
"C:/Users/User/Desktop/Int-Elligence/models/GREAT BRITAIN POUNDS",
"C:/Users/User/Desktop/Int-Elligence/models/HONG KONG DOLLAR",
"C:/Users/User/Desktop/Int-Elligence/models/INDIAN RUPEE",
"C:/Users/User/Desktop/Int-Elligence/models/JAPANESE YEN$",
"C:/Users/User/Desktop/Int-Elligence/models/MALAYSIAN RINGGIT",
"C:/Users/User/Desktop/Int-Elligence/models/THAI BAHT",
"C:/Users/User/Desktop/Int-Elligence/models/NEW TAIWAN DOLLAR",
"C:/Users/User/Desktop/Int-Elligence/models/NEW ZEALAND DOLLAR",
"C:/Users/User/Desktop/Int-Elligence/models/NORWEGIAN KRONE",
"C:/Users/User/Desktop/Int-Elligence/models/SINGAPORE DOLLAR",
"C:/Users/User/Desktop/Int-Elligence/models/SOUTH AFRICAN RAND$",
"C:/Users/User/Desktop/Int-Elligence/models/SRILANKAN RUPEE",
"C:/Users/User/Desktop/Int-Elligence/models/SWEDEN KRONA",
"C:/Users/User/Desktop/Int-Elligence/models/SWISS FRANC"]

for model in models:
    model = joblib.load(model)  
    def make_forecast(X_true, forecast_length):
        model_pred = model.predict(X_true)
        forecast = model_pred[-forecast_length : ]
        return forecast
   

with st.form(key='user_form'):
    # Add input widgets to the form
    # Create the selectbox
    selected_option = st.selectbox('Choose a currency:', options.values())
        
    forecast_length = st.number_input(
        
    "Enter a number",  # Label displayed to the user
    min_value=1,         # Minimum value allowed
    max_value=10,      # Maximum value allowed
    value=1,            # Default value
    step=1              # Increment step
    )
    
    X, y = Sequential_Input_XGB(data[selected_option], n_input)
    X_test, y_test = X[-60:], y[-60:]
    X_test = X_test.astype('float')
    

    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    
    forecast = make_forecast(X_test, forecast_length)
        
    st.write(forecast)
    st.line_chart(forecast)
    st.dataframe(forecast)









































































