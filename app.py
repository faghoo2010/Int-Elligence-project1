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
#   data = pd.read_csv("Foreign_Exchange_Rates.xls")
#   print(data.head())
#   data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
#   model = AutoTS(forecast_length=int(forecast), frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
#   model = model.fit(data, date_col = 'Time Serie', value_col=options[selected_option], id_col=None)
#   prediction = model.predict()
#   forecast = prediction.forecast
# return forecast

    #currently the model is trained on every submit action from streamlit, find a solution to this problem so that on every submit action, a pretrained model for each currecncy is loaded and inferenced.
    
import joblib  # Or your preferred model loading library

# Assuming you have a function to load your model
def make_forecast(selected_option, forecast):
    #loading data
    data = pd.read_csv("Foreign_Exchange_Rates.xls", index_col=0)
    data.replace('ND', np.nan, inplace=True)
    data.reset_index(drop=True)
    data.head()
    data.drop('Unnamed: 24', axis=1, inplace=True)
    data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
    data = data.dropna()
    def Sequential_Input_XGB(df, input_sequence):
        df_np = df.to_numpy()
        y = []
        X = []
        
        for i in range(len(df_np) - input_sequence):
            row = [a for a in df_np[i:i + input_sequence]]
            X.append(row)
            label = df_np[i + input_sequence]
            y.append(label)
        
            return np.array(X), np.array(y)
        
    for option in options.values():
        if selected_option == option:
            n_input = 10
            df = data[selected_option]
            X, y = Sequential_Input_XGB(df, n_input)
    
            # Training data
            X_train, y_train = X[:-1001], y[:-1001]

            # Validation data
            X_val, y_val = X[-1000:-61], y[-1000:-61]

            # Test data
            X_test, y_test = X[-60:], y[-60:]
            X_train, X_val, X_test = X_train.astype('float'), X_val.astype('float'), X_train.astype('float')

            y_train, y_val, y_test = y_train.astype('float'), y_val.astype('float'), y_train.astype('float')
            reg = XGBRegressor()
            reg.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=True) 
            reg_pred = reg.predict(X_test)

            forecast = reg_pred
        return forecast
   

with st.form(key='user_form'):
    # Add input widgets to the form
    # Create the selectbox
    selected_option = st.selectbox('Choose a currency:', options)
    forecast = st.number_input(
    "Enter a number",  # Label displayed to the user
    min_value=1,         # Minimum value allowed
    max_value=100,      # Maximum value allowed
    value=1,            # Default value
    step=1              # Increment step
)
    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    
    forecast = make_forecast(selected_option, forecast)
        
    st.write(forecast)
    st.line_chart(forecast)
    st.dataframe(forecast)
