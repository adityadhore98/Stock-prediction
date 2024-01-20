import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller



app_name = 'Stock Trend Prediction Web Application'
st.title(app_name)
st.subheader('This application is created to forecast the stock market price of the selected company.')
st.image("https://tradingqna.com/uploads/default/original/2X/a/ab1f8a4349929792f7ac3ac82062f4526bcc8e3a.jpg")

st.sidebar.header('Select The Parameter From below')
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime("2020-01-1", format="%Y-%m-%d"))
end_date = st.sidebar.date_input('End date',value=pd.to_datetime("today", format="%Y-%m-%d"))

ticker_list=["AAPL","MSFT","GOOGL","META","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP","TITN","TATAMOTORS.NS","MRF.NS"]
ticker = st.sidebar.selectbox('Select The Company',ticker_list)

#fetch data from user inputs using yfinance library
data = yf.download(ticker,start=start_date,end=end_date)

#add Date as a columns  to the dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from',start_date,'to',end_date)

st.write(data)

#plot the data
st.header('Data Visualization')
st.subheader('plot of the data')
st.write("**Note:** Select your Specific data range on the sidebar , zoom in on the plot  and select your specific column")
fig = px.line(data,x='Date', y=data.columns, title='Closing Price of the Stock',width=1000,height=600)
st.plotly_chart(fig)

#add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting ',data.columns[1:])

#subsetting the data
data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

#Adf test check stationarity
st.header("Is Data Stationary?")
# st.write("**Note:** If p-value is less than  0.05,then the data is stationary ")
st.write(adfuller(data[column])[1]<0.05)

#Lets decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#make same plot in plotly
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend,title='Trend',width=1200,height=400,labels={'x':'Date', 'y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal,title='Seasonality',width=1200,height=400,labels={'x':'Date', 'y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid,title='Residual',width=1200,height=400,labels={'x':'Date', 'y':'Price'}).update_traces(line_color='Red',line_dash='dot'))

#lets run the model
#user input for three parameter of the modle and seasonal order
p=st.slider('Select the Value of p',0,5,2)
d=st.slider('Select the Value of d',0,5,1)
q=st.slider('Select the Value of q',0,5,2)
seasonal_order=st.number_input('Select the value of seasonal p',0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model= model.fit()

#Print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")


st.markdown('<p style="color:orange; font-weight: bold; font-size: 36px;">Forecasting The Data</p>', unsafe_allow_html=True)
#predict the future values(ForeCasting)
forecast_period = st.number_input('Select the number of days to forecast',1,365,10)

#predict the future values
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions= predictions.predicted_mean
#st.write(predictions)



#add index to result dataframes as dates
predictions.index=pd.date_range(start=end_date, periods=len(predictions),freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0,'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## predictions", predictions)
st.write("## Actual Data", data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color="blue")))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color="red")))
fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="price", width=1200, height=400  )
st.plotly_chart(fig)






















