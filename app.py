# Library import
import plotly.express as px
import streamlit as st
import yfinance as yf
from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title='Stock dashboard',
    page_icon=':money_with_wings:'
    )
st.title('Stock prices forecasting')

st.sidebar.subheader('Ticker query parameters')
# Ticker sidebar
with open('ticker_symbols.txt', 'r') as fp:
    ticker_list = fp.read().split('\n')
ticker_selection = st.sidebar.selectbox(label='Stock ticker', options=ticker_list, index=ticker_list.index('AAPL'))
period_list = ['6mo', '1y', '2y', '5y', '10y', 'max']
period_selection = st.sidebar.selectbox(label='Period', options=period_list, index=period_list.index('2y'))

# Retrieving tickers data
ticker_data = yf.Ticker(ticker_selection)
ticker_df = ticker_data.history(period=period_selection)
ticker_df = ticker_df.rename_axis('Date').reset_index()
ticker_df['Date'] = ticker_df['Date'].dt.date

# Prophet sidebar
st.sidebar.subheader('Prophet parameters configuration')
horizon_selection = st.sidebar.slider('Forecasting horizon (days)', min_value=1, max_value=365, value=90)
growth_selection = st.sidebar.radio(label='Growth', options=['linear', 'logistic'])
seasonality_selection = st.sidebar.radio(label='Seasonality', options=['additive', 'multiplicative'])
with st.sidebar.expander('Seasonality components'):
    weekly_selection = st.checkbox('Weekly')
    monthly_selection = st.checkbox('Monthly', value=True)
    yearly_selection = st.checkbox('Yearly', value=True)
with open('holiday_countries.txt', 'r') as fp:
    holiday_country_list = fp.read().split('\n')
    holiday_country_list.insert(0, 'None')
holiday_country_selection = st.sidebar.selectbox(label="Holiday country", options=holiday_country_list)

# Ticker information
company_name = ticker_data.info['longName']
st.header(company_name)
company_summary = ticker_data.info['longBusinessSummary']
st.info(company_summary)

st.header('Ticker data')
# Ticker data
var_list = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
st.dataframe(ticker_df[var_list])

st.header('Forecasting')
# Prophet model fitting
with st.spinner('Model fitting..'):
    prophet_df = ticker_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(
        seasonality_mode=seasonality_selection,
        weekly_seasonality=weekly_selection,
        yearly_seasonality=yearly_selection,
        growth=growth_selection,
        )
    if holiday_country_selection != 'None':
        model.add_country_holidays(country_name=holiday_country_selection)      
    if monthly_selection:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)

# Prophet model forecasting
with st.spinner('Making predictions..'):
    future = model.make_future_dataframe(periods=horizon_selection, freq='D')
    forecast = model.predict(future)

# Prophet forecast plot
fig = px.scatter(prophet_df, x='ds', y='y', labels={'ds': 'Day', 'y': 'Close'})
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='yhat_lower')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='yhat_upper')
st.plotly_chart(fig)
