# Import necessary libraries
import holidays
import requests
import streamlit as st
import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly

# Page configuration
st.set_page_config(page_title="Stock Dashboard", page_icon=":money_with_wings:")

st.title("Stock Prices Forecasting")


# Helper function to get ticker list with caching
@st.cache_data
def get_ticker_list():
    response = requests.get(
        "https://dumbstockapi.com/stock?format=tickers-only&countries=US"
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching ticker list.")
        return ["AAPL"]


# Helper function to fetch ticker data with caching
@st.cache_data
def fetch_ticker_data(ticker_symbol, period):
    try:
        ticker = yf.Ticker(ticker_symbol)
        ticker_df = ticker.history(period=period)
        if ticker_df.empty:
            st.error(
                f"No data available for ticker '{ticker_symbol}'. It may be delisted."
            )
            return None, None
        else:
            ticker_info = ticker.info
            return ticker_df, ticker_info
    except Exception as e:
        st.error(f"Error retrieving data for ticker '{ticker_symbol}': {e}")
        return None, None


# Helper function to fit the Prophet model with caching
@st.cache_data
def fit_prophet_model(
    prophet_df,
    growth,
    seasonality_mode,
    weekly,
    monthly,
    yearly,
    holidays_country,
    cap=None,
):
    model = Prophet(
        growth=growth,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=False,
    )
    if holidays_country != "None":
        model.add_country_holidays(country_name=holidays_country)
    if monthly:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    if growth == "logistic" and cap is not None:
        prophet_df["cap"] = cap
    model.fit(prophet_df)
    return model


# Sidebar - Ticker selection
st.sidebar.subheader("Ticker Query Parameters")
ticker_list = get_ticker_list()
ticker_selection = st.sidebar.selectbox(
    label="Stock Ticker",
    options=sorted(ticker_list),
    index=ticker_list.index("AAPL") if "AAPL" in ticker_list else 0,
)
period_list = ["1y", "2y", "5y", "10y", "max"]
period_selection = st.sidebar.selectbox(label="Period", options=period_list, index=1)

# Fetch data
ticker_df, ticker_info = fetch_ticker_data(ticker_selection, period_selection)

if ticker_df is not None and ticker_info is not None:
    # Prepare data
    ticker_df.reset_index(inplace=True)
    ticker_df["Date"] = ticker_df["Date"].dt.date

    # Display company information
    company_name = ticker_info.get("longName", "Unknown Company")
    st.header(company_name)
    company_summary = ticker_info.get("longBusinessSummary", "No summary available.")
    st.info(company_summary)

    # Display ticker data
    st.header("Ticker Data")
    with st.expander("Show Ticker Data"):
        st.dataframe(ticker_df[["Date", "Open", "High", "Low", "Close", "Volume"]])

    # Sidebar - Prophet parameters
    st.sidebar.subheader("Prophet Parameters Configuration")
    horizon_selection = st.sidebar.slider(
        "Forecasting Horizon (days)", min_value=1, max_value=365, value=90
    )
    growth_selection = st.sidebar.radio("Growth", options=["linear", "logistic"])
    # Additional parameters for logistic growth
    cap_close = None
    if growth_selection == "logistic":
        st.sidebar.info(
            "Configure logistic growth saturation as a percentage of latest Close"
        )
        cap = st.sidebar.slider(
            "Carrying Capacity Multiplier", min_value=1.0, max_value=2.0, value=1.2
        )
        cap_close = cap * ticker_df["Close"].iloc[-1]
        ticker_df["cap"] = cap_close
    seasonality_selection = st.sidebar.radio(
        "Seasonality Mode", options=["additive", "multiplicative"]
    )

    # Seasonality components
    st.sidebar.subheader("Seasonality Components")
    weekly_selection = st.sidebar.checkbox("Weekly Seasonality", value=True)
    monthly_selection = st.sidebar.checkbox("Monthly Seasonality", value=True)
    yearly_selection = st.sidebar.checkbox("Yearly Seasonality", value=True)

    # Holiday effects
    holiday_country_list = ["None"] + sorted(holidays.list_supported_countries())
    holiday_country_selection = st.sidebar.selectbox(
        "Holiday Country", options=holiday_country_list
    )

    # Prepare data for Prophet
    prophet_df = ticker_df.rename(columns={"Date": "ds", "Close": "y"})
    if growth_selection == "logistic" and cap_close is not None:
        prophet_df["cap"] = cap_close

    # Forecasting
    st.header("Forecasting")
    with st.spinner("Fitting the model..."):
        model = fit_prophet_model(
            prophet_df,
            growth=growth_selection,
            seasonality_mode=seasonality_selection,
            weekly=weekly_selection,
            monthly=monthly_selection,
            yearly=yearly_selection,
            holidays_country=holiday_country_selection,
            cap=cap_close,
        )

    with st.spinner("Generating forecast..."):
        future = model.make_future_dataframe(periods=horizon_selection)
        if growth_selection == "logistic" and cap_close is not None:
            future["cap"] = cap_close
        forecast = model.predict(future)

    # Interactive Plotly forecast plot
    st.subheader("Interactive Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # Interactive Plotly forecast components
    st.subheader("Forecast Components")
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2)
else:
    st.warning("Please select a different ticker or time period.")
