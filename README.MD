# Stock prices forecasting

"Information Systems and Business Intelligence" laboratory session repo: a [Streamlit](https://streamlit.io) dashboard for stock prices prediction provided by [Yahoo! Finance](https://finance.yahoo.com/) with [Prophet](https://facebook.github.io/prophet/).

Ticker symbols are fetched from the [dumb stock api](https://dumbstockapi.com/) and market data are fetched from the [yfinance](https://github.com/ranaroussi/yfinance) Python library, while the LaTeX beamer about the math behind Prophet is strongly inspired by [this](https://medium.com/analytics-vidhya/time-series-analysis-using-prophet-in-python-part-1-math-explained-5936509c175c) Medium post.

## How to run it

Clone this repo and then:
```
pip install -r requirements.txt
streamlit run app.py
```

## Screenshot

![Demo screenshot](/assets/screen.png)
