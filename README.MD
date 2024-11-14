# Stock Prices Forecasting

"Information Systems and Business Intelligence" laboratory session repo: a [Streamlit](https://streamlit.io) dashboard for stock prices prediction provided by [Yahoo! Finance](https://finance.yahoo.com/) with [Prophet](https://facebook.github.io/prophet/).

Ticker symbols are fetched from the [dumb stock api](https://dumbstockapi.com/) and market data are fetched from the [yfinance](https://github.com/ranaroussi/yfinance) Python library, while the LaTeX beamer about the math behind Prophet is strongly inspired by [this](https://medium.com/analytics-vidhya/time-series-analysis-using-prophet-in-python-part-1-math-explained-5936509c175c) Medium post.

## How to Run It

You can choose one of the following two options to run the application: **Local deployment using a virtual environment (venv)** or **Docker deployment**.

### Option 1: Local Deployment with venv

1. Clone the repository:

   ```bash
   git clone https://github.com/Uzarel/Stock-prices-forecasting-webapp
   cd Stock-prices-forecasting-webapp
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   
   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

4. Install the required dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

### Option 2: Docker Deployment

1. Clone the repository:

   ```bash
   git clone https://github.com/Uzarel/Stock-prices-forecasting-webapp
   cd Stock-prices-forecasting-webapp
   ```

2. Build the Docker image using the provided `Dockerfile`:

   ```bash
   docker build -t stock-price-forecasting .
   ```

3. Run the Docker container:

   ```bash
   docker run -p 8501:8501 stock-price-forecasting
   ```

4. Access the Streamlit app by visiting `http://localhost:8501` in your web browser.

## Screenshot

![Demo screenshot](/assets/screen.png)
