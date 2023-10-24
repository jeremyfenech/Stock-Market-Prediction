from datetime import datetime as dt
import os
from bs4 import BeautifulSoup
import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import csv
import datetime
import requests
from urllib.parse import quote
import talib

# Set the matplotlib backend to 'Agg'   
matplotlib.use('Agg')

# Web headers for the GET request
get_headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

# Load the Random Forest model from the file
model = joblib.load('models/random_forest_model.joblib')


def get_top_movers(loss=False):
    if loss:
        type = 'losers'
    else:
        type = 'gainers'

    # Top 100 gainers or losers
    count = 100

    base_url = f'https://finance.yahoo.com/{type}?offset=0&count={count}'

    # Send a GET request to the URL
    response = requests.get(base_url, headers=get_headers)

    # Create a BeautifulSoup object with the response text
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the data
    table = soup.find('table')

    # Extract the table headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # Extract the table rows
    rows = []
    for tr in table.find_all('tr'):
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        if row:
            rows.append(row)

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(rows, columns=headers)

    df = df[['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change']]

    # Rename price column
    df = df.rename(columns={'Price (Intraday)': 'Price'})

    # Return the DataFrame
    return df


def get_ticker_data(symbol):
    # Define the base URL for the Yahoo Finance API
    base_url = 'https://query1.finance.yahoo.com/v7/finance/download'

    # Define the parameters for the API request
    params = {
        'period1': 0,
        'period2': int(datetime.datetime.now().timestamp()),
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }

    # Construct the URL for the API request for this stock
    url = f"{base_url}/{quote(symbol)}"

    # Send a GET request to the API and retrieve the response
    response = requests.get(url, headers=get_headers, params=params)

    if response.status_code == 200:
        # Parse the response content as a CSV file and extract the required data
        data = list(csv.DictReader(response.content.decode().splitlines()))

        df = pd.DataFrame(data)
        # Assume 'df' is the DataFrame containing the columns (Open, High, Low, Close, Adj Close, Volume)

        new_column_order = ['Date', 'Low', 'Open',
                            'Volume', 'High', 'Close', 'Adjusted Close']

        # Rename the 'Adj Close' column to 'Adjusted Close'
        df = df.rename(columns={'Adj Close': 'Adjusted Close'})

        # Reindex the DataFrame with the new column order
        df = df.reindex(columns=new_column_order)

        # Convert numeric columns to float type
        num_cols = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
        df[num_cols] = df[num_cols].astype(float)

        # Convert date strings to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])

        df = df.dropna()

        # Calculate Log Returns
        df['LogReturns'] = np.log(
            df['Adjusted Close'] / df['Adjusted Close'].shift(1))

        # Calculate TEMA
        df['TEMA'] = talib.TEMA(df['Adjusted Close'], timeperiod=20)

        # Calculate Simple Moving Average (SMA)
        df['SMA_10'] = talib.SMA(df['Adjusted Close'], timeperiod=10)

        # Calculate Exponential Moving Average (EMA)
        df['EMA_20'] = talib.EMA(df['Adjusted Close'], timeperiod=20)

        # Calculate RSI
        df['RSI'] = talib.RSI(df['Adjusted Close'], timeperiod=14)

        # Calculate MACD
        macd, macd_signal, macd_histogram = talib.MACD(df['Adjusted Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_histogram

        return df
    else:
        print(f"Error: {response.status_code}")


def plot_moving_averages(stock, ticker):

    # Select the last year of data
    stock = stock.tail(365)
    
    last_price = stock['Adjusted Close'].iloc[-1]

    distance = last_price - stock['EMA_20'].iloc[-1]

    # Plot an arrow based on the signal strength
    if distance < 0-stock['EMA_20'].iloc[-1]*0.02:
        arrow = u'$\u2193$'  # down arrow
        color = 'red'
    elif distance > stock['EMA_20'].iloc[-1]*0.02:
        arrow = u'$\u2191$'  # up arrow
        color = 'green'
    else:
        arrow = u'$\u2194$'  # left-right arrow
        color = 'orange'

    # Plot the stock data and indicators
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock['Date'], stock['Adjusted Close'], label='Adjusted Close')
    ax.plot(stock['Date'], stock['SMA_10'], label='SMA_10')
    ax.plot(stock['Date'], stock['EMA_20'], label='EMA_20')
    ax.legend()
    ax.set_title(f"${ticker} Stock Analysis")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.annotate(arrow, xy=(stock['Date'].max(), last_price), xytext=(stock['Date'].max() - pd.Timedelta(days=5), last_price),
                fontsize=35, color=color)

    # Create a twin axis for RSI
    ax2 = ax.twinx()
    ax2.plot(stock['Date'], stock['RSI'], label='RSI', color='purple')
    ax2.axhline(y=30, color='red', linestyle='--')
    ax2.axhline(y=70, color='red', linestyle='--')
    ax2.set_ylabel('RSI')
    ax2.legend(loc='lower right')

    # Show the plot
    plt.savefig('visuals/ma_plot.png', bbox_inches='tight',
                pad_inches=0.1, transparent=True)

    plt.close()


def get_buyability(stock, predictions):

    # Calculate the maximum profit
    max_profit = predictions['Adjusted Close'].max() - stock['Adjusted Close'].iloc[-1]

    # Calculate the signal strength
    signal_strength = max_profit / stock['Adjusted Close'].iloc[-1]
    
    # Normalize the signal strength between -1 and 1
    if signal_strength > 1:
        signal_strength = 1
    elif signal_strength < -1:
        signal_strength = -1    

    # Calculate the angle of the arrow, as a value between -90 and 90 degrees and invert the values
    arrow_angle = 180-((signal_strength + 1) * 90)

    # Calculate the sin and cos of the arrow angle
    arrow_sin = np.sin(np.radians(arrow_angle))
    arrow_cos = np.cos(np.radians(arrow_angle))

    # Create a semi-circle gauge with a red-green color gradient
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Rotate the plot by 90 degrees
    ax.invert_xaxis()

    ax.add_patch(patches.Wedge((0.5, 0.05), 0.4, 0, 180, width=0.2,
                               edgecolor='black', facecolor='none', linewidth=2))
    # Set the color gradient
    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(-1, 1)
    # Plot the color gradient
    for angle in np.linspace(-90, 88, 100):
        start_angle = 90 + angle
        color = cmap(norm(angle / 90))
        ax.add_patch(patches.Wedge((0.5, 0.05), 0.4, start_angle, start_angle + 1.8,
                                   width=0.2, edgecolor='none', facecolor=color))

    # Rotate the arrow by 90 degrees
    arrow = patches.Arrow(
        0.5, 0.05, -0.3 * arrow_cos, 0.3 * arrow_sin, width=0.2, color='black')
    ax.add_patch(arrow)

    # Add the "Buy", "Sell", and "Hold" labels
    ax.text(0.02, 0.05, "Buy", fontsize=12, ha="left", va="center")
    ax.text(0.98, 0.05, "Sell", fontsize=12, ha="right", va="center")
    ax.text(0.5, 0.45, "Hold", fontsize=12, ha="center", va="bottom")

    # Set the limits and axis labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.savefig('visuals/gauge.png', bbox_inches='tight',
                pad_inches=0.1, transparent=True)

    plt.close()


def predict_future_close(stock):

    no_date_data = stock.drop('Date', axis=1)

    # Select the last 20 days
    last_items = no_date_data.tail(20)

    # Predict price of next 20 days
    pred = model.predict(last_items)

    # Select the last rows
    stock = stock.tail(62)

    # Plot the historical data
    plt.figure(figsize=(12, 6))
    plt.plot(stock['Date'], stock['Adjusted Close'], label='Historical Price')

    # Generate the dates for the next 20 days
    last_date = stock['Date'].iloc[-1]
    next_dates = pd.date_range(last_date, periods=20, freq='D')

    # Plot the predicted data
    plt.plot(next_dates, pred, color='red', label='Predicted Price')

    # Connect the last historical price with the first predicted price
    plt.plot([stock['Date'].iloc[-1], next_dates[0]],
             [stock['Adjusted Close'].iloc[-1], pred[0]], color='purple')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Historical and Predicted Adjusted Close Prices')

    # Add legend
    plt.legend()

    # Show the plot
    plt.savefig('visuals/prediction.png', bbox_inches='tight',
                pad_inches=0.1, transparent=True)

    plt.close()

    df_predicted = pd.DataFrame({'Date': next_dates, 'Adjusted Close': pred})

    # Convert the dates to "dd/mm/yyyy" format
    df_predicted['Date'] = df_predicted['Date'].dt.strftime('%d/%m/%Y')

    # Round the Adjusted Close values to 2 decimal places
    df_predicted['Adjusted Close'] = df_predicted['Adjusted Close'].round(2)

    # Calculate the spacing between the selected items
    spacing = max(len(df_predicted) // 5, 1)

    # Select the evenly spaced items using iloc
    selected_df = df_predicted.iloc[::spacing]

    return selected_df


def get_index_tickers(index='sp500'):

    base_url = f'https://www.slickcharts.com/{index}'

    # Send a GET request to the URL
    response = requests.get(base_url, headers=get_headers)

    # Create a BeautifulSoup object with the response text
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the data
    table = soup.find('table')

    # Extract the table headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # Extract the table rows
    rows = []
    for tr in table.find_all('tr'):
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        if row:
            rows.append(row)

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(rows, columns=headers)

    df = df[['Symbol', 'Company', 'Price', 'Chg', '% Chg']]

    # Rename a single column
    df = df.rename(columns={'Chg': 'Change', '% Chg': '% Change'})

    # Remove symbols from % Change column
    df['% Change'] = df['% Change'].str.replace(r'[()%]', '', regex=True)

    # Remove commas from string columns and convert to float
    numeric_columns = ['Price', 'Change', '% Change']
    df[numeric_columns] = df[numeric_columns].apply(
        lambda x: x.str.replace(',', '').astype(float))

    # Return the DataFrame
    return df