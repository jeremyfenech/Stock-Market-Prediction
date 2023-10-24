import trader
from flask import Flask, redirect, render_template, send_file, url_for
from datetime import datetime

# Define the app using Flask
app = Flask(__name__)

# Function to fetch or load top gainers and losers data

# Redirect to the top movers page when the user visits the index page
@app.route('/')
def index():
    return redirect(url_for('top_movers'))


@app.route('/top-movers')
def top_movers():
    # Fetch the top gainers and losers data
    wdata = trader.get_top_movers()

    ldata = trader.get_top_movers(loss=True)

    data = {
        'tab_name': 'Top Daily Movers',
        'top_gainers_table': wdata,
        'top_losers_table': ldata,
        'date': datetime.today().strftime('%A, %d %B, %Y')
    }

    return render_template('top_movers.html', data=data)


@app.route('/view/<ticker>')
def view_ticker(ticker):

    # Fetch the ticker data
    stock_data = trader.get_ticker_data(ticker)

    try:
        # Get the latest price, previous price and price change
        latest_price = round(stock_data['Adjusted Close'].iloc[-1], 2)
        previous_price = round(stock_data['Adjusted Close'].iloc[-2], 2)
        price_change = round(latest_price - previous_price, 2)
    except (IndexError, TypeError, ValueError):
        latest_price = None
        previous_price = None
        price_change = None

    # Get the predictions for the ticker
    predictions = trader.predict_future_close(stock=stock_data)

    # Plot the buyability of the ticker
    trader.get_buyability(stock=stock_data, predictions=predictions)

    # Plot the indicators
    trader.plot_moving_averages(stock=stock_data, ticker=ticker)

    # Calculate the spacing between the selected items
    spacing = max(len(predictions) // 5, 1)

    # Select the evenly spaced items using iloc
    selected_predictions = predictions.iloc[::spacing]

    data = {
        'tab_name': f"{ticker} Predictions",
        'current_ticker': ticker,
        'latest_price': latest_price,
        'previous_price': previous_price,
        'price_change': price_change,
        'positive': price_change > 0 if price_change else None,
        'predictions': selected_predictions,
    }

    return render_template('view_ticker.html', data=data)


@app.route('/index/<name>')
def view_index(name):

    if name == 'sp500':
        proper_name = 'S&P 500'
    elif name == 'nasdaq100':
        proper_name = 'NASDAQ 100'

    # Fetch the index data
    index_data = trader.get_index_tickers(name)

    data = {
        'tab_name': f"{proper_name} Tickers",
        'index_name': proper_name,
        'index_data': index_data,
    }

    return render_template('index_tickers.html', data=data)


@app.route('/about')
def about():

    data = {'tab_name': 'Project About Page'}

    return render_template('about.html', data=data)


# Get media links
@app.route('/gauge')
def gauge():
    return send_file('visuals/gauge.png', mimetype='image/png')


@app.route('/ma-plot')
def ma_plot():
    return send_file('visuals/ma_plot.png', mimetype='image/png')


@app.route('/predict-plot')
def predict_plot():
    return send_file('visuals/prediction.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
