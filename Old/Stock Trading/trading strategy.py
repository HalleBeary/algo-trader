from ib_insync import *
from data_collection import get_stock_data, get_news, analyze_sentiment
from utils import calculate_technical_indicators
from utils import prepare_features, train_model, predict


def trading_decision(ib, symbol, model):
    # Get latest data
    data = get_stock_data(symbol, period="2d")
    data = calculate_technical_indicators(data)
    news = get_news(symbol)
    sentiment = analyze_sentiment(news)
    features = prepare_features(data.tail(1), sentiment)
    prediction = predict(model, features)[0]

    if prediction == 1:
        order = MarketOrder('BUY', 100) # Adjust position size as needed
        trade = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), order)
        print(f"Placed BUY order for {symbol}")
    else:
        order = MarketOrder('SELL', 100) # Adjust position size as needed
        trade = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), order)
        print(f"Placed SELL order for {symbol}")


def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    symbols = ['AAPL', 'GOOGL', 'MSFT'] # Add more symbols as needed
    # Train models for each symbol
    models = {}
    for symbol in symbols:
        data = get_stock_data(symbol, period="1y")
        data = calculate_technical_indicators(data)
        sentiment = analyze_sentiment(get_news(symbol))
        features = prepare_features(data, sentiment)
        models[symbol] = train_model(features)

    # Main trading loop
    while True:
        for symbol in symbols:
            trading_decision(ib, symbol, models[symbol])
            ib.sleep(3600) # Wait for 1 hour before next iteration
            ib.disconnect()


if __name__ == "main":
        main()