import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from data_collection import get_stock_data, get_news, analyze_sentiment
from utils import calculate_technical_indicators

def prepare_sequences(features, sequence_length=10):
    """Prepare sequences for LSTM"""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(features['Target'].iloc[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_features(data, sentiment):
    features = data[['RSI', 'MACD', 'Signal_Line']].copy()
    features['Sentiment'] = sentiment
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Scale features to [0,1] range
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    return features_scaled.dropna(), scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(features, sequence_length=10):
    X, y = prepare_sequences(features, sequence_length)
    
    # Reshape for LSTM [samples, time steps, features]
    n_features = X.shape[2]
    
    model = create_lstm_model((sequence_length, n_features))
    model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    return model

def backtest(symbol, start_date, end_date, initial_capital=100000, position_size=0.1):
    # Get historical data
    data = get_stock_data(symbol, start=start_date, end=end_date)
    data = calculate_technical_indicators(data)
    
    sentiment = analyze_sentiment(get_news(symbol))
    features, scaler = prepare_features(data, sentiment)
    
    sequence_length = 10
    results = []
    portfolio_value = initial_capital
    position = 0
    window = 252  # One year of trading days
    
    for i in range(window + sequence_length, len(features)):
        # Train on the past year
        train_features = features.iloc[i-window:i]
        model = train_model(train_features, sequence_length)
        
        # Prepare sequence for prediction
        pred_sequence = features.iloc[i-sequence_length:i].values
        pred_sequence = pred_sequence.reshape(1, sequence_length, pred_sequence.shape[1])
        
        # Get prediction
        prediction = (model.predict(pred_sequence, verbose=0)[0][0] > 0.5).astype(int)
        
        # Execute trades
        close_price = data.iloc[i]['Close']
        if prediction == 1:
            # Buy
            shares_to_buy = (portfolio_value * position_size) // close_price
            position += shares_to_buy
            portfolio_value -= shares_to_buy * close_price
        elif prediction == 0:
            # Sell
            portfolio_value += position * close_price
            position = 0
        
        # Record results
        total_value = portfolio_value + position * close_price
        results.append({
            'Date': data.index[i],
            'Close': close_price,
            'Prediction': prediction,
            'Position': position,
            'Portfolio_Value': portfolio_value,
            'Total_Value': total_value
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def calculate_metrics(results_df):
    print(results_df)
    results_df['Returns'] = results_df['Total_Value'].pct_change()
    total_return = (results_df['Total_Value'].iloc[-1] / results_df['Total_Value'].iloc[0]) - 1
    sharpe_ratio = results_df['Returns'].mean() / results_df['Returns'].std() * np.sqrt(252)  # Sharpe ratio is measure of risk vs reward
    max_drawdown = (results_df['Total_Value'] / results_df['Total_Value'].cummax() - 1).min()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    return total_return, sharpe_ratio, max_drawdown

def run_multi_stock_backtest(symbols, start_date, end_date):
    all_results = {}
    portfolio_value = pd.Series(index=pd.date_range(start=start_date, end=end_date))

    for symbol in symbols:
        print(f"\nRunning backtest for {symbol}")
        results_df = backtest(symbol, start_date, end_date)  # Note: just call backtest directly here
        all_results[symbol] = results_df
        
        # Align dates and add to portfolio value
        symbol_returns = results_df.set_index('Date')['Total_Value'].pct_change()
        portfolio_value = portfolio_value.add(symbol_returns, fill_value=0)

    # Calculate portfolio metrics
    portfolio_value = (1 + portfolio_value).cumprod()
    portfolio_df = pd.DataFrame({'Total_Value': portfolio_value})
    
    print("\nAggregate Portfolio Metrics:")
    calculate_metrics(portfolio_df)

    return all_results, portfolio_df
# Rest of the functions (calculate_metrics, plot_results, etc.) remain the same

if __name__ == "__main__":
    symbols = ['IBM', 'BA', 'GOOGL', 'AAPL']
    start_date = '2018-01-01'
    end_date = '2022-01-01'
    all_results, portfolio_df = run_multi_stock_backtest(symbols, start_date, end_date)