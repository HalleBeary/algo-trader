import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def calculate_technical_indicators(data):
    # Existing RSI calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Existing MACD calculation
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Existing Bollinger Bands calculation
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['StdDev20'] = data['Close'].rolling(window=20).std()
    data['UpperBB'] = data['SMA20'] + (data['StdDev20'] * 2)
    data['LowerBB'] = data['SMA20'] - (data['StdDev20'] * 2)

    # Existing Stochastic Oscillator calculation
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = (data['Close'] - low_14) * 100 / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()

    # ATR calculation


    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # Moving Averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # Volume indicators
    data['OBV'] = (data['Volume'] * ((data['Close'] - data['Close'].shift()) > 0).astype(int) * 2 - 1).cumsum()
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Price_Volume'] = data['Close'] * data['Volume']
    
    return data

def calculate_returns(data):
    data['Returns'] = data['Close'].pct_change()
    return data

def calculate_volatility(data, window=20):
    data['Volatility'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)
    return data

def add_date_features(data):
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter
    return data

def preprocess_data(data):
    data = calculate_technical_indicators(data)
    data = calculate_returns(data)
    data = calculate_volatility(data)
    data = add_date_features(data)
    return data.dropna()

def prepare_features(data, sentiment):
    features = data[['RSI', 'MACD', 'Signal_Line', '%K', '%D', 'Volatility']].copy()
    features['Sentiment'] = sentiment
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return features.dropna()

def train_model(features):
    X = features.drop('Target', axis=1)
    y = features['Target']
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X, y)
    return model

def predict(model, features):
    return model.predict(features.drop('Target', axis=1))

def evaluate_model(model, features):
    X = features.drop('Target', axis=1)
    y = features['Target']
    return model.score(X, y)