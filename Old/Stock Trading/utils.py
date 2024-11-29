import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def calculate_technical_indicators(data):
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['UpperBB'] = data['SMA20'] + (data['Close'].rolling(window=20).std() * 2)
    data['LowerBB'] = data['SMA20'] - (data['Close'].rolling(window=20).std() * 2)
    
    # Calculate Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = (data['Close'] - low_14) * 100 / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    return data

def calculate_technical_indicators_lstm(data):
    # For RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # For MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data.dropna()

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