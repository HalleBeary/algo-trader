from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_collection import get_stock_data, get_news, analyze_sentiment
from utils import calculate_technical_indicators_lstm

print(pd.__file__)

def prepare_sequences(features, sequence_length=10): 
    """Prepare sequences for LSTM""" # Sequence length is how many past days LSTM looks at to predict future value. Core property of LSTMs, time memory
    X, y = [], []
    for i in range(len(features) - sequence_length): # chop the input data in blocks of length sequence_length. This value is used in LSTM as time dimension.
        X.append(features[i:(i + sequence_length)])  # shape: each row has 10x4 matrix of 10 entries of all features.
        y.append(features['Target'].iloc[i + sequence_length])

    return np.array(X), np.array(y)

def prepare_features(data, sentiment): # prepare input features and assign them value between 0 and 1
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

def create_lstm_model(input_shape): # initialize lstm model. 
    model = Sequential([         # sequential type to capture time dependencies   
        Input(shape=input_shape),  # input layer
        LSTM(50, return_sequences=True),  # LSTM layer with 50 neurons
        Dropout(0.2), # Randomly "drops" (turns off) 20% of neurons each batch to prevent overfitting.
        LSTM(50, return_sequences=False), # another LSTM layer with 50 neurons, this time only outputs (but still considers!! 10 timesteps) state of last timestep (return_sequence=False) as output (Dense layer) needs single vector
        Dropout(0.2),
        Dense(1, activation='sigmoid') # Takes the LSTM's output and converts it to a buy/sell probability
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
        X, y, # X features, y target buy/sell
        epochs=50, # 50 cycles using the same data
        batch_size=32, # process 32 samples at a time, then update weights, then take next 32"
        validation_split=0.2, # use 80 % for training, and 20 % for validation
        verbose=0
    )
    return model

def backtest(symbol, start_date, end_date, initial_capital=100000, position_size=0.1):
    # Get historical data
    data = get_stock_data(symbol, start=start_date, end=end_date)
    data = calculate_technical_indicators_lstm(data)
    
    sentiment = analyze_sentiment(get_news(symbol))
    features, scaler = prepare_features(data, sentiment)
    
    sequence_length = 10
    results = []
    portfolio_value = initial_capital
    position = 0
    window = 252  # One year of trading days
    
    # Create model once outside the loop
    n_features = len(features.columns)
    model = create_lstm_model((sequence_length, n_features)) # Here the neural network is created. 
    
    for i in range(window + sequence_length, len(features)): # Here the LSTM NN is trained and backtested
        # Get training data
        train_features = features.iloc[i-window:i]
        X_train, y_train = prepare_sequences(train_features, sequence_length)
        
        # Train model (reuse same model instance)
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Prepare sequence for prediction
        pred_sequence = features.iloc[i-sequence_length:i].values # take 10 days of features
        pred_sequence = pred_sequence.reshape(1, sequence_length, pred_sequence.shape[1]) #reshape to 1, 10, num_features as LSTM expects 3D input (batch_size, sequence_length, num_features)
                                                                                          # 3D is characteristic to LSTMs. Sequence_length is again core to LSTM as it captures the temporal dependency.  
        # Get prediction
        prediction = (model.predict(pred_sequence, verbose=0)[0][0] > 0.5).astype(int)
        
        # Rest of the trading logic remains the same
        close_price = data.iloc[i]['Close']
        if prediction == 1:
            shares_to_buy = (portfolio_value * position_size) // close_price
            position += shares_to_buy
            portfolio_value -= shares_to_buy * close_price
        elif prediction == 0:
            portfolio_value += position * close_price
            position = 0
        
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