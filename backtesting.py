import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from data_collection import get_stock_data, get_news, analyze_sentiment
from utils import calculate_technical_indicators

def prepare_features(data, sentiment): # Preparing data for machine learning model. RSI and MACD now used as input.
    features = data[['RSI', 'MACD', 'Signal_Line']].copy() # copy these columns into features data frame
    features['Sentiment'] = sentiment
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int) # creation of array with 1 and 0 saying whether the closing value of the stock was up (1) or down (0), compared to the previous day.
    return features.dropna()

def train_model(features): # RandomForest model 
    X = features.drop('Target', axis=1) # matrix with each column representing a feature with for every row  a date with the entry of that feature for that specifc date 

    y = features['Target'] # array with 1 and 0 saying whether the closing value of the stock was up (1) or down (0), compared to the previous day.

    model = RandomForestClassifier(n_estimators=100, random_state=1) # This is the class of scikit learn package yielding RF classifier. Here a forest of 100 trees is designed.
                                                                      # Random_state controls the randomness of the model. This ensure you get same results every time you run model with same data. 
                                                                      # Randomnes in random forest returns in randomly subsetting of features at each node and in randomly selecting subset of training data (bootstrapping) to train the tree
                                                                      # Bagging uses bootstrapping as a means to extract the sample of the data. Bootstrapping is a way of sampling.
    model.fit(X.values, y) # Fit the data(X=features, y=close up or down) to the RandomForest model. 
    return model

# Random forest classifier's learning process:
# Forest is ensemble of decision trees. Each tree has a random sample of traning data
# Random sampling of data is called bagging and is happening to increase model diversity, reduces overfitting and imrproves stability ( If models are diverse, they're likely to make different mistakes)
# At each split in a tree only 2 of 4 features (RSI, MACD, Signal Line (MACD), sentiment) are considered
# Then for different hypothetical cases say, RSI > 70 or Sentiment > 0.5 it determines when to buy or sell.
#=

def backtest(symbol, start_date, end_date, initial_capital=100000, position_size=0.1):
    # Get historical data
    data = get_stock_data(symbol, start=start_date, end=end_date)
    data = calculate_technical_indicators(data)
    
    # For simplicity, we'll use a constant sentiment. In a real scenario, you'd need historical sentiment data.
    sentiment = analyze_sentiment(get_news(symbol))
    
    features = prepare_features(data, sentiment)
    
    # Initialize results
    results = []
    portfolio_value = initial_capital
    position = 0
    window = 252  # One year of trading days
    

    for i in range(window, len(features)):
        # Train on the past year
        train_features = features.iloc[i-window:i]
        model = train_model(train_features)
        

        # Predict the next day after training period.

        prediction = model.predict(features.iloc[i].drop('Target').values.reshape(1, -1))[0] # features at day i (iloc), dropping target (closing price). 
                                                                                             # values changes dataframe to numpy array and reshape reshapes the 1 D array to 2D array consisting of 1 row with n columns (the features). This is needed as input for machine learning model
                    # Data point is ran through all the trees and decision is made on aggregate output of all trees. 
                    # the ..[0] ensures output is a 1 or a 0. Buy or sell. else output would be numpy array.
        

        # Execute trades
        close_price = data.iloc[i]['Close']
        if prediction == 1: #and position <= 0: #! Here we could implement a better risk strategy. Now only buying if position is 0, which means no stocks are in portfolio yet. Or letting go of that condition all together.
            # Buy
            shares_to_buy = (portfolio_value * position_size) // close_price
            position += shares_to_buy
            portfolio_value -= shares_to_buy * close_price
        elif prediction == 0: #and position > 0:
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
    sharpe_ratio = results_df['Returns'].mean() / results_df['Returns'].std() * np.sqrt(252) # Sharpe ratio is measure of risk vs reward. Spread of return is measure of risk in Modern Portfolio Theory
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
        results_df = run_backtest(symbol, start_date, end_date)[0]
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

def plot_results(results_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot portfolio value
    ax1.plot(results_df['Date'], results_df['Total_Value'])
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value')
    
    # Plot buy/sell signals
    ax2.plot(results_df['Date'], results_df['Close'])
    buy_signals = results_df[results_df['Prediction'] == 1]
    sell_signals = results_df[results_df['Prediction'] == 0]
    ax2.scatter(buy_signals['Date'], buy_signals['Close'], color='green', label='Buy', marker='^')
    ax2.scatter(sell_signals['Date'], sell_signals['Close'], color='red', label='Sell', marker='v')
    ax2.set_title('Buy/Sell Signals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stock Price')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def run_backtest(symbol, start_date, end_date):
    print(f"Running backtest for {symbol} from {start_date} to {end_date}")
    results_df = backtest(symbol, start_date, end_date)
    total_return, sharpe_ratio, max_drawdown = calculate_metrics(results_df)
    plot_results(results_df)
    return results_df, total_return, sharpe_ratio, max_drawdown

if __name__ == "__main__":
    symbols = ['IBM', 'BA', 'GOOGL', 'AAPL']  # Add or remove stocks as needed
    start_date = '2018-01-01'
    end_date = '2022-01-01'
    all_results, portfolio_df = run_multi_stock_backtest(symbols, start_date, end_date)

    for symbol, results_df in all_results.items():
        print(f"\nMetrics for {symbol}:")
        calculate_metrics(results_df)

    # Plot the aggregate portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['Total_Value'])
    plt.title('Aggregate Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.show()