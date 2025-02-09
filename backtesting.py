import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from data_collection import get_stock_data, get_news, analyze_sentiment
from utils import calculate_technical_indicators
import risk_manager

"""
def prepare_features(data, sentiment): # Preparing data for machine learning model. RSI and MACD now used as input.
    features = data[['RSI', 'MACD', 'Signal_Line']].copy() # copy these columns into features data frame
    features['Sentiment'] = sentiment
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int) # creation of array with 1 and 0 saying whether the closing value of the stock was up (1) or down (0), compared to the previous day.
    
    
    
    return features.dropna()
"""

def prepare_features(data): # some extended list of features.
    features = pd.DataFrame()
    
    # Technical indicators
    features['RSI'] = data['RSI']
    features['MACD'] = data['MACD']
    features['Returns_21d'] = data['Close'].pct_change(21)
    features['BB_Position'] = (data['Close'].iloc[:, 0] - data['LowerBB']) / (data['UpperBB'] - data['LowerBB'])
    features['ATR'] = data['ATR']
    features['Volume_Ratio'] = data['Volume'].iloc[:, 0] / data['Volume_SMA']
    
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return features.dropna()

def train_model(features): # RandomForest model 
    X = features.drop('Target', axis=1) # matrix with each column representing a feature with for every row  a date with the entry of that feature for that specifc date 
    y = features['Target'] # array with 1 and 0 saying whether the closing value of the stock was up (1) or down (0), compared to the previous day.

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,  # Reduce overfitting
        max_features='sqrt',  
        random_state=1
    )# This is the class of scikit learn package yielding RF classifier. Here a forest of 100 trees is designed.
                                                                      # Random_state controls the randomness of the model. This ensure you get same results every time you run model with same data. 
                                                                      # Randomnes in random forest returns in randomly subsetting of features at each node and in randomly selecting subset of training data (bootstrapping) to train the tree
                                                                      # Bagging uses bootstrapping as a means to extract the sample of the data. Bootstrapping is a way of sampling.

    weights = np.linspace(1, 1, len(y)) # this is a weight that tells model to focus on recent data.

    model.fit(X.values, y, sample_weight=weights) # Fit the data(X=features, y=close up or down) to the RandomForest model. 
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
    # sentiment = analyze_sentiment(get_news(symbol))
    
    features = prepare_features(data)
    

    # Initialize results
    results = []
    portfolio_value = initial_capital
    position = 0
    window = 252  # One year of trading days

    #----------------------------#
    #        Risk Manager        #
    #----------------------------#

    rm = risk_manager.RiskManager(n_regimes=2)
    rm.fit(data.iloc[:window])# Fits HMM on the first year of data to learn what different regimes look like.

    rm.analyze_regime_characteristics()

    #----------------------------#
    for i in range(window, len(features)):

        # Train on the past year
        train_features = features.iloc[i-window:i]
        model = train_model(train_features)
        

        # features at day i (iloc), dropping target (closing price). 
        # values changes dataframe to numpy array and reshape reshapes the 1 D array to 2D array consisting of 1 row with n columns (the features). This is needed as input for machine learning model
        # Data point is ran through all the trees and decision is made on aggregate output of all trees. 


     #   prediction = model.predict(features.iloc[i].drop('Target').values.reshape(1, -1))[0] # this gives binary prediction 1 or 0
        # Get probability estimates for next day
        prob_up = model.predict_proba(features.iloc[i].drop('Target').values.reshape(1, -1))[0][1] # Gives probability of prediction instead of 1 and 0 outcome. Use for trade position
        # Execute trades based on probability
        close_price = data.iloc[i]['Close']


        #-----------------------#
        #      Risk manager     #
        #-----------------------#

        # Risk manager: predict regime. no need to refit model. It has been trained/fitted above.
        current_data = data[i - 60:i] # use recent data (60 days) to predict regime.
        current_regime = rm.predict_regime(current_data)

        trading_params  = rm.adjust_trading_params(current_regime, prob_up)
        position_scalar = trading_params['position_scalar']
        long_threshold  = trading_params['long_threshold']
        short_threshold = trading_params['short_threshold']
        should_trade    = trading_params['should_trade']


        if should_trade:
            if prob_up > long_threshold:  # Strong bullish signal
                position_multiplier = (prob_up - 0.5) * 10  # Scales with confidence
                target_long_position = (portfolio_value * position_size * position_multiplier * position_scalar) // close_price

                shares_to_trade = target_long_position - position
                position += shares_to_trade
                portfolio_value -= shares_to_trade * close_price

            elif prob_up < short_threshold:  # Strong bearish signal
                position_multiplier = (0.5 - prob_up) * 10  # Scales with confidence
                target_short_position = -(portfolio_value * position_size * position_multiplier * position_scalar) // close_price

                shares_to_trade = target_short_position - position
                position += shares_to_trade
                portfolio_value -= shares_to_trade * close_price
                
        # Record results
        total_value = portfolio_value + position * close_price
        results.append({
            'Date': data.index[i],
            'Close': close_price,
            'Probability_Up': prob_up,
            'Position': position,
            'Portfolio_Value': portfolio_value,
            'Total_Value': total_value,
            'Current_Regime': current_regime,  # Add regime information
            'Position_Scalar': position_scalar,
            'Should_Trade': should_trade
        })

    
    results_df = pd.DataFrame(results)
    return results_df

def calculate_metrics(results_df):
    # Ensure returns are calculated correctly

    print(results_df['Total_Value'])
    results_df['Returns'] = results_df['Total_Value'].pct_change().dropna()
    
    # Check if we have enough data
    if len(results_df) < 2:
        print("Insufficient data for metrics calculation")
        return 0, 0, 0
    
    try:
        total_return = (results_df['Total_Value'].iloc[-1] / results_df['Total_Value'].iloc[0]) - 1
        
        # Handle potential division by zero or other calculation issues
        returns = results_df['Returns'].dropna()
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        max_drawdown = (results_df['Total_Value'] / results_df['Total_Value'].cummax() - 1).min()
        
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        return total_return, sharpe_ratio, max_drawdown
    
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        return 0, 0, 0


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
  #  calculate_metrics(portfolio_df)

    return all_results, portfolio_df

def plot_results(results_df):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))  # Added fourth subplot
    
    # Plot portfolio value
    ax1.plot(results_df['Date'], results_df['Total_Value'])
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value')
    
    # Plot price and positions
    ax2.plot(results_df['Date'], results_df['Close'])
    
    # Color points based on probability
    for idx, row in results_df.iterrows():
        if row['Probability_Up'] > 0.7:
            color = 'green'
            size = (row['Probability_Up'] - 0.5) * 500
            ax2.scatter(row['Date'], row['Close'], color=color, s=size, alpha=0.5)
        elif row['Probability_Up'] < 0.3:
            color = 'red'
            size = (0.5 - row['Probability_Up']) * 500
            ax2.scatter(row['Date'], row['Close'], color=color, s=size, alpha=0.5)
    
    ax2.set_title('Price and Trading Signals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    
    # Plot probability over time
    ax3.plot(results_df['Date'], results_df['Probability_Up'])
    ax3.axhline(y=0.7, color='g', linestyle='--', alpha=0.3)
    ax3.axhline(y=0.3, color='r', linestyle='--', alpha=0.3)
    ax3.set_title('Prediction Probabilities')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Probability of Price Increase')
    
    # Add position/inventory plot
    ax4.plot(results_df['Date'], results_df['Position'])
    ax4.set_title('Position Size Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Number of Shares')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Add zero line
    
    plt.tight_layout()
    plt.show()

def run_backtest(symbol, start_date, end_date):
    print(f"Running backtest for {symbol} from {start_date} to {end_date}")
    results_df = backtest(symbol, start_date, end_date)
   # total_return, sharpe_ratio, max_drawdown = calculate_metrics(results_df)
    plot_results(results_df)
 #   return results_df, total_return, sharpe_ratio, max_drawdown

if __name__ == "__main__":
    symbols = ['TSLA', 'ACRE'] 
    start_date = '2020-01-01'
    end_date = '2024-01-01'
  #  all_results, portfolio_df,  sharpe_ratio, max_drawdown = 
    for symbol in symbols:
        run_backtest(symbol, start_date, end_date)

  #  for symbol, results_df in all_results.items():
   #     print(f"\nMetrics for {symbol}:")
  #      calculate_metrics(results_df)
