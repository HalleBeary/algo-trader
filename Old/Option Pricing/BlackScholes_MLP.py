import pandas as pd
import numpy as np
from ib_insync import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.stats import norm
import datetime as dt

class DataCollector:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
    
    def get_option_data(self, symbol='SPX', exchange='CBOE', duration='1 Y'): 
        """Collect option data from IBKR"""
        # Create base index contract
        contract = Index(symbol, exchange)
        
        data_list = []
        
        # Get option chains
        chains = self.ib.reqSecDefOptParams(
            contract.symbol, '', contract.secType, contract.conId)
        
        for chain in chains:
            for expiry in chain.expirations[:5]:  # First 5 expirations
                for strike in chain.strikes:
                    # Create option contract
                    option = Option(symbol, expiry, strike, 'C', exchange)
                    
                    try:
                        # Get historical data
                        bars = self.ib.reqHistoricalData(
                            option,
                            endDateTime='',
                            durationStr=duration,
                            barSizeSetting='1 day',
                            whatToShow='TRADES',
                            useRTH=True
                        )
                        
                        if bars:
                            for bar in bars:
                                data_list.append({
                                    'date': bar.date,
                                    'strike': strike,
                                    'expiry': expiry,
                                    'price': bar.close,
                                    'volume': bar.volume,
                                    'underlying_price': None,  # Will update later
                                    'iv': None  # Will update later
                                })
                    
                    except Exception as e:
                        print(f"Error collecting data for strike {strike}, expiry {expiry}: {e}")
        
        df = pd.DataFrame(data_list)
        return self._enrich_data(df, symbol)
    
    def _enrich_data(self, df, symbol):
        """Add underlying price and other necessary data"""
        # Get underlying price history
        contract = Index(symbol, 'CBOE')
        underlying_data = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Create underlying price dictionary
        underlying_prices = {bar.date: bar.close for bar in underlying_data}
        
        # Update DataFrame
        df['underlying_price'] = df['date'].map(underlying_prices)
        
        # Calculate time to expiry in years
        df['T'] = df.apply(lambda row: (pd.to_datetime(row['expiry']) - 
                                      pd.to_datetime(row['date'])).days / 365, axis=1)
        
        # Add risk-free rate (you might want to get this from a proper source)
        df['r'] = 0.02  # Placeholder
        
        return df

class BlackScholes:
    @staticmethod # Staticmethod makes you call this fucntion with seperately initializing the class first. This shouldnt be done if the function uses variables
                    # class variables that are unique to every instance of the class. So for utility functions its fine. If you want to update something unique to an instance you shouldnt do it
    def calculate_price(S, K, T, r, sigma, option_type='call'):
        """Calculate BS price"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using binary search"""
        max_iter = 100
        precision = 0.00001
        sigma = 0.5  # Initial guess
        
        for i in range(max_iter):
            price_bs = BlackScholes.calculate_price(S, K, T, r, sigma, option_type)
            diff = price_bs - price
            
            if abs(diff) < precision:
                return sigma
            
            if diff > 0:
                sigma = sigma - sigma/2
            else:
                sigma = sigma + sigma/2
        
        return sigma

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def prepare_features(self, df):
        """Prepare features for the neural network"""
        # Calculate BS price and implied volatility
        df['iv'] = df.apply(lambda row: BlackScholes.implied_volatility(
            row['price'], row['underlying_price'], row['strike'],
            row['T'], row['r']), axis=1)
        
        df['bs_price'] = df.apply(lambda row: BlackScholes.calculate_price(
            row['underlying_price'], row['strike'], row['T'],
            row['r'], row['iv']), axis=1)
        
        # Create feature matrix
        X = np.column_stack([
            df['underlying_price'].values,
            df['strike'].values,
            df['T'].values,
            df['r'].values,
            df['iv'].values,
            df['bs_price'].values
        ])
        
        # Calculate residuals (target)
        y = df['price'].values - df['bs_price'].values
        
        return X, y, df['bs_price'].values
    
    def split_and_scale(self, X, y, test_size=0.2):
        """Split data and scale features"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False)  # Time series data
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

class ResidualModel:
    def __init__(self, input_dim=6):
        self.model = self._build_model(input_dim)
    
    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10,
                                               restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )

class Backtester:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def backtest(self, X_test, y_test, bs_prices_test):
        """Run backtest on test data"""
        # Get predictions
        residual_pred = self.model.predict(X_test)
        final_prices_pred = bs_prices_test + residual_pred.flatten()
        final_prices_true = bs_prices_test + y_test
        
        # Calculate metrics
        results = {
            'residual_mse': np.mean((y_test - residual_pred.flatten())**2),
            'price_mse': np.mean((final_prices_true - final_prices_pred)**2),
            'price_mae': np.mean(np.abs(final_prices_true - final_prices_pred)),
            'mape': np.mean(np.abs((final_prices_true - final_prices_pred) / 
                                 final_prices_true)) * 100
        }
        
        return results, final_prices_pred

def main():
    # Initialize components
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    
    # Collect data
    data = collector.get_option_data()
    
    # Prepare features
    X, y, bs_prices = preprocessor.prepare_features(data)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.split_and_scale(X, y)
    
    # Train model
    model = ResidualModel()
    history = model.train(X_train_scaled, y_train,
                         X_test_scaled, y_test)
    
    # Backtest
    backtester = Backtester(model.model, preprocessor)
    results, predictions = backtester.backtest(
        X_test_scaled, y_test,
        bs_prices[-len(X_test_scaled):])  # Use corresponding BS prices
    
    print("Backtest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()