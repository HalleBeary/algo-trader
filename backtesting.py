# src/backtesting/backtester.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

@dataclass
class BacktestResults:
    """Structure for backtest results"""
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    precision: float
    recall: float
    f1_score: float
    predictions_df: pd.DataFrame
    portfolio_history: pd.DataFrame

class FeatureGenerator:
    """Handles feature engineering for the ML model"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_definitions = {
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd,
            'Volatility': self._calculate_volatility,
            'MOM': self._calculate_momentum
        }

    def create_features(self, data: pd.DataFrame, sentiment: float) -> pd.DataFrame:
        """Generate all features for the model"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Technical indicators
            for name, func in self.feature_definitions.items():
                features[name] = func(data)
            
            # Add sentiment
            features['Sentiment'] = sentiment
            
            # Create target
            features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error in feature generation: {str(e)}")
            raise

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        return data['Close'].pct_change().rolling(window=window).std()

    def _calculate_momentum(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """Calculate momentum"""
        return data['Close'].pct_change(window)

class MLModel:
    """Handles model training and prediction"""
    
    def __init__(self, model_params: Dict = None):
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        }
        self.model = None
        self.logger = logging.getLogger(__name__)

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the model and log metrics"""
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.model_params)
                
                # Train model
                self.model = RandomForestClassifier(**self.model_params)
                self.model.fit(features, target)
                
                # Log feature importance
                importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': self.model.feature_importances_
                })
                mlflow.log_dict(importance.to_dict(), 'feature_importance.json')
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(features)

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, 
                 data_fetcher,
                 initial_capital: float = 100000,
                 position_size: float = 0.1):
        self.data_fetcher = data_fetcher
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.feature_generator = FeatureGenerator()
        self.model = MLModel()
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, 
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    window: int = 252) -> BacktestResults:
        """Run backtest for single symbol"""
        try:
            # Fetch data
            data = self.data_fetcher.get_complete_data(
                symbol, start=start_date, end=end_date
            )
            
            # Generate features
            features = self.feature_generator.create_features(
                data['market_data'],
                data['avg_sentiment']
            )
            
            # Initialize results tracking
            results = []
            portfolio_value = self.initial_capital
            position = 0
            
            # Rolling window backtest
            for i in range(window, len(features)):
                # Train on window
                train_features = features.iloc[i-window:i]
                X_train = train_features.drop('Target', axis=1)
                y_train = train_features['Target']
                
                self.model.train(X_train, y_train)
                
                # Predict
                prediction = self.model.predict(
                    features.iloc[i:i+1].drop('Target', axis=1)
                )[0]
                
                # Execute trades
                close_price = data['market_data'].iloc[i]['Close']
                portfolio_value, position = self._execute_trades(
                    prediction, close_price, portfolio_value, position
                )
                
                # Record results
                results.append(self._record_results(
                    date=data['market_data'].index[i],
                    close_price=close_price,
                    prediction=prediction,
                    position=position,
                    portfolio_value=portfolio_value
                ))
            
            # Calculate metrics
            results_df = pd.DataFrame(results)
            metrics = self._calculate_metrics(results_df)
            
            return BacktestResults(
                symbol=symbol,
                **metrics,
                predictions_df=results_df,
                portfolio_history=self._create_portfolio_history(results_df)
            )
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol}: {str(e)}")
            raise

    def _execute_trades(self, 
                       prediction: int,
                       close_price: float,
                       portfolio_value: float,
                       position: int) -> Tuple[float, int]:
        """Execute trades based on prediction"""
        if prediction == 1:  # Buy signal
            shares_to_buy = (portfolio_value * self.position_size) // close_price
            position += shares_to_buy
            portfolio_value -= shares_to_buy * close_price
        elif prediction == 0:  # Sell signal
            portfolio_value += position * close_price
            position = 0
            
        return portfolio_value, position

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        returns = results_df['Total_Value'].pct_change()
        total_return = (results_df['Total_Value'].iloc[-1] / 
                       results_df['Total_Value'].iloc[0]) - 1
        
        return {
            'total_return': total_return,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (results_df['Total_Value'] / 
                           results_df['Total_Value'].cummax() - 1).min(),
            'precision': precision_score(results_df['Target'], 
                                      results_df['Prediction']),
            'recall': recall_score(results_df['Target'], 
                                 results_df['Prediction']),
            'f1_score': f1_score(results_df['Target'], 
                               results_df['Prediction'])
        }

    def plot_results(self, results: BacktestResults) -> None:
        """Plot backtest results using plotly"""
        fig = go.Figure()
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=results.portfolio_history.index,
            y=results.portfolio_history['Total_Value'],
            name='Portfolio Value'
        ))
        
        # Buy/Sell signals
        buy_signals = results.predictions_df[results.predictions_df['Prediction'] == 1]
        sell_signals = results.predictions_df[results.predictions_df['Prediction'] == 0]
        
        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
        
        fig.update_layout(
            title=f'Backtest Results for {results.symbol}',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        fig.show()