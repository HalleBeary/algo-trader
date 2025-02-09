import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator
class RiskManager:
    def __init__(self, n_regimes=2):
        self.n_regimes = n_regimes
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100
        )
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.features = None

    def prepare_risk_features(self, data):
        features = pd.DataFrame()
        
        # Keep existing features
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['volume_std'] = data['Volume'].rolling(20).std()
        features['atr'] = data['ATR']
        features['bb_width'] = (data['UpperBB'] - data['LowerBB']) / data['Close'].iloc[:,0]
        
        # Add drawdown and vol of vol
        rolling_max = data['Close'].rolling(window=20).max()
        features['drawdown'] = (data['Close'] - rolling_max) / rolling_max
        features['vol_of_vol'] = features['volatility'].rolling(window=20).std()
        
        # adx
        adx_indicator = ADXIndicator(data['High'].iloc[:,0], data['Low'].iloc[:,0], data['Close'].iloc[:,0])
        features['adx'] = adx_indicator.adx()
        
        return features.dropna()

    def fit(self, data):
        """Fit HMM model to historical data"""
        features = self.prepare_risk_features(data)
        scaled_features = self.scaler.fit_transform(features)
        self.features = self.prepare_risk_features(data) 
        
        # Fit HMM
        self.hmm_model.fit(scaled_features)
        
        # Get regime sequence
        self.regime_labels = self.hmm_model.predict(scaled_features)
        
        # Calculate regime properties
        self.analyze_regimes(features, self.regime_labels)

    def predict_regime(self, data):
        """Predict current regime"""
        features = self.prepare_risk_features(data)
        scaled_features = self.scaler.transform(features)
        current_regime = self.hmm_model.predict(scaled_features)[-1]
        return current_regime

    def analyze_regimes(self, features, regime_labels):
        """Analyze properties of each regime"""
        self.regime_properties = {}
        for regime in range(self.n_regimes):
            regime_data = features[regime_labels == regime]
            self.regime_properties[regime] = {
                'volatility': regime_data['volatility'].mean(),
                'volume': regime_data['volume_ma'].mean(),
                'atr': regime_data['atr'].mean(),
                'drawdown': regime_data['drawdown'].mean(),
                'vol_of_vol': regime_data['vol_of_vol'].mean(),
                'adx': regime_data['adx'].mean()
            }

    def get_position_adjustments(self, current_regime):
        regime_props = self.regime_properties[current_regime]
        
        # More conservative in high risk regimes
        high_risk = (regime_props['volatility'] > np.median([r['volatility'] for r in self.regime_properties.values()]) or
                    regime_props['vol_of_vol'] > np.median([r['vol_of_vol'] for r in self.regime_properties.values()]) or
                    regime_props['drawdown'] < np.median([r['drawdown'] for r in self.regime_properties.values()]))
        
        position_scalar = 0.5 if high_risk else 1.0
        threshold_adjustment = 0.05 if high_risk else 0
                
        return {
            'position_scalar': position_scalar,
            'threshold_adjustment': threshold_adjustment
        }

    def adjust_trading_params(self, current_regime, prob_up):
        """Adjust trading parameters based on regime"""
        adjustments = self.get_position_adjustments(current_regime)
      

        # Adjust thresholds
        long_threshold = 0.7 + adjustments['threshold_adjustment']
        short_threshold = 0.3 - adjustments['threshold_adjustment']
        
        # Determine if we should trade
        should_trade = True
        if self.regime_properties[current_regime]['volatility'] > np.percentile(
            [r['volatility'] for r in self.regime_properties.values()], 90):
            should_trade = False  # Too volatile
            
        return {
            'position_scalar': adjustments['position_scalar'],
            'long_threshold': long_threshold,
            'short_threshold': short_threshold,
            'should_trade': should_trade
        }
    
    def analyze_regime_characteristics(self):
            
        feature_list = ['returns', 'drawdown', 'volatility', 'vol_of_vol', 'atr', 'adx', 'bb_width']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()  # Flatten the 2D array of axes
        
        # Colors for each regime
        colors = ['blue', 'red']
        labels = ['Regime 0', 'Regime 1']
        
        # Plot each feature
        for i, feature in enumerate(feature_list):
            ax = axes[i]
            
            # Plot histogram for each regime
            for regime in range(self.n_regimes):
                mask = self.regime_labels == regime
                data = self.features[mask][feature]
                
                # Calculate and print statistics
                feature_mean = np.mean(data)
                feature_std = np.std(data)
                print(f"\n{feature} - {labels[regime]}:")
                print(f"  Mean: {feature_mean:.4f}")
                print(f"  Std:  {feature_std:.4f}")
                
                # Plot histogram with transparency
                ax.hist(data, bins=30, alpha=0.5, color=colors[regime], 
                    label=f'{labels[regime]} (μ={feature_mean:.3f})')
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print overall regime statistics
        for regime in range(self.n_regimes):
            mask = self.regime_labels == regime
            print(f"\n{labels[regime]} overall statistics:")
            print(f"Percentage of time in this regime: {np.mean(mask) * 100:.2f}%")