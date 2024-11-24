# src/trading/execution.py
from ib_insync import *
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Optional
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import redis
from prometheus_client import Counter, Gauge, start_http_server
from data_collection import *

@dataclass
class TradeSignal:
    """Structure for trade signals"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    timestamp: datetime
    price: float
    size: int

@dataclass
class Position:
    """Structure for position tracking"""
    symbol: str
    size: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float

class RiskManager:
    """Handles position sizing and risk limits"""
    
    def __init__(self, config: Dict):
        self.max_position_size = config.get('max_position_size', 100)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)
        self.max_position_risk = config.get('max_position_risk', 0.01)
        self.max_drawdown = config.get('max_drawdown', 0.1)
        
        self.position_cache = redis.Redis(
            host='localhost',
            port=6379,
            db=1
        )
        
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, 
                              signal: TradeSignal,
                              portfolio_value: float,
                              volatility: float) -> int:
        """Calculate position size based on risk parameters"""
        try:
            # Position sizing based on volatility
            risk_amount = portfolio_value * self.max_position_risk
            price_risk = volatility * signal.price
            position_size = min(
                int(risk_amount / price_risk),
                self.max_position_size
            )
            
            # Check portfolio constraints
            current_exposure = self.get_total_exposure()
            if (current_exposure + position_size * signal.price) / portfolio_value > 0.5:
                position_size = int(
                    (0.5 * portfolio_value - current_exposure) / signal.price
                )
            
            return max(0, position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        positions = self.position_cache.hgetall('positions')
        return sum(float(pos['size']) * float(pos['price']) 
                  for pos in positions.values())

class TradingExecutor:
    """Handles trade execution and monitoring"""
    
    def __init__(self, config: Dict):
        self.ib = IB()
        self.risk_manager = RiskManager(config.get('risk_config', {}))
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.trade_counter = Counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'action']
        )
        self.position_gauge = Gauge(
            'position_size',
            'Current position size',
            ['symbol']
        )
        self.pnl_gauge = Gauge(
            'unrealized_pnl',
            'Unrealized PnL',
            ['symbol']
        )
        
        # Start metrics server
        start_http_server(8000)

    def connect(self):
        """Connect to IB with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.ib.connect('127.0.0.1', 7497, clientId=1) # 7497 is paper, 7496 is real.
                self.logger.info("Successfully connected to IB")
                return
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError("Failed to connect to IB after max retries")

    def execute_trade(self, signal: TradeSignal):
        """Execute trade with error handling and logging"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
            
            # Get current market data
            contract = Stock(signal.symbol, 'SMART', 'USD')
            market_data = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data
            
            # Calculate position size
            portfolio_value = self._get_portfolio_value()
            volatility = self._calculate_volatility(signal.symbol)
            size = self.risk_manager.calculate_position_size(
                signal, portfolio_value, volatility
            )
            
            if size == 0:
                self.logger.info(f"Zero size calculated for {signal.symbol}")
                return
                
            # Create and place order
            order = MarketOrder(
                action=signal.action,
                totalQuantity=size,
                tif='IOC',  # Immediate or cancel
                transmit=True
            )
            
            # Add stop loss
            stop_price = self._calculate_stop_price(signal)
            stop_order = StopOrder(
                action='SELL' if signal.action == 'BUY' else 'BUY',
                totalQuantity=size,
                stopPrice=stop_price
            )
            
            # Place orders
            trade = self.ib.placeOrder(contract, order)
            stop_trade = self.ib.placeOrder(contract, stop_order)
            
            # Wait for fill
            while not trade.isDone():
                self.ib.sleep(0.1)
            
            # Update metrics
            self.trade_counter.labels(
                symbol=signal.symbol,
                action=signal.action
            ).inc()
            
            # Log trade
            self._log_trade(trade, signal)
            
            # Update position tracking
            self._update_position(signal.symbol, trade)
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            self._handle_error(e, signal)

    def run_trading_cycle(self, models: Dict, data_fetcher):
        """Run one complete trading cycle"""
        try:
            for symbol, model in models.items():
                # Get latest data
                data = data_fetcher.get_complete_data(symbol, period="2d")
                
                # Generate prediction
                features = self._prepare_features(data)
                prediction = model.predict(features)
                confidence = model.predict_proba(features).max()
                
                # Create signal
                signal = TradeSignal(
                    symbol=symbol,
                    action='BUY' if prediction == 1 else 'SELL',
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=data['market_data']['Close'][-1],
                    size=0  # Will be calculated by risk manager
                )
                
                # Execute if confidence is high enough
                if confidence > 0.7:
                    self.execute_trade(signal)
                
                # Update metrics
                self._update_metrics(symbol)
                
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {str(e)}")
            self._handle_cycle_error(e)

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trading signal"""
        return (
            signal.confidence > 0.7 and
            self._is_market_open() and
            not self._is_market_stressed(signal.symbol)
        )

    def _calculate_stop_price(self, signal: TradeSignal) -> float:
        """Calculate stop loss price"""
        volatility = self._calculate_volatility(signal.symbol)
        if signal.action == 'BUY':
            return signal.price * (1 - 2 * volatility)
        return signal.price * (1 + 2 * volatility)

    def _update_metrics(self, symbol: str):
        """Update monitoring metrics"""
        position = self._get_position(symbol)
        if position:
            self.position_gauge.labels(symbol=symbol).set(position.size)
            self.pnl_gauge.labels(symbol=symbol).set(position.unrealized_pnl)

    def _handle_error(self, error: Exception, signal: TradeSignal):
        """Handle trading errors"""
        self.logger.error(f"Error executing {signal.action} for {signal.symbol}: {str(error)}")
        # Add notification system here
        
def main():
    # Configuration
    config = {
        'risk_config': {
            'max_position_size': 100,
            'max_portfolio_risk': 0.02,
            'max_position_risk': 0.01,
            'max_drawdown': 0.1
        },
        'trading_config': {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'cycle_interval': 3600  # 1 hour
        }
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    trader = TradingExecutor(config)
    data_fetcher = MarketDataFetcher()  # From previous example
    
    try:
        # Connect to IB
        trader.connect()
        
        # Main trading loop
        while True:
            trader.run_trading_cycle(models, data_fetcher)
            time.sleep(config['trading_config']['cycle_interval'])
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading system...")
    finally:
        trader.ib.disconnect()

if __name__ == "__main__":
    main()