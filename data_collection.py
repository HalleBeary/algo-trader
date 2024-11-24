# src/data/market_data.py
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import redis
from ratelimit import limits, sleep_and_retry
import backoff
from dataclasses import dataclass

@dataclass
class NewsItem:
    """Structure for news data"""
    timestamp: datetime
    headline: str
    source: str
    url: str
    sentiment: float

class MarketDataFetcher:
    """Handles all market data collection with error handling and caching"""
    
    def __init__(self, cache_config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = redis.Redis(**cache_config) if cache_config else None
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, yf.YFinanceError))
    @sleep_and_retry
    @limits(calls=10, period=1)  # Rate limiting
    def get_stock_data(self, 
                      symbol: str, 
                      period: Optional[str] = None, 
                      start: Optional[str] = None, 
                      end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data with caching and error handling
        """
        try:
            # Check cache first
            cache_key = f"stock_data:{symbol}:{period or f'{start}_{end}'}"
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)

            # Fetch from yfinance
            if period:
                data = yf.download(symbol, period=period)
            elif start and end:
                data = yf.download(symbol, start=start, end=end)
            else:
                raise ValueError("Either 'period' or both 'start' and 'end' must be provided")

            # Validate data
            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            
            # Basic data cleaning
            data = self._clean_market_data(data)

            # Cache the result
            if self.cache:
                self.cache.setex(
                    cache_key,
                    timedelta(minutes=15),  # Cache for 15 minutes
                    data.to_json()
                )

            return data

        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            raise

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
    @sleep_and_retry
    @limits(calls=2, period=1)  # More conservative rate limiting for news
    def get_news(self, symbol: str) -> List[NewsItem]:
        """
        Fetch news with structured output and error handling
        """
        try:
            cache_key = f"news:{symbol}"
            if self.cache:
                cached_news = self.cache.get(cache_key)
                if cached_news:
                    return [NewsItem(**item) for item in eval(cached_news)]

            url = f"https://finviz.com/quote.ashx?t={symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find(id='news-table')
            
            if not news_table:
                raise ValueError(f"No news table found for {symbol}")

            news_items = []
            for row in news_table.findAll('tr'):
                try:
                    timestamp = row.td.text
                    headline = row.a.text
                    url = row.a['href']
                    
                    news_item = NewsItem(
                        timestamp=pd.to_datetime(timestamp),
                        headline=headline,
                        source=url.split('/')[2],
                        url=url,
                        sentiment=self._analyze_single_sentiment(headline)
                    )
                    news_items.append(news_item)
                except Exception as e:
                    self.logger.warning(f"Error parsing news item: {str(e)}")
                    continue

            # Cache the results
            if self.cache:
                self.cache.setex(
                    cache_key,
                    timedelta(minutes=5),  # Cache for 5 minutes
                    str([item.__dict__ for item in news_items])
                )

            return news_items

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            raise

    def _analyze_single_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of single text with multiple analyzers
        """
        try:
            # TextBlob sentiment
            tb_sentiment = TextBlob(text).sentiment.polarity
            
            # VADER sentiment
            vader_sentiment = self.vader.polarity_scores(text)['compound']
            
            # Average the sentiments
            return (tb_sentiment + vader_sentiment) / 2
            
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {str(e)}")
            return 0.0

    def _clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean market data
        """
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Remove outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = self._remove_outliers(df[col])
            
        # Add basic derived columns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        
        return df

    def _remove_outliers(self, series: pd.Series, n_std: int = 3) -> pd.Series:
        """
        Remove outliers based on standard deviation
        """
        mean = series.mean()
        std = series.std()
        return series.clip(lower=mean - n_std*std, upper=mean + n_std*std)

class MarketDataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self, symbols: List[str]):
        self.fetcher = MarketDataFetcher()
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)

    def get_complete_data(self, 
                         period: Optional[str] = None,
                         start: Optional[str] = None,
                         end: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get both market data and sentiment for multiple symbols
        """
        with ThreadPoolExecutor() as executor:
            # Fetch market data and news in parallel
            market_data_futures = {
                symbol: executor.submit(
                    self.fetcher.get_stock_data, 
                    symbol, 
                    period, 
                    start, 
                    end
                )
                for symbol in self.symbols
            }
            
            news_futures = {
                symbol: executor.submit(self.fetcher.get_news, symbol)
                for symbol in self.symbols
            }

            # Collect results
            results = {}
            for symbol in self.symbols:
                try:
                    market_data = market_data_futures[symbol].result()
                    news = news_futures[symbol].result()
                    
                    results[symbol] = {
                        'market_data': market_data,
                        'news': news,
                        'avg_sentiment': np.mean([item.sentiment for item in news])
                    }
                except Exception as e:
                    self.logger.error(f"Error processing data for {symbol}: {str(e)}")
                    continue

            return results