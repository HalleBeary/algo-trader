import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

def get_stock_data(symbol, period=None, start=None, end=None):
    if period:
        data = yf.download(symbol, period=period)
    elif start and end:
        data = yf.download(symbol, start=start, end=end)
    else:
        raise ValueError("Either 'period' or both 'start' and 'end' must be provided")
    return data

def get_news(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_table = soup.find(id='news-table')
    return [row.a.text for row in news_table.findAll('tr')]

def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in texts:
        tb_sentiment = TextBlob(text).sentiment.polarity
        vader_sentiment = analyzer.polarity_scores(text)['compound']
        avg_sentiment = (tb_sentiment + vader_sentiment) / 2
        sentiments.append(avg_sentiment)
    return sum(sentiments) / len(sentiments) if sentiments else 0

def get_sentiment(symbol):
    news = get_news(symbol)
    return analyze_sentiment(news)