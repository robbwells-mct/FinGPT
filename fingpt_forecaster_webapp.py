#!/usr/bin/env python3
"""
FinGPT Forecaster - Lightweight Web App
A simplified web interface for stock market analysis and prediction
without requiring heavy LLM models or GPU resources.
"""

import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None, f"No data found for symbol {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def perform_technical_analysis(data):
    """Perform comprehensive technical analysis"""
    df = data.copy()
    
    # Calculate technical indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Generate trading signals
    df['Signal'] = 0
    df.loc[(df['SMA_10'] > df['SMA_30']) & (df['RSI'] < 70), 'Signal'] = 1  # Buy
    df.loc[(df['SMA_10'] < df['SMA_30']) & (df['RSI'] > 30), 'Signal'] = -1  # Sell
    
    return df

def generate_prediction(data, symbol):
    """Generate prediction based on technical analysis"""
    recent_data = data.tail(30)
    
    # Current indicators
    current_price = data['Close'].iloc[-1]
    current_rsi = recent_data['RSI'].iloc[-1]
    current_macd = recent_data['MACD'].iloc[-1]
    current_signal = recent_data['Signal'].iloc[-1]
    recent_trend = recent_data['Close'].pct_change(5).iloc[-1]  # 5-day trend
    volatility = recent_data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized vol
    
    # Price relative to Bollinger Bands
    bb_position = (current_price - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])
    
    # Volume analysis
    volume_trend = recent_data['Volume_Ratio'].mean()
    
    # Prediction logic
    bullish_signals = 0
    bearish_signals = 0
    
    # RSI signals
    if current_rsi < 30:
        bullish_signals += 2
    elif current_rsi > 70:
        bearish_signals += 2
    elif 40 < current_rsi < 60:
        bullish_signals += 0.5
    
    # MACD signals
    if current_macd > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Moving average signals
    if current_signal == 1:
        bullish_signals += 1.5
    elif current_signal == -1:
        bearish_signals += 1.5
    
    # Bollinger Band signals
    if bb_position < 0.2:
        bullish_signals += 1
    elif bb_position > 0.8:
        bearish_signals += 1
    
    # Volume confirmation
    if volume_trend > 1.2:
        bullish_signals += 0.5
    
    # Final prediction
    total_signals = bullish_signals + bearish_signals
    if total_signals > 0:
        bull_prob = bullish_signals / total_signals
        bear_prob = bearish_signals / total_signals
    else:
        bull_prob = bear_prob = 0.5
    
    if bull_prob > 0.6:
        prediction = "🟢 BULLISH"
        confidence = bull_prob
    elif bear_prob > 0.6:
        prediction = "🔴 BEARISH"  
        confidence = bear_prob
    else:
        prediction = "🟡 NEUTRAL"
        confidence = max(bull_prob, bear_prob)
    
    return {
        'prediction': prediction,
        'confidence': confidence * 100,
        'current_price': current_price,
        'current_rsi': current_rsi,
        'current_macd': current_macd,
        'recent_trend': recent_trend * 100,
        'volatility': volatility,
        'bb_position': bb_position * 100,
        'volume_trend': volume_trend,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals
    }

def create_price_chart(data, symbol):
    """Create interactive price chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_10'],
        name='SMA 10', line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_30'],
        name='SMA 30', line=dict(color='red', width=1)
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Upper'],
        name='BB Upper', line=dict(color='gray', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Lower'],
        name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ))
    
    # Buy/Sell signals
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers', name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'],
            mode='markers', name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f'{symbol} - Price Chart with Technical Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=600,
        showlegend=True
    )
    
    return fig

def create_indicators_chart(data, symbol):
    """Create chart for technical indicators"""
    fig = go.Figure()
    
    # RSI subplot
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        name='RSI', line=dict(color='purple')
    ))
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title=f'{symbol} - RSI Indicator',
        yaxis_title='RSI',
        xaxis_title='Date',
        height=300,
        yaxis_range=[0, 100]
    )
    
    return fig

def predict_30_days(data, symbol):
    """Advanced 30-day price prediction using ensemble methods"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        
        # Create features
        df = data.copy()
        
        # Technical indicators for features
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_30'] = df['Close'].rolling(30).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], _, _ = calculate_macd(df['Close'])
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price momentum features
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        
        # Prepare ML data
        features = ['SMA_10', 'SMA_30', 'RSI', 'MACD', 'Volatility', 'Volume_Ratio', 'Return_5d', 'Return_10d']
        df['Target'] = df['Close'].shift(-30) / df['Close'] - 1  # 30-day future return
        
        # Clean data
        df_clean = df.dropna()
        if len(df_clean) < 50:
            return None
        
        X = df_clean[features]
        y = df_clean['Target']
        
        # Train models on most recent data (last 80% for training)
        split_idx = int(len(df_clean) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_recent = X.iloc[[-1]]  # Most recent data for prediction
        
        # Model 1: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_recent)[0]
        
        # Model 2: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_recent)[0]
        
        # Model 3: Trend extrapolation
        trend_pred = df['Close'].pct_change().tail(30).mean() * 30
        
        # Ensemble prediction
        ensemble_pred = (rf_pred * 0.5 + lr_pred * 0.3 + trend_pred * 0.2)
        
        current_price = data['Close'].iloc[-1]
        predicted_price = current_price * (1 + ensemble_pred)
        
        # Confidence interval using historical volatility
        vol_30d = df['Close'].pct_change().std() * np.sqrt(30)
        conf_lower = predicted_price * (1 - 1.96 * vol_30d)
        conf_upper = predicted_price * (1 + 1.96 * vol_30d)
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_pct': ensemble_pred * 100,
            'conf_lower': conf_lower,
            'conf_upper': conf_upper,
            'volatility': vol_30d * 100
        }
        
    except Exception as e:
        print(f"30-day prediction error: {e}")
        return None

def analyze_stock(symbol, period):
    """Main function to analyze stock and return results"""
    # Validate symbol
    symbol = symbol.upper().strip()
    if not symbol:
        return None, None, None, "Please enter a valid stock symbol"
    
    # Fetch data
    data, error = get_stock_data(symbol, period)
    if error:
        return None, None, None, error
    
    if len(data) < 50:
        return None, None, None, f"Insufficient data for analysis (only {len(data)} days available)"
    
    # Perform analysis
    enhanced_data = perform_technical_analysis(data)
    prediction = generate_prediction(enhanced_data, symbol)
    
    # 30-day prediction
    prediction_30d = predict_30_days(data, symbol)
    
    # Create charts
    price_chart = create_price_chart(enhanced_data.tail(60), symbol)  # Last 60 days
    indicators_chart = create_indicators_chart(enhanced_data.tail(60), symbol)
    
    # Create summary report
    prediction_30d_section = ""
    if prediction_30d:
        signal_30d = "🟢 BULLISH" if prediction_30d['change_pct'] > 5 else "🔴 BEARISH" if prediction_30d['change_pct'] < -5 else "🟡 NEUTRAL"
        prediction_30d_section = f"""
## 🔮 30-Day Price Forecast
- **Predicted Price:** ${prediction_30d['predicted_price']:.2f}
- **Expected Change:** {prediction_30d['change_pct']:+.1f}%
- **Confidence Range:** ${prediction_30d['conf_lower']:.2f} - ${prediction_30d['conf_upper']:.2f}
- **30-Day Signal:** {signal_30d}
- **Forecast Volatility:** ±{prediction_30d['volatility']:.1f}%
"""
    
    report = f"""
# 📊 FinGPT Analysis Report for {symbol}

## 🎯 Short-Term Prediction
**{prediction['prediction']}** with **{prediction['confidence']:.1f}%** confidence

## 💰 Current Metrics
- **Current Price:** ${prediction['current_price']:.2f}
- **5-Day Trend:** {prediction['recent_trend']:+.2f}%
- **Volatility:** {prediction['volatility']:.1f}% (annualized)

{prediction_30d_section}

## 📈 Technical Indicators
- **RSI:** {prediction['current_rsi']:.1f} {'(Oversold)' if prediction['current_rsi'] < 30 else '(Overbought)' if prediction['current_rsi'] > 70 else '(Neutral)'}
- **MACD:** {prediction['current_macd']:+.4f}
- **Bollinger Band Position:** {prediction['bb_position']:.1f}% {'(Near Lower)' if prediction['bb_position'] < 20 else '(Near Upper)' if prediction['bb_position'] > 80 else '(Middle Range)'}
- **Volume Trend:** {prediction['volume_trend']:.2f}x average {'(High)' if prediction['volume_trend'] > 1.5 else '(Low)' if prediction['volume_trend'] < 0.8 else '(Normal)'}

## 🔍 Signal Analysis
- **Bullish Signals:** {prediction['bullish_signals']:.1f}
- **Bearish Signals:** {prediction['bearish_signals']:.1f}

## ⚠️ Disclaimer
This analysis is for educational purposes only and should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.
"""
    
    return price_chart, indicators_chart, report

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="FinGPT Forecaster - Lightweight Edition", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🚀 FinGPT Forecaster - Lightweight Edition</h1>
            <p style="font-size: 18px;">AI-Powered Stock Market Analysis & Prediction</p>
            <p>Enter a stock symbol to get comprehensive technical analysis and predictions</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_input = gr.Textbox(
                    label="Stock Symbol",
                    placeholder="Enter symbol (e.g., AAPL, GOOGL, TSLA)",
                    value="AAPL"
                )
                period_input = gr.Dropdown(
                    choices=["3mo", "6mo", "1y", "2y", "5y"],
                    label="Analysis Period",
                    value="1y"
                )
                analyze_btn = gr.Button("🔍 Analyze Stock", variant="primary", size="lg")
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 10px;">
                    <h3>💡 Popular Symbols:</h3>
                    <p><strong>Tech:</strong> AAPL, GOOGL, MSFT, TSLA, NVDA</p>
                    <p><strong>Finance:</strong> JPM, BAC, GS, V, MA</p>
                    <p><strong>Healthcare:</strong> JNJ, PFE, UNH, ABBV</p>
                    <p><strong>Consumer:</strong> AMZN, WMT, KO, PEP, DIS</p>
                </div>
                """)
        
        with gr.Row():
            with gr.Column():
                price_chart_output = gr.Plot(label="Price Chart & Technical Analysis")
                indicators_chart_output = gr.Plot(label="Technical Indicators")
                report_output = gr.Markdown(label="Analysis Report")
        
        # Event handler
        analyze_btn.click(
            fn=analyze_stock,
            inputs=[symbol_input, period_input],
            outputs=[price_chart_output, indicators_chart_output, report_output]
        )
        
        # Auto-analyze on load
        interface.load(
            fn=lambda: analyze_stock("AAPL", "1y"),
            outputs=[price_chart_output, indicators_chart_output, report_output]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting FinGPT Forecaster Web App...")
    print("=" * 50)
    
    # Create and launch the interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Create public link
        show_error=True,
        quiet=False
    )
