from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings

app = Flask(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

STOCK_LIST = {
    'BBCA.JK': 'Bank Central Asia Tbk',
    'UNVR.JK': 'Unilever Indonesia Tbk',
    'ITMG.JK': 'Indo Tambangraya Megah Tbk',
}

STOCK_NAMES = {
    'BBCA.JK': 'Bank Central Asia',
    'UNVR.JK': 'Unilever Corporation',
    'ITMG.JK': 'Indo Tambangraya Megah'
}

# Configuration
SHORT_TERM_CONFIG = {
    'sequence_length': 30,
    'prediction_days': [1, 2, 3],
    'data_years': 20,
}


class CacheManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_model_path(self, model_name):
        return os.path.join(self.cache_dir, f"{model_name}.h5")

    def load_model(self, model_name):
        path = self.get_model_path(model_name)
        if os.path.exists(path):
            try:
                return load_model(path)
            except Exception as e:
                print(f"Error loading cached model: {e}")
        return None

    def load_object(self, name):
        path = os.path.join(self.cache_dir, f"{name}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached object: {e}")
        return None


cache = CacheManager()


def get_stock_data(ticker, period='12y'):
    """Get stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None

        data['Ticker'] = ticker
        return data
    except Exception as e:
        print(f"Error getting stock data: {e}")
        return None


def add_short_term_features(data):
    """
    Add features optimized for short-term prediction (1-3 days)
    Same as in the notebook
    """
    df = data.copy()

    # === Short-term momentum features ===
    # RSI with short periods
    for period in [5, 9, 14]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        avg_loss = avg_loss.replace(0, 0.00001)
        rs = avg_gain / avg_loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # === Moving Averages for short-term trends ===
    for period in [3, 5, 7, 10, 15, 20]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

    # === MACD optimized for short-term ===
    exp5 = df['Close'].ewm(span=5, adjust=False).mean()
    exp13 = df['Close'].ewm(span=13, adjust=False).mean()
    df['MACD_Short'] = exp5 - exp13
    df['MACD_Short_Signal'] = df['MACD_Short'].ewm(span=5, adjust=False).mean()
    df['MACD_Short_Hist'] = df['MACD_Short'] - df['MACD_Short_Signal']

    # === Short-term volatility ===
    for period in [3, 5, 7, 10]:
        df[f'Volatility_{period}d'] = df['Close'].rolling(window=period).std()
        df[f'Price_Change_{period}d'] = df['Close'].pct_change(period)

    # === Bollinger Bands for short-term ===
    for period in [10, 15, 20]:
        df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'BB_Std_{period}'] = df['Close'].rolling(window=period).std()
        df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + \
            (df[f'BB_Std_{period}'] * 2)
        df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - \
            (df[f'BB_Std_{period}'] * 2)
        df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (
            df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])

    # === Volume indicators ===
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']
    df['Volume_Ratio_10'] = df['Volume'] / df['Volume_MA_10']

    # === Price momentum ===
    for period in [1, 2, 3, 5, 7]:
        df[f'Price_Momentum_{period}d'] = df['Close'] / \
            df['Close'].shift(period) - 1
        df[f'High_Low_Ratio_{period}d'] = (
            df['High'] - df['Low']) / df['Close'].shift(period)

    # === Intraday features ===
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Close_Ratio'] = df['High'] / df['Close']
    df['Low_Close_Ratio'] = df['Low'] / df['Close']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
    df['Upper_Shadow'] = (
        df['High'] - np.maximum(df['Close'], df['Open'])) / df['Close']
    df['Lower_Shadow'] = (np.minimum(
        df['Close'], df['Open']) - df['Low']) / df['Close']

    # === Lag features relevant for short-term ===
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

    # === Rolling statistics ===
    for period in [5, 10, 15]:
        df[f'Close_Rolling_Mean_{period}'] = df['Close'].rolling(
            window=period).mean()
        df[f'Close_Rolling_Std_{period}'] = df['Close'].rolling(
            window=period).std()
        df[f'Close_Rolling_Min_{period}'] = df['Close'].rolling(
            window=period).min()
        df[f'Close_Rolling_Max_{period}'] = df['Close'].rolling(
            window=period).max()

    # === Calendar effects ===
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Is_Month_End'] = df.index.is_month_end.astype(int)
    df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)

    # Cyclic encoding for calendar features
    df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Drop NaN values
    df.dropna(inplace=True)

    return df


def optimize_normalization(data, scalers_dict, ticker):
    """
    Apply the same normalization used during training
    """
    df = data.copy()

    if ticker not in scalers_dict:
        return None

    scalers = scalers_dict[ticker]

    # Apply saved scalers
    for feature, scaler in scalers.items():
        if feature in df.columns:
            values = df[feature].values.reshape(-1, 1)
            df[feature] = scaler.transform(values).flatten()

    return df


def make_lstm_predictions(ticker, days=30):
    """
    Make predictions using trained LSTM model
    Compatible with the template format
    """
    try:
        # Load model and scalers
        model_name = f"short_term_lstm_{ticker.replace('.', '_')}"
        model = cache.load_model(model_name)

        # Load final results which contains scalers
        final_results = cache.load_object('final_results')

        if model is None:
            return None, f"Model tidak ditemukan untuk {ticker}. Silakan training model terlebih dahulu."

        if final_results is None or ticker not in final_results:
            return None, f"Scalers tidak ditemukan untuk {ticker}. Silakan training model terlebih dahulu."

        scalers = final_results[ticker]['scalers']

        # Get fresh data
        data = get_stock_data(ticker, period='2y')
        if data is None:
            return None, "Gagal mengambil data saham"

        # Apply feature engineering
        enhanced_data = add_short_term_features(data)

        # Apply normalization using saved scalers
        normalized_data = optimize_normalization(
            enhanced_data, {ticker: scalers}, ticker)
        if normalized_data is None:
            return None, "Gagal melakukan normalisasi data"

        # Create sequence for prediction
        features = [col for col in normalized_data.columns if col != 'Ticker']
        values = normalized_data[features].values

        sequence_length = SHORT_TERM_CONFIG['sequence_length']
        if len(values) < sequence_length:
            return None, f"Membutuhkan minimal {sequence_length} hari data historis"

        # Use last sequence for prediction
        X = values[-sequence_length:].reshape(1, sequence_length, -1)

        # Make LSTM prediction (for 1, 2, 3 days)
        pred_normalized = model.predict(X, verbose=0)

        # Inverse transform predictions
        close_scaler = scalers['Close']
        lstm_predictions = close_scaler.inverse_transform(
            pred_normalized).flatten()

        # Get current price for reference
        current_price = enhanced_data['Close'].iloc[-1]
        base_date = enhanced_data.index[-1]

        # Create prediction data for the number of days requested
        predictions = []

        # First, add LSTM predictions (1, 2, 3 days)
        for i, days_ahead in enumerate(SHORT_TERM_CONFIG['prediction_days']):
            pred_date = base_date + timedelta(days=days_ahead)
            # Skip weekends
            while pred_date.weekday() > 4:
                pred_date += timedelta(days=1)

            pred_price = lstm_predictions[i]

            # Create OHLC data (simplified - using same price for all)
            predictions.append({
                'Date': pred_date.strftime('%Y-%m-%d'),
                'Open': float(pred_price * 0.999),  # Slightly lower open
                'High': float(pred_price * 1.002),  # Slightly higher high
                'Low': float(pred_price * 0.998),   # Slightly lower low
                'Close': float(pred_price),
                'Volume': 1000000.0,  # Mock volume
                'Predicted': True
            })

        # If more days requested, extend with simple trend continuation
        if days > 3:
            # Calculate trend from LSTM predictions
            # Average daily change
            trend = (lstm_predictions[2] - lstm_predictions[0]) / 2

            last_pred_price = lstm_predictions[2]
            last_pred_date = base_date + timedelta(days=3)

            for day in range(4, days + 1):
                pred_date = last_pred_date + timedelta(days=day-3)
                # Skip weekends
                while pred_date.weekday() > 4:
                    pred_date += timedelta(days=1)

                # Apply trend with some noise
                noise_factor = np.random.normal(0, 0.01)  # 1% random noise
                pred_price = last_pred_price + \
                    (trend * (day - 3)) * (1 + noise_factor)

                predictions.append({
                    'Date': pred_date.strftime('%Y-%m-%d'),
                    'Open': float(pred_price * 0.999),
                    'High': float(pred_price * 1.002),
                    'Low': float(pred_price * 0.998),
                    'Close': float(pred_price),
                    'Volume': 1000000.0,
                    'Predicted': True
                })

        # Calculate metrics (simplified for extended predictions)
        # Use only LSTM predictions for metrics
        pred_prices = [p['Close'] for p in predictions[:3]]
        actual_prices = [current_price] * len(pred_prices)

        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        mae = mean_absolute_error(actual_prices, pred_prices)

        # R-squared calculation (simplified)
        ss_res = sum((actual_prices[i] - pred_prices[i])
                     ** 2 for i in range(len(actual_prices)))
        ss_tot = sum((actual_prices[i] - np.mean(actual_prices))
                     ** 2 for i in range(len(actual_prices)))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return predictions, rmse, mae, r2, None

    except Exception as e:
        return None, None, None, None, f"Error prediksi: {str(e)}"


def format_stock_data_for_chart(data):
    """Format stock data for chart display"""
    result = []
    for index, row in data.iterrows():
        result.append({
            'x': index.strftime('%Y-%m-%d'),
            'y': [float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])]
        })
    return result


def format_prediction_data_for_chart(predictions):
    """Format prediction data for chart display"""
    result = []
    for pred in predictions:
        result.append({
            'x': pred['Date'],
            'y': [pred['Open'], pred['High'], pred['Low'], pred['Close']],
            'predicted': pred['Predicted']
        })
    return result


@app.route('/')
def index():
    return render_template('index.html', stocks=STOCK_LIST)


@app.route('/all_stocks')
def all_stocks():
    try:
        all_stocks_data = {}

        for ticker in STOCK_LIST:
            data = get_stock_data(ticker, '3mo')
            if data is not None and not data.empty:
                all_stocks_data[ticker] = data

        if not all_stocks_data:
            return jsonify({'error': 'Tidak ada data saham tersedia'}), 400

        comparison_data = {}
        performance_data = {}

        for ticker, data in all_stocks_data.items():
            first_price = data['Close'].iloc[0]
            last_price = data['Close'].iloc[-1]
            perf_pct = ((last_price - first_price) / first_price) * 100

            # Format for chart
            ohlc_data = []
            for index, row in data.iterrows():
                ohlc_data.append({
                    'x': index.strftime('%Y-%m-%d'),
                    'y': [float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])]
                })

            performance_data[ticker] = {
                'name': STOCK_LIST[ticker],
                'ticker': ticker,
                'last_price': float(last_price),
                'performance_pct': float(perf_pct),
                'ohlc': ohlc_data[-20:]  # Last 20 days
            }

            # Normalized comparison data
            dates = [index.strftime('%Y-%m-%d') for index in data.index]
            prices = data['Close'].tolist()
            base_price = prices[0]
            norm_prices = [price / base_price * 100 for price in prices]

            comparison_data[ticker] = {
                'name': ticker.replace('.JK', ''),
                'data': [{'x': d, 'y': p} for d, p in zip(dates, norm_prices)]
            }

        return jsonify({
            'comparison': comparison_data,
            'stocks': performance_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form.get('ticker')
        days = int(request.form.get('days', 30))

        if not ticker or ticker not in STOCK_LIST:
            return jsonify({'error': 'Ticker tidak valid'}), 400

        if days < 1 or days > 90:
            return jsonify({'error': 'Jumlah hari harus antara 1-90'}), 400

        # Get historical data for chart
        historical_data = get_stock_data(ticker, '1y')
        if historical_data is None:
            return jsonify({'error': f'Tidak dapat mengambil data untuk {ticker}'}), 400

        # Make LSTM predictions
        predictions, rmse, mae, r2, error = make_lstm_predictions(ticker, days)
        if error:
            return jsonify({'error': error}), 500

        # Format historical data for chart
        historical_ohlc = format_stock_data_for_chart(historical_data)

        # Format prediction data for chart
        prediction_chart_data = format_prediction_data_for_chart(predictions)

        # Get current stock info
        current_price = float(historical_data['Close'].iloc[-1])
        prev_price = float(historical_data['Close'].iloc[-2])
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100

        # Format response compatible with the template
        stock_info = {
            'ticker': ticker,
            'name': STOCK_LIST.get(ticker, 'Unknown'),
            'last_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'prediction': predictions,
            'chart_data': {
                'historical': historical_ohlc[-60:],  # Last 60 days
                'predictions': prediction_chart_data
            },
            'rmse': float(rmse) if rmse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'r_squared': float(r2) if r2 is not None else None
        }

        return jsonify(stock_info)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Terjadi kesalahan saat membuat prediksi: {str(e)}'}), 500


@app.route('/model_info')
def model_info():
    """Get information about available models"""
    model_status = {}

    for ticker in STOCK_LIST:
        model_name = f"short_term_lstm_{ticker.replace('.', '_')}"
        model = cache.load_model(model_name)

        model_status[ticker] = {
            'name': STOCK_LIST[ticker],
            'model_available': model is not None,
            'model_type': 'LSTM Neural Network',
            'prediction_days': SHORT_TERM_CONFIG['prediction_days'],
            'sequence_length': SHORT_TERM_CONFIG['sequence_length']
        }

    return jsonify({
        'models': model_status,
        'config': SHORT_TERM_CONFIG
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    available_models = []
    for ticker in STOCK_LIST:
        model_name = f"short_term_lstm_{ticker.replace('.', '_')}"
        if cache.load_model(model_name) is not None:
            available_models.append(ticker)

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(available_models),
        'available_models': available_models,
        'total_models': len(STOCK_LIST)
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint tidak ditemukan'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Terjadi kesalahan server internal'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 LSTM Stock Prediction Flask Application")
    print("=" * 60)
    print(f"📊 Available stocks: {list(STOCK_LIST.keys())}")
    print("🔍 Checking model availability...")

    # Check if models are available
    available_models = []
    missing_models = []

    for ticker in STOCK_LIST:
        model_name = f"short_term_lstm_{ticker.replace('.', '_')}"
        if cache.load_model(model_name) is not None:
            available_models.append(ticker)
        else:
            missing_models.append(ticker)

    if available_models:
        print(f"✅ Models tersedia untuk: {available_models}")

    if missing_models:
        print(f"⚠️  Models belum tersedia untuk: {missing_models}")
        print(
            "💡 Silakan jalankan notebook training terlebih dahulu untuk model yang hilang")

    if not available_models:
        print("🚨 WARNING: Tidak ada model yang tersedia!")
        print("📝 Silakan jalankan notebook Stock_Prediction_LSTM.ipynb terlebih dahulu")

    # Check if final_results.pkl exists
    final_results = cache.load_object('final_results')
    if final_results is None:
        print("⚠️  final_results.pkl tidak ditemukan - diperlukan untuk scalers")
    else:
        print("✅ Scalers tersedia")

    print("=" * 60)
    print("🌐 Starting Flask application...")
    print("📱 Access: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
