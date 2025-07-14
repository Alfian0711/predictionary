from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
from joblib import load
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import ta

app = Flask(__name__)

STOCK_LIST = {
    'BBCA.JK': 'Bank Central Asia Tbk',
    'BBRI.JK': 'Bank Rakyat Indonesia Tbk',
    'BMRI.JK': 'Bank Mandiri Tbk',
    'TLKM.JK': 'Telekomunikasi Indonesia Tbk',
    'ASII.JK': 'Astra International Tbk',
    'UNVR.JK': 'Unilever Indonesia Tbk',
    'GOTO.JK': 'GoTo Gojek Tokopedia Tbk',
    'BREN.JK': 'Barito Renewables Energy Tbk',
    'AMMN.JK': 'Amman Mineral Internasional Tbk',
    'BBNI.JK': 'Bank Negara Indonesia Tbk',
    'ITMG.JK': 'Indo Tambangraya Megah Tbk',
}


def get_stock_data(ticker, period='12y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


def prepare_features(data, selected_features=None, retain_original=False):
    df = data.copy()

    for window_size in [5, 20, 50, 95, 155, 230]:
        df[f'MA_{window_size}'] = df['Close'].rolling(window=window_size).mean()

    df['rolling_std_10'] = df['Close'].rolling(window=10).std()
    df['exp_moving_avg_50'] = df['Close'].ewm(span=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    for i in range(1, 30, 2):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    for col in df.columns:
        if col not in ['Date', 'Volume'] and not col.endswith('_normalized'):
            df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])

    df.bfill(inplace=True)

    if selected_features is not None and len(selected_features) > 0:
        available_cols = df.columns.tolist()
        valid_selected = [f for f in selected_features if f in available_cols]
        if 'Date' in df.columns:
            valid_selected.append('Date')
        features_df = df[valid_selected].copy()
    else:
        features_df = df.copy()

    return (features_df, df if retain_original else None, scaler)



def make_predictions(model, scaler, last_features_row, selected_features, start_date, actual_close_value=None, days=30):
    predictions = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    current_features = last_features_row[selected_features].values.flatten()

    predicted_all = []

    for _ in range(days):
        X_pred_df = pd.DataFrame([current_features], columns=selected_features)
        pred_scaled = model.predict(X_pred_df)[0]
        pred_close = scaler.inverse_transform([[pred_scaled]])[0][0]
        predicted_all.append(pred_close)

        current_date += timedelta(days=1)
        while current_date.weekday() > 4:
            current_date += timedelta(days=1)

        predictions.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'Open': pred_close,
            'High': pred_close,
            'Low': pred_close,
            'Close': pred_close,
            'Volume': 0.0,
            'Predicted': True
        })

    actual = [actual_close_value] * len(predicted_all)
    rmse = np.sqrt(mean_squared_error(actual, predicted_all))
    mae = mean_absolute_error(actual, predicted_all)
    r2 = r2_score(actual, predicted_all)

    return predictions, rmse, mae, r2

def format_stock_data_for_chart(data):
    return [
        {
            'x': row['Date'],
            'y': [float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])]
        }
        for _, row in data.iterrows()
    ]


@app.route('/')
def index():
    return render_template('index.html', stocks=STOCK_LIST)


@app.route('/all_stocks')
def all_stocks():
    try:
        all_stocks_data = {ticker: get_stock_data(ticker, '3mo') for ticker in STOCK_LIST}
        all_stocks_data = {k: v for k, v in all_stocks_data.items() if v is not None and not v.empty}

        if not all_stocks_data:
            return jsonify({'error': 'Tidak ada data saham'}), 400

        comparison_data = {}
        performance_data = {}

        for ticker, data in all_stocks_data.items():
            first_price = data['Close'].iloc[0]
            last_price = data['Close'].iloc[-1]
            perf_pct = ((last_price - first_price) / first_price) * 100
            ohlc = [
                {
                    'x': row['Date'],
                    'y': [float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])]
                } for _, row in data.iterrows()
            ]

            performance_data[ticker] = {
                'name': STOCK_LIST[ticker],
                'ticker': ticker,
                'last_price': float(last_price),
                'performance_pct': float(perf_pct),
                'ohlc': ohlc[-20:]
            }

            dates = data['Date'].tolist()
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
    ticker = request.form.get('ticker')
    days = int(request.form.get('days', 30))

    if not ticker:
        return jsonify({'error': 'Ticker tidak valid'}), 400

    data = get_stock_data(ticker)
    if data is None:
        return jsonify({'error': f'Tidak dapat mengambil data untuk {ticker}'}), 400

    try:
        model_bundle = load('model/svr_model2.joblib')
        model = model_bundle['model']
        scaler = model_bundle['scaler']
        selected_features = model_bundle['features']
    except Exception as e:
        return jsonify({'error': f'Gagal memuat model: {e}'}), 500

    df_features, original_df, _ = prepare_features(data, selected_features=selected_features, retain_original=True)
    if df_features.empty:
        return jsonify({'error': 'Data fitur kosong setelah preprocessing.'}), 500

    missing = [f for f in selected_features if f not in df_features.columns]
    if missing:
        return jsonify({'error': f'Fitur hilang di data: {missing}'}), 500

    last_row = df_features.iloc[-1:]
    last_date = last_row['Date'].values[0]
    actual_close_value = original_df['Close'].iloc[-1]

    predictions, rmse, mae, r2 = make_predictions(
        model=model,
        scaler=scaler,
        last_features_row=last_row,
        selected_features=selected_features,
        start_date=last_date,
        actual_close_value=actual_close_value,
        days=days
    )
    
    if not predictions:
        return jsonify({'error': 'Tidak ada prediksi yang dihasilkan.'}), 500
    
    historical_ohlc = format_stock_data_for_chart(data)

    
    chart_data = {
        'historical': historical_ohlc,
        'predictions': [
            {
                'x': p['Date'],
                'y': [p['Open'], p['High'], p['Low'], p['Close']],
                'predicted': p['Predicted']
            }
            for p in predictions
        ]
    }

     # Format response
    stock_info = {
        'ticker': ticker,
        'name': STOCK_LIST.get(ticker, 'Unknown'),
        'last_price': float(data['Close'].iloc[-1]),
        'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
        'change_percent': float((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100),
        'prediction': predictions,
        'chart_data': chart_data,
         'rmse': rmse,
        'mae': mae,
        'r_squared': r2
    }

    
    return jsonify(stock_info)


if __name__ == '__main__':
    app.run(debug=True)
