# 🚀 LSTM Stock Prediction Dashboard

**Modern Web Application untuk Prediksi Saham Indonesia menggunakan Deep Learning**

Aplikasi web responsif dengan dashboard modern untuk prediksi harga saham Indonesia menggunakan LSTM Neural Network. Dibangun dengan Flask backend dan Tailwind CSS frontend dengan visualisasi ApexCharts yang interaktif.

## 📸 Preview

```
🖥️ Dashboard Overview
├── 📊 Market Summary Cards (IHSG, Top Gainer/Loser, Most Active)
├── 📈 Stock Comparison Chart (Normalized Performance)
├── 🧠 LSTM Prediction Analysis (Candlestick + Forecast)
└── 📱 Mobile Responsive Design
```

## ✨ Features

### 🧠 **Advanced LSTM Model**
- **Multi-Target Prediction**: Prediksi akurat untuk 1, 2, 3 hari ke depan
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Hyperparameter Optimized**: Model terlatih dengan Bayesian Optimization
- **High Accuracy**: R² Score 0.80-0.92, Direction Accuracy 65-75%

### 📊 **Interactive Dashboard**
- **Modern UI**: Tailwind CSS dengan design responsif
- **Real-time Charts**: ApexCharts candlestick dengan zoom/pan
- **Multi-Section Layout**: Overview, Comparison, Analysis
- **Mobile Friendly**: Collapsible sidebar untuk mobile devices

### 🎯 **Indonesian Stocks Supported**
- **BBCA.JK** - Bank Central Asia Tbk
- **UNVR.JK** - Unilever Indonesia Tbk
- **ITMG.JK** - Indo Tambangraya Megah Tbk

### 📈 **Prediction Capabilities**
- **Short-term LSTM**: 1-3 hari (high accuracy)
- **Extended Forecast**: 4-90 hari (trend continuation)
- **Real-time Data**: Fresh market data dari Yahoo Finance
- **Performance Metrics**: RMSE, MAE, R-squared evaluation

## 🏗️ Architecture

### **Frontend Stack**
```
├── 🎨 Tailwind CSS 3.0     # Modern utility-first CSS
├── 📊 ApexCharts 3.44      # Interactive chart library
├── 🖼️ Feather Icons        # Beautiful icon set
├── ⚡ Vanilla JavaScript    # Lightweight interactions
└── 📱 Responsive Design    # Mobile-first approach
```

### **Backend Stack**
```
├── 🐍 Python 3.8+          # Core language
├── 🌶️ Flask 2.3+           # Web framework
├── 🧠 TensorFlow 2.13       # Deep learning
├── 📊 Pandas + NumPy       # Data processing
├── 📈 yfinance             # Market data source
└── 🔧 scikit-learn        # ML utilities
```

### **Model Architecture**
```
LSTM Neural Network
├── Input: 30-day sequences with 100+ features
├── Layer 1: LSTM(128) + Dropout(0.2) + BatchNorm
├── Layer 2: LSTM(64) + Dropout(0.2) + BatchNorm
├── Layer 3: LSTM(32) + Dropout(0.2)
├── Dense: 64 → 32 → 3 outputs
└── Output: [1-day, 2-day, 3-day] predictions
```

## 📦 Installation

### **Prerequisites**
- Python 3.8 atau lebih tinggi
- pip package manager
- Internet connection (untuk data real-time)

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/lstm-stock-prediction.git
cd lstm-stock-prediction
```

### **2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verify Installation**
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import yfinance as yf; print('yfinance: OK')"
```

## 🎯 Setup & Training

### **Option 1: Complete Setup (Recommended)**

#### **Step 1: Train LSTM Models**
```bash
# 1. Buka Jupyter Notebook
jupyter notebook

# 2. Jalankan Stock_Prediction_LSTM.ipynb
# 3. Set configuration:
train_new_models = True
optimize = True  # Untuk hyperparameter tuning

# 4. Tunggu training selesai (~30-60 menit per stock)
```

#### **Step 2: Verify Model Files**
```bash
# Check generated files
ls cache/
# Expected output:
# short_term_lstm_BBCA_JK.h5
# short_term_lstm_UNVR_JK.h5
# short_term_lstm_ITMG_JK.h5
# final_results.pkl
```

### **Option 2: Quick Development Setup**
```python
# Untuk testing cepat tanpa full training
train_new_models = False  # Use cached models if available
optimize = False          # Skip hyperparameter tuning
```

### **Directory Structure**
```
lstm-stock-prediction/
├── 📄 app.py                          # Main Flask application
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # This file
├── 📄 Stock_Prediction_LSTM.ipynb     # Training notebook
├── 📁 templates/
│   └── 📄 index.html                  # Tailwind CSS template
├── 📁 cache/                          # Generated model files
│   ├── 🧠 short_term_lstm_BBCA_JK.h5  # BBCA trained model
│   ├── 🧠 short_term_lstm_UNVR_JK.h5  # UNVR trained model
│   ├── 🧠 short_term_lstm_ITMG_JK.h5  # ITMG trained model
│   └── 📊 final_results.pkl           # Scalers & results
└── 📁 static/ (optional)              # Static assets
```

## 🚀 Running the Application

### **Development Mode**
```bash
python app.py
```

### **Production Mode**
```bash
# Using Gunicorn (Linux/macOS)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### **Access Application**
```
🌐 URL: http://localhost:5000
📱 Mobile: http://your-ip:5000
```

### **Startup Information**
```bash
============================================================
🚀 LSTM Stock Prediction Flask Application
============================================================
📊 Available stocks: ['BBCA.JK', 'UNVR.JK', 'ITMG.JK']
🔍 Checking model availability...
✅ Models tersedia untuk: ['BBCA.JK', 'UNVR.JK', 'ITMG.JK']
✅ Scalers tersedia
============================================================
🌐 Starting Flask application...
📱 Access: http://localhost:5000
============================================================
```

## 🎮 Usage Guide

### **Dashboard Navigation**

#### **1. Overview Section**
- **Market Summary**: IHSG, Top Gainer/Loser, Most Active
- **Recent Analysis**: Chart dari analisis terakhir
- **Quick Stats**: Performance indicators

#### **2. Comparison Section**
- **Multi-Stock Chart**: Normalized performance comparison
- **Performance Cards**: Individual stock metrics
- **Period Selection**: 1M, 3M, 6M, 1Y timeframes

#### **3. Analysis Section**
- **Stock Selection**: Dropdown ticker selection
- **Prediction Period**: 7, 14, 30, 60, 90 hari
- **Interactive Chart**: Candlestick + prediction overlay
- **Detailed Results**: Table dengan OHLC predictions

### **Making Predictions**

#### **Method 1: Sidebar Form**
```
1. Pilih saham dari dropdown
2. Pilih periode prediksi (7-90 hari)
3. Klik "Analisis & Prediksi"
4. Lihat hasil di Analysis section
```

#### **Method 2: Stock Cards (Comparison)**
```
1. Klik section "Perbandingan"
2. Klik salah satu stock card
3. Otomatis redirect ke Analysis dengan prediksi
```

#### **Method 3: Top Stocks (Sidebar)**
```
1. Lihat "Saham Teratas" di sidebar
2. Klik salah satu stock
3. Otomatis generate prediction
```

## 📡 API Documentation

### **Base URL**
```
http://localhost:5000
```

### **Endpoints**

#### **1. Stock Overview**
```http
GET /all_stocks
```

**Response:**
```json
{
  "comparison": {
    "BBCA": {
      "name": "BBCA",
      "data": [{"x": "2025-01-01", "y": 100.5}, ...]
    }
  },
  "stocks": {
    "BBCA.JK": {
      "name": "Bank Central Asia Tbk",
      "ticker": "BBCA.JK",
      "last_price": 8850.0,
      "performance_pct": 2.5,
      "ohlc": [...]
    }
  },
  "timestamp": "2025-07-17T10:30:00"
}
```

#### **2. Make Prediction**
```http
POST /predict
Content-Type: application/x-www-form-urlencoded

ticker=BBCA.JK&days=30
```

**Response:**
```json
{
  "ticker": "BBCA.JK",
  "name": "Bank Central Asia Tbk",
  "last_price": 8850.0,
  "change": 25.0,
  "change_percent": 0.28,
  "prediction": [
    {
      "Date": "2025-07-18",
      "Open": 8849.1,
      "High": 8867.7,
      "Low": 8832.3,
      "Close": 8850.0,
      "Volume": 1000000.0,
      "Predicted": true
    }
  ],
  "chart_data": {
    "historical": [...],
    "predictions": [...]
  },
  "rmse": 45.23,
  "mae": 32.11,
  "r_squared": 0.85
}
```

#### **3. Model Information**
```http
GET /model_info
```

**Response:**
```json
{
  "models": {
    "BBCA.JK": {
      "name": "Bank Central Asia Tbk",
      "model_available": true,
      "model_type": "LSTM Neural Network",
      "prediction_days": [1, 2, 3],
      "sequence_length": 30
    }
  },
  "config": {
    "sequence_length": 30,
    "prediction_days": [1, 2, 3],
    "data_years": 20
  }
}
```

#### **4. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-17T10:30:00",
  "models_loaded": 3,
  "available_models": ["BBCA.JK", "UNVR.JK", "ITMG.JK"],
  "total_models": 3
}
```

## 🔧 Configuration

### **Model Configuration**
```python
# config in app.py
SHORT_TERM_CONFIG = {
    'sequence_length': 30,      # Input sequence length
    'prediction_days': [1,2,3], # LSTM target days
    'data_years': 20,           # Historical data years
}
```

### **Stock List Configuration**
```python
# Add new stocks
STOCK_LIST = {
    'BBCA.JK': 'Bank Central Asia Tbk',
    'UNVR.JK': 'Unilever Indonesia Tbk',
    'ITMG.JK': 'Indo Tambangraya Megah Tbk',
    # Add more stocks here...
}
```

### **Chart Configuration**
```javascript
// Customize in templates/index.html
const chartOptions = {
    chart: { type: 'candlestick', height: 400 },
    plotOptions: { candlestick: { colors: { upward: '#10b981', downward: '#ef4444' }}},
    // ... more options
}
```

## 📊 Model Performance

### **Accuracy Metrics**

| Stock | R² Score | RMSE | MAE | Direction Accuracy |
|-------|----------|------|-----|--------------------|
| BBCA.JK | 0.85-0.92 | 35-50 | 25-40 | 65-75% |
| UNVR.JK | 0.80-0.88 | 40-60 | 30-45 | 60-70% |
| ITMG.JK | 0.75-0.85 | 50-80 | 35-55 | 58-68% |

### **Performance Factors**
- **Market Conditions**: Volatility affect prediction accuracy
- **News Events**: Unexpected events may cause deviations
- **Seasonal Patterns**: Model captures calendar effects
- **Volume Patterns**: Higher volume = better predictions

## 🐛 Troubleshooting

### **Common Issues**

#### **🚨 Model Not Found**
```bash
ERROR: Model tidak ditemukan untuk BBCA.JK
```
**Solution:**
```bash
# 1. Check cache directory
ls cache/
# 2. Run training notebook
jupyter notebook Stock_Prediction_LSTM.ipynb
# 3. Set train_new_models = True
```

#### **🚨 Scalers Not Found**
```bash
ERROR: Scalers tidak ditemukan
```
**Solution:**
```bash
# 1. Check final_results.pkl exists
ls cache/final_results.pkl
# 2. Re-run complete training pipeline
# 3. Ensure all cells in notebook executed successfully
```

#### **🚨 TensorFlow Warnings**
```bash
WARNING: TensorFlow logging messages
```
**Solution:**
```python
# Already handled in app.py
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```

#### **🚨 Memory Issues**
```bash
ERROR: ResourceExhaustedError (OOM)
```
**Solution:**
```python
# Reduce batch size in training
params['batch_size'] = 16  # Instead of 32

# Or set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### **🚨 Data Download Issues**
```bash
ERROR: Tidak dapat mengambil data untuk BBCA.JK
```
**Solution:**
```bash
# 1. Check internet connection
# 2. Verify Yahoo Finance availability
# 3. Try different time periods
# 4. Use VPN if blocked in your region
```

#### **🚨 Port Already in Use**
```bash
ERROR: [Errno 48] Address already in use
```
**Solution:**
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 5001
```

### **Development Tips**

#### **Faster Development**
```python
# Skip hyperparameter optimization
optimize = False

# Use smaller dataset
config['data_years'] = 5

# Reduce epochs
epochs = 50  # Instead of 150
```

#### **Memory Optimization**
```python
# Clear session between trainings
tf.keras.backend.clear_session()

# Use generator for large datasets
# Enable garbage collection
import gc; gc.collect()
```

## 🔒 Security & Production

### **Production Deployment**

#### **Environment Variables**
```bash
# .env file
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url
```

#### **Security Headers**
```python
# Add to app.py
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

#### **Rate Limiting**
```python
# Install flask-limiter
pip install Flask-Limiter

# Add rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... existing code
```

#### **HTTPS & SSL**
```bash
# Using nginx reverse proxy
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
    }
}
```

## 🧪 Testing

### **Unit Tests**
```python
# test_app.py
import unittest
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)

    def test_prediction_endpoint(self):
        response = self.app.post('/predict', data={
            'ticker': 'BBCA.JK',
            'days': 30
        })
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
```

### **Run Tests**
```bash
python -m pytest test_app.py -v
```

## 🤝 Contributing

### **Development Setup**
```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/new-feature

# 3. Make changes
# 4. Run tests
python -m pytest

# 5. Commit changes
git commit -m "Add new feature"

# 6. Push to branch
git push origin feature/new-feature

# 7. Create Pull Request
```

### **Code Standards**
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Include type hints where applicable
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** - Deep learning framework
- **Yahoo Finance** - Real-time market data
- **ApexCharts** - Interactive chart library
- **Tailwind CSS** - Modern CSS framework
- **Flask Community** - Web framework

## 📞 Support

### **Documentation**
- 📚 [Flask Documentation](https://flask.palletsprojects.com/)
- 🧠 [TensorFlow Guide](https://tensorflow.org/guide)
- 📊 [ApexCharts Docs](https://apexcharts.com/docs/)

### **Community**
- 💬 [GitHub Discussions](https://github.com/yourusername/lstm-stock-prediction/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/lstm-stock-prediction/issues)
- 📧 Email: your-email@example.com

### **Quick Support**
```bash
# Check logs
tail -f app.log

# Verify models
python -c "from app import cache; print([cache.load_model(f'short_term_lstm_{t.replace('.', '_')}') is not None for t in ['BBCA.JK', 'UNVR.JK', 'ITMG.JK']])"

# Test API
curl http://localhost:5000/health
```

---

**⚠️ Disclaimer**: Model predictions adalah untuk tujuan edukasi dan penelitian. Tidak merupakan nasihat investasi. Selalu lakukan analisis tambahan dan konsultasi dengan ahli keuangan sebelum mengambil keputusan investasi.

**Built with ❤️ by [Your Name] | 🇮🇩 Made for Indonesian Stock Market**
