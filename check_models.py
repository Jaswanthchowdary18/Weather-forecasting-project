import joblib
from tensorflow.keras.models import load_model

print("Loading ARIMA model...")
arima = joblib.load("outputs/models/arima_model.pkl.joblib")
print("ARIMA Loaded Successfully!")

print("Loading LSTM model...")
lstm = load_model("outputs/models/lstm_model.h5", compile=False)
print("LSTM Loaded Successfully!")

print("All models loaded correctly!")
