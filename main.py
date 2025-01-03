import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import matplotlib
import joblib
import yfinance as yf

ENABLE_VISUALIZATION = False
VALID_TICKERS = ['NVDA', 'INTC', 'AMD']

# Contains the paths of each source files for each of the different companies
SOURCE_FILES = {
    'NVDA': 'learning_data/NVIDIA (1999 -11.07.2023).csv',
    'INTC': 'learning_data/INTEL (1980 - 11.07.2023).csv',
    'AMD': 'learning_data/AMD (1980 -11.07.2023).csv',
}
ALGORITHM_METADATA = {
    'RandForestReg': [
        'stock_predictor_random_forest_reg.pkl',
        'Rand Forest Reg'
    ],
    'LinReg': [
        'stock_predictor_lin_reg.pkl',
        'Linear Reg'
    ],
    'Ridge': [
        'stock_predictor_ridge.pkl',
        'Ridge Reg'
    ]
}
matplotlib.use('TkAgg')


def main():
    print(', '.join(VALID_TICKERS))
    ticker = input('Enter a valid ticker: ').upper()
    while ticker not in VALID_TICKERS:
        ticker = input('Enter a valid ticker: ').upper()

    # Load the Data
    data, features = load_data(ticker)

    # Select Target
    target = data['Close']

    # Scale the Features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Save the Scaler for Future Use
    joblib.dump(scaler, 'scaler.pkl')

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Train the three Models and save them
    for key, metadata in ALGORITHM_METADATA.items():
        model_class = {
            'RandForestReg': RandomForestRegressor,
            'LinReg': LinearRegression,
            'Ridge': Ridge
        }[key]
        model = model_class()
        train_model(model, X_train, y_train, X_test, y_test, metadata)

    # Predict Today's Stock Price
    stock = yf.Ticker(ticker)
    data = stock.history(period='5d')

    current_day = data.iloc[-1]
    previous_day = data.iloc[-2]

    current_open = current_day['Open']
    current_volume = current_day['Volume']

    previous_close = previous_day['Close']
    previous_volume = previous_day['Volume']

    print(f"Current Day (Open, Volume): {current_open}, {current_volume}")
    print(f"Previous Day (Close, Volume): {previous_close}, {previous_volume}")

    today_data = {
        'Open': current_open,
        'Volume': current_volume,
        'Lag_Close': previous_close,
        'Lag_Volume': previous_volume
    }

    today_features = pd.DataFrame([today_data])

    scaler = joblib.load('scaler.pkl')
    today_features_scaled = scaler.transform(today_features)

    for key, value in ALGORITHM_METADATA.items():
        predict_stock_price(value, today_features_scaled)


def find_mean_squared_error(model, X_test, y_test, debug_text):
    """
    Evaluates the model and its mean squared error
    Visualizes the Predictions vs Actual if ENABLE_VISUALIZATION is set to True
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error - {debug_text}: {mse}")

    if ENABLE_VISUALIZATION:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Stock Prices')
        plt.legend()
        plt.show()


def predict_stock_price(metadata, today_features_scaled):
    """
    Loads the given model and predicts the stock price
    """
    file_name, debug_text = metadata
    model = joblib.load(file_name)
    predicted_price = model.predict(today_features_scaled)

    print(f"Predicted Stock Price for Today - {debug_text}: {predicted_price[0]:.2f}")


def prepare_data(data):
    # Convert 'Date' Column to Datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Feature Engineering: Add Lag Features
    data['Lag_Close'] = data['Close'].shift(1)
    data['Lag_Volume'] = data['Volume'].shift(1)

    # Drop rows with missing values due to lagging
    data.dropna(inplace=True)

    # Set features
    features = data[['Open', 'Volume', 'Lag_Close', 'Lag_Volume']]
    return [data, features]


def load_data(ticker):
    try:
        data = pd.read_csv(SOURCE_FILES[ticker])
    except FileNotFoundError:
        raise FileNotFoundError('Source file not found')
    prepared_data = prepare_data(data)
    return prepared_data


def train_model(model, X_train, y_train, X_test, y_test, algorithm_metadata):
    """
    Trains the model and saves it with the given file_name from the metadata
    """
    file_name, debug_text = algorithm_metadata
    model.fit(X_train, y_train)
    joblib.dump(model, file_name)
    find_mean_squared_error(model, X_test, y_test, debug_text)


if __name__ == '__main__':
    main()
