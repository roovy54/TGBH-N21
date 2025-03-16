import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# Step 1: Generate Synthetic Spatiotemporal Data
def generate_synthetic_data(num_samples=10000, time_steps=10, grid_size=100):
    data = []
    for _ in range(num_samples):
        time_idx = np.random.randint(0, 24)  # Simulating hourly data
        x = np.random.randint(0, grid_size)  # X coordinate in spatial grid
        y = np.random.randint(0, grid_size)  # Y coordinate in spatial grid
        data.append([time_idx, x, y])
    return pd.DataFrame(data, columns=["time", "x", "y"])


# Generate dataset
df = generate_synthetic_data()

# Step 2: Preprocess Data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 1:])  # Predict next x, y coordinates
    return np.array(X), np.array(y)


# Create sequences
time_steps = 10
X, y = create_sequences(df_scaled, time_steps)

# Step 3: Build LSTM Model
model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(time_steps, 3)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(2, activation="linear"),  # Predict x, y coordinates
    ]
)

model.compile(optimizer="adam", loss="mse")

# Step 4: Train Model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Step 5: Predict Future Demand
predictions = model.predict(X[-1].reshape(1, time_steps, 3))
predicted_coordinates = scaler.inverse_transform([[0] + predictions.tolist()[0]])[0][1:]
print("Predicted future ride location:", predicted_coordinates)
