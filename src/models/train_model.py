import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense



# OBDELAVA PODATKOV
bike_data = pd.read_csv('../../data/mbajk_dataset.csv')

# Sortiranje zapisov glede na čas zapis
bike_data['date'] = pd.to_datetime(bike_data['date'])
bike_data = bike_data.sort_values(by='date')
#print(bike_data.head())


# Filtriramo ciljno značilnost "available_bike_stands"
time_series = bike_data[['date', 'available_bike_stands']].copy()

# Preoblikujemo DataFrame v numpy array
time_series_np = time_series['available_bike_stands'].values.reshape(-1, 1)

# Razdelimo na učno in testno množico
train_size = len(time_series_np) - 1302
train_model, test = time_series_np[:train_size], time_series_np[train_size:]

# Standardizacija podatkov
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_model)
test_scaled = scaler.transform(test)
joblib.dump(scaler, '../../models/standard_scaler.pkl')



# PRIPRAVA PODATKOV

# Funkcija za pripravo vhodnih in izhodnih učnih podatkov
def create_time_series_data(time_series, window_size):
    X, y = [], []
    for i in range(len(time_series) - window_size):
        window = time_series[i:(i + window_size), 0]
        X.append(window)
        y.append(time_series[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 186

# Vhodni in izhodni učni podatki za učenje
X_train, y_train = create_time_series_data(train_scaled, window_size)
X_test, y_test = create_time_series_data(test_scaled, window_size)

# Preoblikujemo vhodne učne podatke v obliko
# (število primerkov, velikost koraka, število vrednosti)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# print("Oblika učnih podatkov:")
# print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
# print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")




# IZGRADNJA IN UČENJE NAPOVEDNEGA MODELA

def build_model():
    model = Sequential()

    model.add(SimpleRNN(32, activation='relu', input_shape=(1, window_size), return_sequences=True))
    model.add(SimpleRNN(32, activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    # Open the file in write mode and save the printed values
    with open('../../reports/train_metrics.txt', "w") as file:
        file.write(f"Loss: {history.history['loss']} \nVal_loss: {history.history['val_loss']}")
    print("Train metrics saved!")

# Izgradnja modela
rnn_model = build_model()

# Učenje modela
epochs = 15
train_model(rnn_model, X_train, y_train, epochs)

rnn_model.save('../../models/prediction_model.h5')
print("Model trained and saved!")



# OVREDNOTENJE NAPOVEDNEGA MODELA
def evaluate_model(model, X_test, y_test):
    # Napovedovanje na testni množici
    y_pred = model.predict(X_test)

    # Inverzna standardizacija napovedanih in dejanskih vrednosti
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Izračun metrik
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    ev = explained_variance_score(y_test_inv, y_pred_inv)

    return mse, mae, ev, y_pred_inv

mse_rnn, mae_rnn, ev_rnn, predictions_inv_rnn = evaluate_model(rnn_model, X_test, y_test)


# Open the file in write mode and save the printed values
with open('../../reports/metrics.txt', "w") as file:
    file.write(f"MSE: {mse_rnn}\nMAE: {mae_rnn}\nEV: {ev_rnn}")
print("Test metrics saved!")