import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import time
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Sequential

logger = logging.getLogger('azzraelcode-yt')

def load_and_preprocess_data(file='data/candles_data.csv'):
    if not os.path.exists(file):  # Новое: проверка
        logger.error(f"Файл {file} не найден. Загрузка данных пропущена.")
        return pd.DataFrame()
    df = pd.read_csv(file)
    logger.info(f"Сырые данные: {len(df)} строк, столбцы: {df.columns.tolist()}")
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"После очистки NaN/inf: {len(df)} строк")
    df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    logger.info(f"После добавления direction и dropna: {len(df)} строк")
    return df

def create_sequences(data, seq_length=100, features=['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'atr', 'stoch_k', 'stoch_d']):
    logger.info(f"Features для модели: {features}")
    print("Пример первых 5 строк features:\n", data[features].head())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    sequences = []
    labels = []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i + seq_length])
        labels.append(data['direction'].iloc[i + seq_length])
    return np.array(sequences), np.array(labels), scaler

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    log_file = 'data/predictor_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=['timestamp', 'predicted', 'actual', 'correct', 'win_rate'])
    seq_length = 100
    features = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'atr', 'stoch_k', 'stoch_d']
    while True:
        data = load_and_preprocess_data()
        if data.empty:  # Новое: если файл не найден или пустой
            time.sleep(60)
            continue
        if len(data) < seq_length + 1:
            logger.warning(f"Недостаточно данных: {len(data)}, требуется {seq_length + 1}. Ожидаю обновления...")
            time.sleep(60)
            continue
        X, y, scaler = create_sequences(data, seq_length, features)
        model = build_model((seq_length, X.shape[2]))
        model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        logger.info("Модель обучена на исторических данных")
        latest_scaled = scaler.transform(data[features].tail(seq_length))
        prediction = model.predict(np.array([latest_scaled]))[0][0]
        predicted_dir = 1 if prediction > 0.5 else 0
        logger.info(f"Предсказание направления следующей свечи: {predicted_dir} (1=up, 0=down)")
        time.sleep(60)
        data = load_and_preprocess_data()
        actual_dir = data['direction'].iloc[-1]
        correct = predicted_dir == actual_dir
        win_rate = (log_df['correct'].sum() + correct) / (len(log_df) + 1) * 100 if not log_df.empty else (100 if correct else 0)
        new_row = pd.DataFrame([[data['timestamp'].iloc[-1], predicted_dir, actual_dir, correct, win_rate]], columns=log_df.columns)
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        log_df.to_csv(log_file, index=False)
        logger.info(f"Факт направления: {actual_dir}, Угадала: {correct}, Win-rate: {win_rate:.2f}%")

if __name__ == '__main__':
    main()