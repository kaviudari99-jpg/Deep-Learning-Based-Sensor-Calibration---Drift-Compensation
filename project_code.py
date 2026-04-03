import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization Functions
def plot_data_analysis(truth, drift, pairs):
    #Distribution Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(truth['Average value'], bins=50, alpha=0.5, label='Truth', color='blue')
    plt.hist(drift['Average value'], bins=50, alpha=0.5, label='Drift', color='orange')
    plt.title('Distribution of PM2.5 Values')
    plt.legend()
    
    # Time Series Plot (First 500 Hours)
    plt.subplot(1, 2, 2)
    subset = pairs.head(500)
    plt.plot(subset['Measurement date'], subset['Average value_x'], label='Truth', color='blue')
    plt.plot(subset['Measurement date'], subset['Average value_y'], label='Drift', color='orange', alpha=0.7)
    plt.title('Time Series (First 500 Hours)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_results(results_dict):

    #Loss Curves (Top Row) & 2. Scatter Plots (Bottom Row) for each model in a 2x4 grid.
    models = list(results_dict.keys())
    fig,axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, m_name in enumerate(models):
        ax_loss = axes[0, i]
        history = results_dict[m_name]['history']
        ax_loss.plot(history.history['loss'], label='Train MSE')
        ax_loss.plot(history.history['val_loss'], label='Val MSE')
        ax_loss.set_title(f'{m_name} Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.legend()

        ax_fit = axes[1, i]
        actual = results_dict[m_name]['actual']
        preds = results_dict[m_name]['preds']
        r2 = results_dict[m_name]['test_r2']
        ax_fit.scatter(actual, preds, alpha=0.3, color='teal')
        ax_fit.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax_fit.set_title(f'{m_name} Fit (R2: {r2:.2f})')
        ax_fit.set_xlabel('True PM2.5')
        ax_fit.set_ylabel('Predicted PM2.5')

    plt.tight_layout()
    plt.show()

def plot_metric_comparison(results_dict):
    # Bar charts comparing Train vs Test metrics for each model in a 2x2 grid.
    models = list(results_dict.keys())
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    colors = {'RMSE': 'red', 'MAE': 'blue', 'MAPE': 'orange', 'R2': 'purple'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, m_metric in enumerate(metrics):
        ax = axes[i]
        train_vals = [results_dict[m][f'train_{m_metric.lower()}'] for m in models]
        test_vals = [results_dict[m][f'test_{m_metric.lower()}'] for m in models]
        
        x = np.arange(len(models))
        ax.bar(x, test_vals, 0.7, label=f'Test {m_metric}', color=colors[m_metric], alpha=0.2)
        ax.bar(x, train_vals, 0.5, label=f'Train {m_metric}', color=colors[m_metric], alpha=0.8)
        
        ax.set_title(m_metric if m_metric != 'R2' else '$R^2$', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        if m_metric == 'R2': ax.set_ylim(0.9, 1.0)

    plt.tight_layout()
    plt.show()

# Modeling & Evaluation Functions
def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100

def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_and_evaluate(name, model, X_train, y_train, X_test, y_test, scaler_y, epochs=100, callbacks=None):
    print(f"Training {name}...")
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0)
    
    def get_metrics(X, y):
        p_log = model.predict(X, verbose=0)
        p = np.expm1(scaler_y.inverse_transform(p_log))
        real = np.expm1(scaler_y.inverse_transform(y))

        return p, real, np.sqrt(mean_squared_error(real, p)), mean_absolute_error(real, p), mape_score(real, p), r2_score(real, p)

    t_p, t_r, tr_rmse, tr_mae, tr_mape, tr_r2 = get_metrics(X_train, y_train)
    te_p, te_r, te_rmse, te_mae, te_mape, te_r2 = get_metrics(X_test, y_test)

    return {
        'history': history, 
        'preds': te_p, 
        'actual': te_r,
        'train_rmse': tr_rmse, 
        'train_mae': tr_mae, 
        'train_mape': tr_mape, 
        'train_r2': tr_r2,
        'test_rmse': te_rmse, 
        'test_mae': te_mae, 
        'test_mape': te_mape, 
        'test_r2': te_r2
    }
def plot_calibration_results(results_dict, model_name, pairs):
    plt.figure(figsize=(15, 6))
    actual = results_dict[model_name]['actual'].flatten()
    preds = results_dict[model_name]['preds'].flatten()
    test_size = len(actual)
    time_axis = pairs['Measurement date'].iloc[-test_size:]
    original_drift = pairs['Average value_y'].iloc[-test_size:]

    plt.plot(time_axis, actual, label='Reference Truth', color='blue', linewidth=2)
    plt.plot(time_axis, original_drift, label='Original Drift (Status 1)', color='orange', alpha=0.5, linestyle='--')
    plt.plot(time_axis, preds, label=f'Calibrated ({model_name})', color='green', linewidth=1.5)

    plt.title(f'Time Series Comparison: Before vs. After Calibration ({model_name})')
    plt.xlabel('Measurement Date')
    plt.ylabel('PM2.5 Concentration ($\mu g/m^3$)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Load Data (Seoul Air Pollution Dataset)
    df = pd.read_csv(os.path.join('data', 'Measurement_info.csv'))
    pm25 = df[df['Item code'] == 9].copy()
    pm25['Measurement date'] = pd.to_datetime(pm25['Measurement date'])
    pm25['H_sin'] = np.sin(2 * np.pi * pm25['Measurement date'].dt.hour / 24)
    pm25['H_cos'] = np.cos(2 * np.pi * pm25['Measurement date'].dt.hour / 24)
    
    truth = pm25[pm25['Instrument status'] == 0].groupby('Measurement date')['Average value'].mean().reset_index()
    drift = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value', 'H_sin', 'H_cos']]
    pairs = pd.merge(truth, drift, on='Measurement date').dropna()

    plot_data_analysis(truth, drift, pairs)

    #Data Preparation
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(pairs[['Average value_y', 'H_sin', 'H_cos']].values)
    y_scaled = scaler_y.fit_transform(np.log1p(pairs[['Average value_x']].values))
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, 24)
    X_tr_lstm, X_te_lstm, y_tr, y_te = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_tr_mlp, X_te_mlp = X_tr_lstm.reshape(X_tr_lstm.shape[0], -1), X_te_lstm.reshape(X_te_lstm.shape[0], -1)

    results = {}
    stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    #Model Building & Evaluation
    m1 = tf.keras.Sequential([layers.Input(shape=(X_tr_mlp.shape[1],)), layers.Dense(32, activation='relu'), layers.Dense(1)])
    l1 = tf.keras.Sequential([layers.Input(shape=(24, 3)), layers.LSTM(32), layers.Dense(1)])
    results['Iter1_MLP'] = build_and_evaluate("MLP v1", m1, X_tr_mlp, y_tr, X_te_mlp, y_te, scaler_y, 50)
    results['Iter1_LSTM'] = build_and_evaluate("LSTM v1", l1, X_tr_lstm, y_tr, X_te_lstm, y_te, scaler_y, 50)

    m2 = tf.keras.Sequential([layers.Input(shape=(X_tr_mlp.shape[1],)), layers.Dense(128, activation='relu'), layers.Dropout(0.2), layers.Dense(1)])
    l2 = tf.keras.Sequential([layers.Input(shape=(24, 3)), layers.LSTM(64, return_sequences=True), layers.LSTM(32), layers.Dense(1)])
    results['Iter2_MLP'] = build_and_evaluate("MLP v2", m2, X_tr_mlp, y_tr, X_te_mlp, y_te, scaler_y, 100, [stop])
    results['Iter2_LSTM'] = build_and_evaluate("LSTM v2", l2, X_tr_lstm, y_tr, X_te_lstm, y_te, scaler_y, 100, [stop])

    # Visualization of Results
    plot_training_results(results)
    plot_metric_comparison(results)
    plot_calibration_results(results, 'Iter2_LSTM', pairs)

    #Summary Table
    summary = [{"Model": k, "R2": round(v['test_r2'], 3), "RMSE": round(v['test_rmse'], 3)} for k, v in results.items()]
    print("\nModel Performance Summary:")
    print(pd.DataFrame(summary))

if __name__ == '__main__': main()
