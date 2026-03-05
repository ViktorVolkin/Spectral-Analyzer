import pandas as pd
import numpy as np

def load_spectral_data(file_path):
    """Загружает данные и возвращает x (nm) и y (R%)"""
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError("Файл должен содержать минимум 2 столбца (λ и R)")
    
    x = np.array(df.iloc[:, 0].values, dtype=float)
    y = np.array(df.iloc[:, 1].values, dtype=float)
    return x, y