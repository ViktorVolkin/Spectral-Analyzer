import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import config
from core.math_utils import find_extrema_from_derivative, find_top_two_peaks, find_valley_between_peaks

def calculate_auto_window_nm(x, y):
    """Двухпроходный автоподбор оптимального окна"""
    delta_x = abs(np.mean(np.diff(x)))
    rough_pts = int(config.ROUGH_WINDOW_NM / delta_x)
    rough_pts = max(config.MIN_POINTS_LIMIT, rough_pts if rough_pts % 2 != 0 else rough_pts + 1)
    
    try:
        y_rough = savgol_filter(y, window_length=rough_pts, polyorder=config.POLYORDER)
        dy_dx_rough = savgol_filter(y, window_length=rough_pts, polyorder=config.POLYORDER, deriv=1, delta=delta_x)
        
        peaks, _ = find_extrema_from_derivative(x, dy_dx_rough)
        if len(peaks) >= 2:
            p1, p2 = find_top_two_peaks(x, y_rough, peaks)
            v = find_valley_between_peaks(y_rough, p1, p2)
            distance = abs(x[v] - x[p1])
            return max(5.0, min(distance / 2.0, 85.0))
    except: pass
    return config.DEFAULT_WINDOW_NM

def process_spectrum(x, y, window_nm):
    """Чистовое сглаживание и расчет первой производной (без второй)"""
    delta_x = abs(np.mean(np.diff(x)))
    window_len = int(window_nm / delta_x)
    window_len = max(config.MIN_POINTS_LIMIT, window_len if window_len % 2 != 0 else window_len + 1)
    
    y_smooth = savgol_filter(y, window_len, config.POLYORDER)
    dy_dx = savgol_filter(y, window_len, config.POLYORDER, deriv=1, delta=delta_x)
    
    smooth_size = max(5, int(len(x) / 55))
    if smooth_size % 2 == 0: smooth_size += 1
    
    dy_dx = uniform_filter1d(dy_dx, size=smooth_size)
    
    return y_smooth, dy_dx