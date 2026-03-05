import numpy as np

def find_extrema_from_derivative(x, dy_dx):
    """Надежный поиск экстремумов с фильтрацией близких ложных срабатываний"""
    peaks = []
    valleys = []

    s = np.sign(dy_dx.astype(float))
    if np.any(s == 0):
        for i in range(1, len(s)):
            if s[i] == 0 and s[i-1] != 0:
                s[i] = s[i-1]
        for i in range(len(s)-2, -1, -1):
            if s[i] == 0 and s[i+1] != 0:
                s[i] = s[i+1]

    for i in range(1, len(s)):
        if s[i-1] > 0 and s[i] < 0:
            peaks.append(int(i))
        elif s[i-1] < 0 and s[i] > 0:
            valleys.append(int(i))

    peaks = np.unique(peaks)
    valleys = np.unique(valleys)

    min_distance = 2
    if len(peaks) > 1:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks, dtype=int)
    else:
        peaks = np.array(peaks, dtype=int)

    if len(valleys) > 1:
        filtered_valleys = [valleys[0]]
        for v in valleys[1:]:
            if v - filtered_valleys[-1] >= min_distance:
                filtered_valleys.append(v)
        valleys = np.array(filtered_valleys, dtype=int)
    else:
        valleys = np.array(valleys, dtype=int)

    return peaks, valleys

def find_top_two_peaks(x, y, peaks):
    """Поиск двух самых высоких пиков с проверкой дистанции между ними"""
    if len(peaks) < 2:
        raise ValueError("Не найдено достаточно пиков.")
    
    peak_values = y[peaks]
    sorted_idx = np.argsort(peak_values)[::-1]

    first = int(peaks[sorted_idx[0]])
    min_sep = max(int(0.05 * len(x)), 2)

    second = None
    for idx in sorted_idx[1:]:
        cand = int(peaks[idx])
        if abs(cand - first) >= min_sep:
            second = cand
            break

    if second is None:
        second = int(peaks[sorted_idx[1]])

    peak1_idx, peak2_idx = (first, second) if x[first] < x[second] else (second, first)
    return int(peak1_idx), int(peak2_idx)

def find_valley_between_peaks(y_smooth, peak1_idx, peak2_idx):
    """Поиск самой глубокой точки строго между двумя пиками"""
    if peak2_idx <= peak1_idx:
        start, end = min(peak1_idx, peak2_idx), max(peak1_idx, peak2_idx)
    else:
        start, end = peak1_idx, peak2_idx

    slice_vals = y_smooth[start:end+1]
    if slice_vals.size == 0:
        return int(start)

    local_min = int(np.argmin(slice_vals))
    return int(start + local_min)

def calculate_physics(x, y_smooth, p1, p2, v):
    """Геометрический расчет Delta R и Q-фактора"""
    y_base = np.interp(x[v], [x[p1], x[p2]], [y_smooth[p1], y_smooth[p2]])
    delta_r = y_base - y_smooth[v]
    
    r_peak1 = y_smooth[p1]
    q_factor = (delta_r / r_peak1 * 100) if r_peak1 != 0 else 0
    
    return float(delta_r), float(q_factor)

def compute_precise_x(x, dy_dx, idx):
    """Суб-пиксельный поиск нуля производной"""
    n = len(dy_dx)
    if idx <= 0 or idx >= n:
        return float(x[min(max(idx, 0), n-1)])

    if dy_dx[idx] == 0:
        return float(x[idx])

    i0, i1 = idx-1, idx
    y0, y1 = float(dy_dx[i0]), float(dy_dx[i1])
    if y0 * y1 < 0:
        x0 = x[i0] - y0 * (x[i1] - x[i0]) / (y1 - y0)
        return float(x0)

    if idx+1 < n:
        i0, i1 = idx, idx+1
        y0, y1 = float(dy_dx[i0]), float(dy_dx[i1])
        if y0 * y1 < 0:
            x0 = x[i0] - y0 * (x[i1] - x[i0]) / (y1 - y0)
            return float(x0)

    return float(x[idx])