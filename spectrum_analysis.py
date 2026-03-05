import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import os
from pathlib import Path


def calculate_auto_window_nm(x, y, polyorder=3, rough_window_nm=60.0):
    """
    ДВУХПРОХОДНЫЙ АВТОПОДБОР:
    Делает черновое сильное сглаживание, находит расстояние между левым пиком 
    и долиной, и возвращает оптимальное окно в нанометрах.
    """
    delta_x = abs(float(np.mean(np.diff(x))))
    if delta_x == 0: delta_x = 1.0
    
    rough_pts = int(rough_window_nm / delta_x)
    if rough_pts % 2 == 0: rough_pts += 1
    
    # ЖЕСТКАЯ ПОДУШКА БЕЗОПАСНОСТИ ДЛЯ ЧЕРНОВИКА
    rough_pts = max(11, rough_pts)
    
    if rough_pts < polyorder + 2: 
        rough_pts = max(3, polyorder + 2)
        if rough_pts % 2 == 0: rough_pts += 1
    if rough_pts > len(y): 
        rough_pts = len(y) if len(y) % 2 == 1 else max(1, len(y) - 1)
        
    try:
        y_rough = savgol_filter(y, window_length=rough_pts, polyorder=polyorder, deriv=0)
        dy_dx_rough = savgol_filter(y, window_length=rough_pts, polyorder=polyorder, deriv=1, delta=delta_x)
        
        peaks, valleys = find_extrema_from_derivative(x, dy_dx_rough)
        
        if len(peaks) >= 2:
            peak1_idx, peak2_idx = find_top_two_peaks(x, y_rough, peaks)
            valley_idx = find_valley_between_peaks(x, y_rough, peak1_idx, peak2_idx)
            
            if valley_idx is not None:
                distance_nm = abs(x[valley_idx] - x[peak1_idx])
                
                # Форсированное сглаживание: берем окно равное 40% расстояния
                optimal_window = distance_nm / 2.0
                optimal_window = max(5.0, min(optimal_window, 80.0))
                return optimal_window
    except Exception:
        pass
        
    return 15.0


def calculate_all_physics(x, y, window_nm=15.0, polyorder=3):
    """
    Универсальная функция: возвращает сглаженную кривую и её первые две производные.
    Внедрена адаптивная полировка для устранения "ребристости" на больших сетах данных.
    """
    polyorder = int(polyorder)
    
    if len(x) > 1:
        delta_x = abs(float(np.mean(np.diff(x))))
        if delta_x == 0: delta_x = 1.0
    else:
        delta_x = 1.0

    window_length = int(float(window_nm) / delta_x)
    if window_length % 2 == 0:
        window_length += 1 
        
    window_length = max(11, window_length)
        
    if window_length < polyorder + 2:
        window_length = max(3, polyorder + 2)
        window_length = window_length if window_length % 2 != 0 else window_length + 1
        
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 == 1 else max(1, len(y) - 1)

    try:
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=0)
        
        dy_dx = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1, delta=delta_x)
        

        smooth_size = max(5, int(len(x) / 55)) 
        if smooth_size % 2 == 0: 
            smooth_size += 1
            
        dy_dx = uniform_filter1d(dy_dx, size=smooth_size)
        
        d2y_dx2 = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=2, delta=delta_x)
        d2y_dx2 = uniform_filter1d(d2y_dx2, size=smooth_size)

    except Exception:
        y_smooth = savgol_filter(y, window_length=max(3, min(len(y), 5)), polyorder=1)
        dy_dx = np.gradient(y_smooth, x)
        d2y_dx2 = np.gradient(dy_dx, x)
        if len(y) >= 5:
            dy_dx = uniform_filter1d(dy_dx, size=5)
            d2y_dx2 = uniform_filter1d(d2y_dx2, size=5)

    return np.asarray(y_smooth), np.asarray(dy_dx), np.asarray(d2y_dx2), window_length

def find_extrema_from_derivative(x, dy_dx):
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


def compute_zero_crossing_x(x, dy_dx, idx):
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


def find_top_two_peaks(x, y, peaks):
    if len(peaks) < 2:
        raise ValueError("Не найдено достаточно пиков. Попробуйте увеличить окно сглаживания.")
    
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


def find_valley_between_peaks(x, y_smooth, peak1_idx, peak2_idx):
    if peak2_idx <= peak1_idx:
        start, end = min(peak1_idx, peak2_idx), max(peak1_idx, peak2_idx)
    else:
        start, end = peak1_idx, peak2_idx

    slice_vals = y_smooth[start:end+1]
    if slice_vals.size == 0:
        return int(start)

    local_min = int(np.argmin(slice_vals))
    return int(start + local_min)


def calculate_difference(x, y_smooth, peak1_idx, peak2_idx, valley_idx):
    x1, y1 = x[peak1_idx], y_smooth[peak1_idx]
    x2, y2 = x[peak2_idx], y_smooth[peak2_idx]

    x_valley = x[valley_idx]
    y_valley = y_smooth[valley_idx]
    
    y_line_at_v = np.interp(x_valley, [x1, x2], [y1, y2])

    difference = y_line_at_v - y_valley
    return difference


def save_raw_plot(x, y, y_smooth, dy_dx, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', alpha=0.5, linewidth=1.5, label='Исходный спектр')
    
    try:
        if peak1_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak1_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax.plot(x0, y0, 'ro', markersize=8, label='Пик')
        if peak2_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak2_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax.plot(x0, y0, 'ro', markersize=8)
        if valley_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax.plot(x0, y0, 'go', markersize=8, label='Долина')
    except Exception:
        if peak1_idx is not None: ax.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=8)
        if peak2_idx is not None: ax.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=8)
        if valley_idx is not None: ax.plot(x[valley_idx], y_smooth[valley_idx], 'go', markersize=8)

    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel('R, %', fontsize=12)
    ax.set_title('Исходный спектр (с найденными экстремумами)')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'raw_with_points.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_geometry_plot(x, y, y_smooth, dy_dx, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Исходный')
    ax.plot(x, y_smooth, 'r-', linewidth=1.5, label='Сглаженный спектр', alpha=0.8)
    
    if peak1_idx is not None and peak2_idx is not None:
        ax.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=8, label='Пик')
        ax.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=8)

        x_line = np.array([x[peak1_idx], x[peak2_idx]])
        y_line = np.array([y_smooth[peak1_idx], y_smooth[peak2_idx]])
        ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.8, label='Базовая линия')

        if valley_idx is not None:
            xv = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            yv = float(np.interp(xv, x, y_smooth))
            ax.plot(xv, yv, 'go', markersize=8, label='Долина')

            y_line_at_v = np.interp(xv, x_line, y_line)
            ax.vlines(xv, yv, y_line_at_v, colors='darkred', linestyles='-', linewidth=2.5)
            
            width = (x.max() - x.min()) * 0.015
            ax.hlines([yv, y_line_at_v], xv - width, xv + width, colors='darkred', linewidth=1.5)
            
            ax.text(xv + width*2, (yv + y_line_at_v)/2, r'$\Delta R$', 
                     color='darkred', fontsize=14, fontweight='bold', va='center')

            diff = y_line_at_v - yv
            p1_y = y_smooth[peak1_idx]
            q_val = (diff / p1_y * 100) if p1_y != 0 else 0
            
            info_text = f"$\\Delta R$: {diff:.4f}\n$Q$: {q_val:.2f}%"
            ax.text(0.05, 0.05, info_text, transform=ax.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel('R, %', fontsize=12)
    ax.set_title('Сглаженный спектр')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'smoothed_geometry.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_first_derivative_plot(x, dy_dx, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, dy_dx, 'g-', linewidth=1.5, label='Первая производная')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    try:
        if peak1_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak1_idx))
            ax.plot(x0, 0.0, 'ro', markersize=6)
        if peak2_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak2_idx))
            ax.plot(x0, 0.0, 'ro', markersize=6)
        if valley_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            ax.plot(x0, 0.0, 'go', markersize=6)
    except Exception:
        if peak1_idx is not None: ax.plot(x[peak1_idx], dy_dx[peak1_idx], 'ro', markersize=6)
        if peak2_idx is not None: ax.plot(x[peak2_idx], dy_dx[peak2_idx], 'ro', markersize=6)
        if valley_idx is not None: ax.plot(x[valley_idx], dy_dx[valley_idx], 'go', markersize=6)
    
    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel(r'$\frac{dR}{d\lambda}$', fontsize=12)
    ax.set_title('Первая производная')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'first_derivative.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_second_derivative_plot(x, d2y_dx2, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, d2y_dx2, 'm-', linewidth=1.5, label='Вторая производная')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    if peak1_idx is not None:
        ax.plot(x[peak1_idx], d2y_dx2[peak1_idx], 'ro', markersize=6)
    if peak2_idx is not None:
        ax.plot(x[peak2_idx], d2y_dx2[peak2_idx], 'ro', markersize=6)
    if valley_idx is not None:
        ax.plot(x[valley_idx], d2y_dx2[valley_idx], 'go', markersize=6)
    
    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel(r'$\frac{d^2R}{d\lambda^2}$', fontsize=12)
    ax.set_title('Вторая производная')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'second_derivative.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_results(x, y, y_smooth, dy_dx, d2y_dx2, peak1_idx=None, peak2_idx=None, valley_idx=None, 
                 peak1_idx_orig=None, peak2_idx_orig=None, valley_idx_orig=None, output_file=None, show_plot=False, verbose=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Анализ спектра', fontsize=16, fontweight='bold')
    
    diff = None
    q_val = None
    
    ax1 = axes[0, 0]
    ax1.plot(x, y, 'b-', alpha=0.5, linewidth=1, label='Исходный спектр')
    try:
        if peak1_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak1_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax1.plot(x0, y0, 'ro', markersize=8)
        if peak2_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak2_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax1.plot(x0, y0, 'ro', markersize=8)
        if valley_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            y0 = float(np.interp(x0, x, y_smooth))
            ax1.plot(x0, y0, 'go', markersize=8)
    except Exception:
        pass
    ax1.set_title('Исходный спектр')
    ax1.set_xlabel('λ, нм', fontsize=11)
    ax1.set_ylabel('R, %', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Исходный')
    ax2.plot(x, y_smooth, 'r-', linewidth=1.5, label='Сглаженный спектр', alpha=0.8)
    
    if peak1_idx is not None and peak2_idx is not None:
        ax2.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=8, label='Пик')
        ax2.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=8)

        x_line = np.array([x[peak1_idx], x[peak2_idx]])
        y_line = np.array([y_smooth[peak1_idx], y_smooth[peak2_idx]])
        ax2.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.8, label='Базовая линия')

        if valley_idx is not None:
            xv = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            yv = float(np.interp(xv, x, y_smooth))
            ax2.plot(xv, yv, 'go', markersize=8, label='Долина')

            y_line_at_v = np.interp(xv, x_line, y_line)
            ax2.vlines(xv, yv, y_line_at_v, colors='darkred', linestyles='-', linewidth=2.5)
            
            width = (x.max() - x.min()) * 0.015
            ax2.hlines([yv, y_line_at_v], xv - width, xv + width, colors='darkred', linewidth=1.5)
            
            ax2.text(xv + width*2, (yv + y_line_at_v)/2, r'$\Delta R$', 
                     color='darkred', fontsize=14, fontweight='bold', va='center')

            diff = y_line_at_v - yv
            p1_y = y_smooth[peak1_idx]
            q_val = (diff / p1_y * 100) if p1_y != 0 else 0
            
            info_text = f"$\\Delta R$: {diff:.4f}\n$Q$: {q_val:.2f}%"
            ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax2.set_title('Сглаженный спектр')
    ax2.set_xlabel('λ, нм', fontsize=11)
    ax2.set_ylabel('R, %', fontsize=11)
    ax2.legend(loc='upper right', fontsize='x-small')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(x, dy_dx, 'g-', label='1-я производная')
    ax3.axhline(0, color='black', lw=1, ls='--')
    try:
        if peak1_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak1_idx))
            ax3.plot(x0, 0.0, 'ro', markersize=6, label='Пик')
        if peak2_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(peak2_idx))
            ax3.plot(x0, 0.0, 'ro', markersize=6)
        if valley_idx is not None:
            x0 = compute_zero_crossing_x(x, dy_dx, int(valley_idx))
            ax3.plot(x0, 0.0, 'go', markersize=6, label='Долина')
    except Exception:
        pass
            
    ax3.set_xlabel('λ, нм', fontsize=11)
    ax3.set_ylabel(r'$\frac{dR}{d\lambda}$', fontsize=14)
    ax3.set_title('Первая производная')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.axis('off')  
    
    if diff is not None and q_val is not None:
        text_str = f"$\\Delta R = {diff:.4f}$\n\n$Q = {q_val:.2f}\\%$"
        ax4.text(0.5, 0.5, text_str, transform=ax4.transAxes, 
                 fontsize=40, ha='center', va='center', 
                 fontweight='bold', color='#212529')
    else:
        ax4.text(0.5, 0.5, "Пики или долина\nне найдены", transform=ax4.transAxes, 
                 fontsize=28, ha='center', va='center', color='darkred', fontweight='bold')
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()


def analyze_single_file(input_file, output_dir, window_arg='auto', polyorder=3, no_plot=False, verbose=True):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(input_file)
        if df.shape[1] < 2:
            raise ValueError("CSV файл должен содержать как минимум 2 столбца")
        
        x = np.array(df.iloc[:, 0].values, dtype=float)
        y = np.array(df.iloc[:, 1].values, dtype=float)
        
        if verbose:
            print(f"\nЗагружено {len(x)} точек")
            print(f"Диапазон абсцисс: [{x.min()}, {x.max()}]")
            print(f"Диапазон ординат: [{y.min():.4f}, {y.max():.4f}]")
        
        if str(window_arg).lower() == 'auto':
            if verbose: print("\nЗапуск автоподбора окна (Черновой проход)...")
            window_nm = calculate_auto_window_nm(x, y, polyorder=polyorder)
            if verbose: print(f"Оптимальное окно вычислено: {window_nm:.2f} нм")
        else:
            try:
                window_nm = float(window_arg)
            except ValueError:
                window_nm = 15.0
                if verbose: print("Внимание: окно задано некорректно, используем 15.0 нм")

        if verbose:
            print(f"\nЧистовое сглаживание спектра (окно {window_nm:.2f} нм)...")
            
        y_smooth, dy_dx, d2y_dx2, actual_window_pts = calculate_all_physics(x, y, window_nm=window_nm, polyorder=polyorder)
        
        if verbose:
            print(f"Это окно составило {actual_window_pts} точек.")
        
        peak1_idx = None
        peak2_idx = None
        valley_idx = None
        peak1_idx_orig = None
        peak2_idx_orig = None
        valley_idx_orig = None
        difference = None
        q_percent = None
        
        if verbose:
            print("Поиск экстремумов через нули первой производной...")
        peaks, valleys = find_extrema_from_derivative(x, dy_dx)
        if len(peaks) >= 2:
            try:
                peak1_idx, peak2_idx = find_top_two_peaks(x, y_smooth, peaks)                
                valley_idx = find_valley_between_peaks(x, y_smooth, peak1_idx, peak2_idx)
                if valley_idx is not None:
                    difference = calculate_difference(x, y_smooth, peak1_idx, peak2_idx, valley_idx)
                    p1_coords = (x[peak1_idx], y_smooth[peak1_idx])
                    p2_coords = (x[peak2_idx], y_smooth[peak2_idx])
                    peaks_sorted_by_x = sorted([p1_coords, p2_coords], key=lambda p: p[0])
                    
                    r_peak1 = peaks_sorted_by_x[0][1]
                    q_percent = (difference / r_peak1 * 100) if r_peak1 != 0 else 0
                    
                    if verbose:
                        print(f"Пик 1 (левый): x={peaks_sorted_by_x[0][0]:.2f}, y={peaks_sorted_by_x[0][1]:.4f}")
                        print(f"Результат: Delta R = {difference:.6f}, Q = {q_percent:.2f}%")

                peak1_val = float(x[peak1_idx])
                peak2_val = float(x[peak2_idx])
                valley_val = float(x[valley_idx]) if valley_idx is not None else None
                
                peak1_idx_orig = int(np.argmin(np.abs(x - peak1_val)))
                peak2_idx_orig = int(np.argmin(np.abs(x - peak2_val)))
                if valley_val is not None:
                    valley_idx_orig = int(np.argmin(np.abs(x - valley_val)))
                
            except Exception as e:
                if verbose:
                    print(f"Предупреждение: не удалось обработать экстремумы: {e}")
        else:
            if verbose:
                print("Предупреждение: не найдено достаточно пиков (нужно минимум 2).")
        
        if verbose:
            print("\nСохранение графиков...")
            
        save_raw_plot(x, y, y_smooth, dy_dx, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        save_geometry_plot(x, y, y_smooth, dy_dx, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        save_first_derivative_plot(x, dy_dx, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        save_second_derivative_plot(x, d2y_dx2, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        
        combined_output = output_dir / 'combined.png'
        plot_results(x, y, y_smooth, dy_dx, d2y_dx2, 
                    peak1_idx, peak2_idx, valley_idx,
                    peak1_idx_orig, peak2_idx_orig, valley_idx_orig,
                    output_file=str(combined_output), show_plot=not no_plot, verbose=verbose)
        
        result_data = {
            "input_file": str(input_file),
            "output_directory": str(output_dir),
            "status": "success",
            "data_info": {
                "num_points": int(len(x)),
                "x_range": [float(x.min()), float(x.max())],
                "y_range": [float(y.min()), float(y.max())]
            },
            "smoothing_parameters": {
                "window_arg": str(window_arg),
                "window_nm_used": float(window_nm),
                "actual_window_points": int(actual_window_pts), 
                "polyorder": int(polyorder)
            },
            "extrema": {},
            "result": {}
        }
        
        if peak1_idx is not None and peak2_idx is not None:
            p1_x, p1_y = float(x[peak1_idx]), float(y_smooth[peak1_idx])
            p2_x, p2_y = float(x[peak2_idx]), float(y_smooth[peak2_idx])
            
            result_data["extrema"] = {
                "peak1": {"x": p1_x, "y": p1_y, "derivative": float(dy_dx[peak1_idx])},
                "peak2": {"x": p2_x, "y": p2_y, "derivative": float(dy_dx[peak2_idx])}
            }
            if valley_idx is not None:
                result_data["extrema"]["valley"] = {
                    "x": float(x[valley_idx]), "y": float(y_smooth[valley_idx]), "derivative": float(dy_dx[valley_idx])
                }

        if difference is not None:
            result_data["result"] = {
                "success": True,
                "difference": float(difference),
                "Q_percent": float(q_percent) if q_percent is not None else None,
                "description": "Delta R и коэффициент Q (отношение Delta R к высоте первого пика)"
            }
        else:
            result_data["result"] = {
                "success": False,
                "error": "Не удалось вычислить геометрические параметры"
            }
            
        return result_data
        
    except Exception as e:
        return {
            "input_file": str(input_file),
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

