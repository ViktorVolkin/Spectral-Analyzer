import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import UnivariateSpline
import argparse
import sys
import os
import json
from pathlib import Path

# 'diff' from sympy не используется в модуле — импорт убран


def smooth_spectrum(x, y, window_length=51, polyorder=3):
    """
    Сглаживание спектра с помощью фильтра Савицкого-Голея.
    
    Parameters:
    -----------
    x : array-like
        Абсциссы (например, длина волны)
    y : array-like
        Ординаты (например, коэффициент отражения)
    window_length : int
        Длина окна для сглаживания (должна быть нечетной)
    polyorder : int
        Порядок полинома для сглаживания
    
    Returns:
    --------
    y_smooth : array
        Сглаженные значения ординат
    """
    # Убеждаемся, что window_length нечетное и меньше длины данных
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
    
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth


def find_extrema_from_derivative(x, dy_dx):
    """
    Надёжный поиск экстремумов по первой производной.
    Обрабатываются участки с нулевой производной (flat regions). При
    переходе знака регистрируется индекс перехода (дискретная позиция).
    """
    peaks = []
    valleys = []

    # Берём знак производной и корректируем нулевые значения
    s = np.sign(dy_dx.astype(float))
    if np.any(s == 0):
        # forward fill
        for i in range(1, len(s)):
            if s[i] == 0 and s[i-1] != 0:
                s[i] = s[i-1]
        # backward fill
        for i in range(len(s)-2, -1, -1):
            if s[i] == 0 and s[i+1] != 0:
                s[i] = s[i+1]

    # Находим переходы знака: + -> - (максимум), - -> + (минимум)
    for i in range(1, len(s)):
        # При переходе знака регистрируем индекс i (позиция нулевого пересечения в дискретном массиве)
        if s[i-1] > 0 and s[i] < 0:
            peaks.append(int(i))
        elif s[i-1] < 0 and s[i] > 0:
            valleys.append(int(i))

    peaks = np.unique(peaks)
    valleys = np.unique(valleys)

    # Фильтрация по минимальному расстоянию между соседними экстремумами
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
    """
    Находит два верхних экстремума (пика).
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    y : array-like
        Ординаты
    peaks : array
        Индексы локальных максимумов
    
    Returns:
    --------
    peak1_idx : int
        Индекс первого пика (левее)
    peak2_idx : int
        Индекс второго пика (правее)
    """
    if len(peaks) < 2:
        raise ValueError("Не найдено достаточно пиков. Попробуйте изменить параметры сглаживания.")
    
    # Сортируем пики по высоте (убывание)
    peak_values = y[peaks]
    sorted_idx = np.argsort(peak_values)[::-1]

    first = int(peaks[sorted_idx[0]])
    # Минимальное разделение между пиками (как доля от длины сигнала)
    min_sep = max(int(0.05 * len(x)), 2)

    second = None
    for idx in sorted_idx[1:]:
        cand = int(peaks[idx])
        if abs(cand - first) >= min_sep:
            second = cand
            break

    # Если не найдено достаточно удалённого пика — берём второй по высоте
    if second is None:
        second = int(peaks[sorted_idx[1]])

    # Уточнение: для каждого из найденных пиков берём окно ±15 точек
    # и перемещаемся к истинному максимуму по сглаженной кривой (np.argmax)
    w = 15
    def snap(idx):
        start = max(0, idx - w)
        end = min(len(y) - 1, idx + w)
        local = y[start:end+1]
        if local.size == 0:
            return int(idx)
        return int(start + np.argmax(local))

    first = snap(first)
    second = snap(second)

    peak1_idx, peak2_idx = (first, second) if x[first] < x[second] else (second, first)
    return int(peak1_idx), int(peak2_idx)


def find_valley_between_peaks(x, y_smooth, peak1_idx, peak2_idx):
    """
    Находит индекс долины между двумя пиками строго на сглаженной кривой.

    Логика: формируем срез y_smooth[peak1_idx:peak2_idx+1] и возвращаем индекс
    глобального минимума внутри этого интервала. Это гарантирует, что мы
    получим точную точку "дна" на красной (сглаженной) кривой.
    """
    if peak2_idx <= peak1_idx:
        # Защита от неверного порядка
        start, end = min(peak1_idx, peak2_idx), max(peak1_idx, peak2_idx)
    else:
        start, end = peak1_idx, peak2_idx

    # Включаем пик2 в срез => +1
    slice_vals = y_smooth[start:end+1]
    if slice_vals.size == 0:
        return int(start)

    local_min = int(np.argmin(slice_vals))
    return int(start + local_min)



def calculate_difference(x, y_smooth, peak1_idx, peak2_idx, valley_idx):
    """
    Вычисляет разность ординат между нижним экстремумом и точкой 
    с той же абсциссой на прямой, соединяющей два пика.
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    y : array-like
        Ординаты
    peak1_idx : int
        Индекс первого пика
    peak2_idx : int
        Индекс второго пика
    valley_idx : int
        Индекс долины
    
    Returns:
    --------
    difference : float
        Разность ординат
    """
    # Координаты пиков (используем значения для пиков, обычно сглаженные)
    x1, y1 = x[peak1_idx], y_smooth[peak1_idx]
    x2, y2 = x[peak2_idx], y_smooth[peak2_idx]

    # Координаты долины по сглаженной кривой (синхронизация по X)
    x_valley = x[valley_idx]
    y_valley = y_smooth[valley_idx]
    
    # Уравнение прямой через два пика: y = k*x + b
    k = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    b = y1 - k * x1
    # Значение хордовой линии (между пиками) в X_v: используем интерполяцию по (x1,x2)
    y_line_at_v = np.interp(x_valley, [x1, x2], [y1, y2])

    # Разность ординат (хордовая линия по сглаженным пикам минус сглаженная глубина в точке долины)
    difference = y_line_at_v - y_valley
    
    return difference


def calculate_derivatives(x, y, smooth_window=None):
    """
    Вычисляет первую и вторую производные по сглаженному спектру.
    Использует сплайны для получения гладких непрерывных производных.
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    y : array-like
        Ординаты (сглаженный спектр)
    smooth_window : int, optional
        Параметр сглаживания для сплайна (по умолчанию автоматически)
    
    Returns:
    --------
    dy_dx : array
        Первая производная (гладкая)
    d2y_dx2 : array
        Вторая производная (гладкая)
    """
    # Используем сплайн для получения гладких производных
    # Сплайн автоматически обеспечивает непрерывность производных
    try:
        # Определяем параметр сглаживания
        if smooth_window is None:
            # Используем небольшое сглаживание для сохранения формы
            s = len(y) * np.var(y) * 0.01  # небольшой параметр сглаживания
        else:
            s = smooth_window
        
        # Создаем сплайн
        spline = UnivariateSpline(x, y, s=s, k=3)  # кубический сплайн
        
        # Вычисляем производные через сплайн
        dy_dx = np.asarray(spline.derivative(n=1)(x))
        d2y_dx2 = np.asarray(spline.derivative(n=2)(x))
        
    except Exception as e:
        # Fallback на метод с np.gradient и сглаживанием
        dy_dx = np.gradient(y, x)
        d2y_dx2 = np.gradient(dy_dx, x)
        
        # Применяем сглаживание к производным для непрерывности
        if smooth_window is None:
            smooth_window = max(5, min(21, len(y) // 10))
            if smooth_window % 2 == 0:
                smooth_window += 1
        
        if len(y) >= smooth_window:
            try:
                dy_dx = savgol_filter(dy_dx, smooth_window, 3)
                d2y_dx2 = savgol_filter(d2y_dx2, smooth_window, 3)
            except:
                dy_dx = uniform_filter1d(dy_dx, size=smooth_window)
                d2y_dx2 = uniform_filter1d(d2y_dx2, size=smooth_window)
        
        dy_dx = np.asarray(dy_dx)
        d2y_dx2 = np.asarray(d2y_dx2)
    
    return dy_dx, d2y_dx2


def save_raw_plot(x, y, output_dir, verbose=True):
    """
    Сохраняет исходный график.
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    y : array-like
        Исходные ординаты
    output_dir : str
        Директория для сохранения
    verbose : bool
        Выводить сообщения
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=1.5, label='Исходный спектр')
    ax.set_xlabel('Абсцисса')
    ax.set_ylabel('Ордината')
    ax.set_title('Исходный спектр')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Сохранен график: {os.path.join(output_dir, 'raw.png')}")


def save_smoothed_plot(x, y, y_smooth, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    """
    Сохраняет график с исходным и сглаженным спектром с выделенными точками.
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    y : array-like
        Исходные ординаты
    y_smooth : array-like
        Сглаженные ординаты
    output_dir : str
        Директория для сохранения
    peak1_idx : int, optional
        Индекс первого пика
    peak2_idx : int, optional
        Индекс второго пика
    valley_idx : int, optional
        Индекс долины
    verbose : bool, optional
        Выводить сообщения
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', alpha=0.5, linewidth=1, label='Исходный спектр')
    ax.plot(x, y_smooth, 'r-', linewidth=2, label='Сглаженный спектр')
    
    if peak1_idx is not None and peak2_idx is not None:
        ax.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=2, 
                label=f'Пик 1 ({x[peak1_idx]:.2f}, {y_smooth[peak1_idx]:.4f})')
        ax.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=2, 
                label=f'Пик 2 ({x[peak2_idx]:.2f}, {y_smooth[peak2_idx]:.4f})')
        
        # Рисуем прямую между пиками
        x_line = np.array([x[peak1_idx], x[peak2_idx]])
        y_line = np.array([y_smooth[peak1_idx], y_smooth[peak2_idx]])
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.5, label='Прямая между пиками')
    
    if valley_idx is not None:
        ax.plot(x[valley_idx], y_smooth[valley_idx], 'go', markersize=2, 
                label=f'Долина ({x[valley_idx]:.2f}, {y_smooth[valley_idx]:.4f})')
    
    ax.set_xlabel('Абсцисса')
    ax.set_ylabel('Ордината')
    ax.set_title('Исходный и сглаженный спектр с выделенными точками')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Сохранен график: {os.path.join(output_dir, 'smoothed.png')}")


def save_first_derivative_plot(x, dy_dx, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    """
    Сохраняет график первой производной (вычисленной по сглаженному спектру).
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    dy_dx : array
        Первая производная (по сглаженному спектру)
    output_dir : str
        Директория для сохранения
    peak1_idx : int, optional
        Индекс первого пика
    peak2_idx : int, optional
        Индекс второго пика
    valley_idx : int, optional
        Индекс долины
    verbose : bool, optional
        Выводить сообщения
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, dy_dx, 'g-', linewidth=1.5, label='Первая производная')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    if peak1_idx is not None:
        ax.plot(x[peak1_idx], dy_dx[peak1_idx], 'ro', markersize=2)
    if peak2_idx is not None:
        ax.plot(x[peak2_idx], dy_dx[peak2_idx], 'ro', markersize=2)
    if valley_idx is not None:
        ax.plot(x[valley_idx], dy_dx[valley_idx], 'go', markersize=2)
    
    ax.set_xlabel('Абсцисса')
    ax.set_ylabel('dy/dx')
    ax.set_title('Первая производная')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'first_derivative.png', dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Сохранен график: {os.path.join(output_dir, 'first_derivative.png')}")


def save_second_derivative_plot(x, d2y_dx2, output_dir: str, peak1_idx=None, peak2_idx=None, valley_idx=None, verbose=True):
    """
    Сохраняет график второй производной (вычисленной по сглаженному спектру).
    
    Parameters:
    -----------
    x : array-like
        Абсциссы
    d2y_dx2 : array
        Вторая производная (по сглаженному спектру)
    output_dir : str
        Директория для сохранения
    peak1_idx : int, optional
        Индекс первого пика
    peak2_idx : int, optional
        Индекс второго пика
    valley_idx : int, optional
        Индекс долины
    verbose : bool, optional
        Выводить сообщения
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, d2y_dx2, 'm-', linewidth=1.5, label='Вторая производная')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    if peak1_idx is not None:
        ax.plot(x[peak1_idx], d2y_dx2[peak1_idx], 'ro', markersize=2)
    if peak2_idx is not None:
        ax.plot(x[peak2_idx], d2y_dx2[peak2_idx], 'ro', markersize=2)
    if valley_idx is not None:
        ax.plot(x[valley_idx], d2y_dx2[valley_idx], 'go', markersize=2)
    
    ax.set_xlabel('Абсцисса')
    ax.set_ylabel('d²y/dx²')
    ax.set_title('Вторая производная')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'second_derivative.png', dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Сохранен график: {os.path.join(output_dir, 'second_derivative.png')}")


def plot_results(x, y, y_smooth, dy_dx, d2y_dx2, peak1_idx=None, peak2_idx=None, valley_idx=None, 
                 peak1_idx_orig=None, peak2_idx_orig=None, valley_idx_orig=None, output_file=None, show_plot=False, verbose=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Анализ спектра отражения (Уточненный)', fontsize=16, fontweight='bold')
    
    # 1. Исходный график (Raw)
    ax1 = axes[0, 0]
    ax1.plot(x, y, 'b-', alpha=0.5, linewidth=1, label='Исходный спектр')
    # Маркеры на raw-окне рисуем по значениям сглаженной кривой, чтобы видно было согласование
    if peak1_idx is not None: ax1.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=8)
    if peak2_idx is not None: ax1.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=8)
    if valley_idx is not None: ax1.plot(x[valley_idx], y_smooth[valley_idx], 'go', markersize=8)
    ax1.set_title('Исходный спектр (область поиска)')
    ax1.grid(True, alpha=0.3)
    
    # 2. ГЕОМЕТРИЯ DELTA R (Основной график для заказчика)
    ax2 = axes[0, 1]
    ax2.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Исходный (шум)')
    ax2.plot(x, y_smooth, 'r-', linewidth=1.5, label='Сглаженный', alpha=0.8)
    
    if peak1_idx is not None and peak2_idx is not None:
        # Пики рисуем по сглаженной кривой — маркеры должны лежать на красной линии
        ax2.plot(x[peak1_idx], y_smooth[peak1_idx], 'ro', markersize=8, label='Пик (смягчённый)')
        ax2.plot(x[peak2_idx], y_smooth[peak2_idx], 'ro', markersize=8)

        # Прямая между пиками — строим по сглаженным значениям (chord)
        x_line = np.array([x[peak1_idx], x[peak2_idx]])
        y_line = np.array([y_smooth[peak1_idx], y_smooth[peak2_idx]])
        ax2.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.8, label='Базовая линия (смягчённая)')

        if valley_idx is not None:
            # Точка долины — берём значение на сглаженной кривой
            xv = x[valley_idx]
            yv = y_smooth[valley_idx]
            ax2.plot(xv, yv, 'go', markersize=8, label='Долина (смягчённая)')

            # Находим значение хорды в Xv (интерполяция по сглаженным пикам)
            y_line_at_v = np.interp(xv, x_line, y_line)

            # Отрисовка Delta R — вертикальная линия в одной X-точке
            ax2.vlines(xv, yv, y_line_at_v, colors='darkred', linestyles='-', linewidth=2.5)
            
            # Засечки
            width = (x.max() - x.min()) * 0.015
            ax2.hlines([yv, y_line_at_v], xv - width, xv + width, colors='darkred', linewidth=1.5)
            
            # Метка Delta R
            ax2.text(xv + width*2, (yv + y_line_at_v)/2, r'$\Delta R$', 
                     color='darkred', fontsize=14, fontweight='bold', va='center')

            # Расчет значений для инфо-блока (всё по сглаженной кривой)
            diff = y_line_at_v - yv
            p1_y = y_smooth[peak1_idx]
            q_val = (diff / p1_y * 100) if p1_y != 0 else 0
            
            info_text = f"$Delta R$: {diff:.4f}\n$Q$: {q_val:.2f}%"
            ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax2.set_title(r'Геометрический расчет $\Delta R$ (по реальным точкам)')
    ax2.legend(loc='upper right', fontsize='x-small')
    ax2.grid(True, alpha=0.3)
    
    # 3. Первая производная (показываем, где математика нашла экстремумы)
    ax3 = axes[1, 0]
    ax3.plot(x, dy_dx, 'g-', label='1-я производная')
    ax3.axhline(0, color='black', lw=1, ls='--')
    # Рисуем маркеры на самом графике производной, используя реальные значения dy_dx
    if peak1_idx is not None:
        ax3.plot(x[peak1_idx], dy_dx[peak1_idx], 'ro', markersize=6, label='Пик (экстремум)')
    if peak2_idx is not None:
        ax3.plot(x[peak2_idx], dy_dx[peak2_idx], 'ro', markersize=6)
    if valley_idx is not None:
        ax3.plot(x[valley_idx], dy_dx[valley_idx], 'go', markersize=6, label='Долина (экстремум)')
    ax3.set_title('Математический поиск (Экстремумы)')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    # 4. Вторая производная
    ax4 = axes[1, 1]
    ax4.plot(x, d2y_dx2, 'm-', label='2-я производная')
    ax4.axhline(0, color='black', lw=1, ls='--')
    ax4.set_title('Вторая производная (Кривизна)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()



def analyze_single_file(input_file, output_dir, window=51, polyorder=3, no_plot=False, verbose=True):
    """
    Анализирует один CSV файл: сглаживает спектр, находит экстремумы, 
    вычисляет разность ординат (Delta R) и коэффициент Q.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Чтение данных из CSV
        df = pd.read_csv(input_file)
        if df.shape[1] < 2:
            raise ValueError("CSV файл должен содержать как минимум 2 столбца")
        
        # Предполагаем, что первый столбец - абсциссы, второй - ординаты
        x = np.array(df.iloc[:, 0].values, dtype=float)
        y = np.array(df.iloc[:, 1].values, dtype=float)
        
        if verbose:
            print(f"\nЗагружено {len(x)} точек")
            print(f"Диапазон абсцисс: [{x.min()}, {x.max()}]")
            print(f"Диапазон ординат: [{y.min():.4f}, {y.max():.4f}]")
            print("\nСохранение исходного графика...")
        
        # Сохраняем исходный график
        save_raw_plot(x, y, str(output_dir), verbose=verbose)
        
        # Сглаживание спектра
        if verbose:
            print("\nСглаживание спектра...")
        y_smooth = smooth_spectrum(x, y, window_length=window, polyorder=polyorder)
        
        # Вычисляем производные
        dy_dx, d2y_dx2 = calculate_derivatives(x, y_smooth)
        
        # Инициализируем переменные для экстремумов
        peak1_idx = None
        peak2_idx = None
        valley_idx = None
        peak1_idx_orig = None
        peak2_idx_orig = None
        valley_idx_orig = None
        difference = None
        q_percent = None
        
        # Поиск экстремумов через нули первой производной
        if verbose:
            print("\nПоиск экстремумов через нули первой производной...")
        peaks, valleys = find_extrema_from_derivative(x, dy_dx)
        if len(peaks) >= 2:
            try:
                # Находим два верхних пика (передаём сглаженную кривую для уточнения)
                peak1_idx, peak2_idx = find_top_two_peaks(x, y_smooth, peaks)                
                # Находим долину между пиками (по сглаженной кривой)
                valley_idx = find_valley_between_peaks(x, y_smooth, peak1_idx, peak2_idx)
                if valley_idx is not None:
                    # 1. Вычисляем разность ординат (Delta R) по сглаженным точкам
                    difference = calculate_difference(x, y_smooth, peak1_idx, peak2_idx, valley_idx)
                    # 2. Вычисляем коэффициент Q (согласно ТЗ заказчика)
                    # Определяем первый пик (самый левый по оси X)
                    p1_coords = (x[peak1_idx], y_smooth[peak1_idx])
                    p2_coords = (x[peak2_idx], y_smooth[peak2_idx])
                    peaks_sorted_by_x = sorted([p1_coords, p2_coords], key=lambda p: p[0])
                    
                    r_peak1 = peaks_sorted_by_x[0][1] # Высота первого пика (R2 в терминах ТЗ)
                    q_percent = (difference / r_peak1 * 100) if r_peak1 != 0 else 0
                    
                    if verbose:
                        print(f"Пик 1 (левый): x={peaks_sorted_by_x[0][0]:.2f}, y={peaks_sorted_by_x[0][1]:.4f}")
                        print(f"Результат: Delta R = {difference:.6f}, Q = {q_percent:.2f}%")

                # Находим соответствующие точки на исходном графике (для визуализации)
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
        
        # Сохраняем графики
        if verbose:
            print("\nСохранение графиков...")
        save_smoothed_plot(x, y, y_smooth, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        save_first_derivative_plot(x, dy_dx, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        save_second_derivative_plot(x, d2y_dx2, str(output_dir), peak1_idx, peak2_idx, valley_idx, verbose=verbose)
        
        combined_output = output_dir / 'combined.png'
        plot_results(x, y, y_smooth, dy_dx, d2y_dx2, 
                    peak1_idx, peak2_idx, valley_idx,
                    peak1_idx_orig, peak2_idx_orig, valley_idx_orig,
                    output_file=str(combined_output), show_plot=not no_plot, verbose=verbose)
        
        # Формируем итоговый словарь результата
        result_data = {
            "input_file": str(input_file),
            "output_directory": str(output_dir),
            "status": "success",
            "data_info": {
                "num_points": int(len(x)),
                "x_range": [float(x.min()), float(x.max())],
                "y_range": [float(y.min()), float(y.max())]
            },
            "smoothing_parameters": {"window_length": int(window), "polyorder": int(polyorder)},
            "extrema": {},
            "result": {}
        }
        
        if peak1_idx is not None and peak2_idx is not None:
            # Сортируем для JSON отчета
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

def main():
    parser = argparse.ArgumentParser(description='Анализ спектра отражения (Delta R и Q)')
    parser.add_argument('input_path', type=str, help='Путь к CSV файлу или директории')
    # Устанавливаем 11 как стандарт, так как для твоих данных это идеал
    parser.add_argument('--window', type=int, default=11, help='Окно сглаживания (default: 11)')
    parser.add_argument('--polyorder', type=int, default=3, help='Порядок полинома (default: 3)')
    parser.add_argument('--no-plot', action='store_true', help='Не показывать графики')
    
    args = parser.parse_args()
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Ошибка: путь '{input_path}' не существует", file=sys.stderr)
        sys.exit(1)
    
    # 1. ОБРАБОТКА ОДИНОЧНОГО ФАЙЛА
    if input_path.is_file():
        file_name = input_path.stem
        output_dir = Path('output') / file_name
        
        print(f"Анализ файла: {input_path}")
        result_data = analyze_single_file(str(input_path), output_dir, args.window, args.polyorder, args.no_plot, verbose=True)
        
        # Сохраняем JSON
        with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ АНАЛИЗА")
        print("="*50)
        if result_data.get("status") == "success" and result_data.get("result", {}).get("success"):
            res = result_data['result']
            print(f"Delta R (глубина): {res['difference']:.6f}")
            print(f"Q (отношение):     {res['Q_percent']:.2f}%")
        else:
            print(f"Ошибка или экстремумы не найдены: {result_data.get('error', 'Unknown error')}")
        print("="*50)

    # 2. ОБРАБОТКА ДИРЕКТОРИИ
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            print(f"В '{input_path}' нет CSV файлов", file=sys.stderr)
            sys.exit(1)
        
        print(f"Найдено {len(csv_files)} файлов. Начинаю обработку...")
        results = []
        
        for i, csv_file in enumerate(sorted(csv_files), 1):
            output_dir = Path('output') / csv_file.stem
            # При массовой обработке отключаем всплывающие окна графиков
            result_data = analyze_single_file(str(csv_file), output_dir, args.window, args.polyorder, no_plot=True, verbose=False)
            results.append(result_data)
            
            with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            # Короткий лог в консоль
            if result_data.get("status") == "success" and result_data.get("result", {}).get("success"):
                res = result_data['result']
                print(f"[{i}/{len(csv_files)}] ✅ {csv_file.name}: Delta R={res['difference']:.4f}, Q={res['Q_percent']:.2f}%")
            else:
                print(f"[{i}/{len(csv_files)}] ❌ {csv_file.name}: Ошибка анализа")

        # --- ГЕНЕРАЦИЯ СВОДНОЙ ТАБЛИЦЫ ---
        csv_summary_file = Path('output') / 'summary.csv'
        csv_rows = []
        for r in results:
            res_val = r.get("result", {})
            csv_rows.append({
                "Filename": Path(r.get("input_file")).name,
                "Status": r.get("status"),
                "Delta_R": res_val.get("difference", ""),
                "Q_percent": res_val.get("Q_percent", "")
            })
        
        pd.DataFrame(csv_rows).to_csv(csv_summary_file, index=False, encoding='utf-8')
        
        print("\n" + "="*70)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Сводная таблица для заказчика: {csv_summary_file}")
        print("="*70)



if __name__ == "__main__":
    main()

