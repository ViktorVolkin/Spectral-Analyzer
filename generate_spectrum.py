import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Параметры шума
noise_level = 0.1  # небольшой шум в процентах

# Определяем контрольные точки спектра
# (wavelength, reflectance)
control_points = [
    (300, 35),   # начало
    (320, 34),   # конец первого интервала
    (420, 43),   # пик 1
    (480, 40),   # долина
    (520, 41),   # пик 2
    (700, 37),   # конец пятого интервала
    (800, 39),   # конец
]

# Извлекаем координаты контрольных точек
w_control = np.array([p[0] for p in control_points])
r_control = np.array([p[1] for p in control_points])

# Создаем кубический сплайн для плавной интерполяции
# Используем граничные условия для плавности
spline = CubicSpline(w_control, r_control, bc_type='natural')  # natural - вторая производная = 0 на концах

# Генерируем равномерную сетку точек
wavelengths = np.linspace(300, 800, 501)  # 501 точка для хорошего разрешения

# Вычисляем значения через сплайн
reflectance = spline(wavelengths)

# Добавляем небольшой шум
reflectance += np.random.normal(0, noise_level, len(reflectance))

# Создаем DataFrame
df = pd.DataFrame({
    'wavelength': wavelengths,
    'reflectance': reflectance
})

# Сортируем по длине волны (на всякий случай)
df = df.sort_values('wavelength').reset_index(drop=True)

# Сохраняем в CSV
df.to_csv('data/generated_spectrum.csv', index=False)

print(f"Сгенерировано {len(df)} точек")
print(f"Диапазон длин волн: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} нм")
print(f"Диапазон отражения: {df['reflectance'].min():.2f} - {df['reflectance'].max():.2f}%")
print(f"\nКонтрольные точки:")
for w, r in zip(w_control, r_control):
    actual_r = df[df['wavelength'] == w]['reflectance'].values
    if len(actual_r) > 0:
        print(f"  {w} нм: задано {r}%, получено {actual_r[0]:.2f}%")
print(f"\nПервые 10 точек:")
print(df.head(10))
print(f"\nПоследние 10 точек:")
print(df.tail(10))

