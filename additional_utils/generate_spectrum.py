import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

noise_level = 0.1  


control_points = [
    (300, 35),   
    (320, 34), 
    (420, 43),  
    (480, 40),  
    (520, 41),   
    (700, 37),  
    (800, 39),  
]

w_control = np.array([p[0] for p in control_points])
r_control = np.array([p[1] for p in control_points])


spline = CubicSpline(w_control, r_control, bc_type='natural')

wavelengths = np.linspace(300, 800, 501)  

reflectance = spline(wavelengths)

reflectance += np.random.normal(0, noise_level, len(reflectance))

df = pd.DataFrame({
    'wavelength': wavelengths,
    'reflectance': reflectance
})

df = df.sort_values('wavelength').reset_index(drop=True)

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

