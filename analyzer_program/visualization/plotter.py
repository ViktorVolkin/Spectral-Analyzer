import matplotlib.pyplot as plt
import numpy as np
import config
from core.math_utils import compute_precise_x

def save_raw_plot(x, y, y_smooth, dy_dx, p1, p2, v, out_dir):
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    ax.plot(x, y, 'b-', alpha=0.5, linewidth=1.5, label='Исходный спектр')
    
    xv_p1 = compute_precise_x(x, dy_dx, p1)
    xv_p2 = compute_precise_x(x, dy_dx, p2)
    xv_v = compute_precise_x(x, dy_dx, v)
    
    ax.plot(xv_p1, np.interp(xv_p1, x, y_smooth), 'ro', markersize=8, label='Пик')
    ax.plot(xv_p2, np.interp(xv_p2, x, y_smooth), 'ro', markersize=8)
    ax.plot(xv_v, np.interp(xv_v, x, y_smooth), 'go', markersize=8, label='Долина')

    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel('R, %', fontsize=12)
    ax.set_title('Исходный спектр (с найденными экстремумами)')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'raw_with_points.png', dpi=config.DPI)
    plt.close()

def save_geometry_plot(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, out_dir):
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    ax.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Исходный')
    ax.plot(x, y_smooth, 'r-', linewidth=1.5, label='Сглаженный спектр', alpha=0.8)
    
    ax.plot(x[p1], y_smooth[p1], 'ro', markersize=8, label='Пик')
    ax.plot(x[p2], y_smooth[p2], 'ro', markersize=8)

    x_line = np.array([x[p1], x[p2]])
    y_line = np.array([y_smooth[p1], y_smooth[p2]])
    ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.8, label='Базовая линия')

    xv_v = compute_precise_x(x, dy_dx, v)
    yv = float(np.interp(xv_v, x, y_smooth))
    ax.plot(xv_v, yv, 'go', markersize=8, label='Долина')

    y_line_at_v = np.interp(xv_v, x_line, y_line)
    ax.vlines(xv_v, yv, y_line_at_v, colors='darkred', linestyles='-', linewidth=2.5)
    
    width = (x.max() - x.min()) * 0.015
    ax.hlines([yv, y_line_at_v], xv_v - width, xv_v + width, colors='darkred', linewidth=1.5)
    ax.text(xv_v + width*2, (yv + y_line_at_v)/2, r'$\Delta R$', color='darkred', fontsize=14, fontweight='bold', va='center')

    info_text = f"$\\Delta R$: {dr:.4f}\n$Q$: {q:.2f}%"
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel('R, %', fontsize=12)
    ax.set_title('Сглаженный спектр')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'smoothed_geometry.png', dpi=config.DPI)
    plt.close()

def save_first_derivative_plot(x, dy_dx, p1, p2, v, out_dir):
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    ax.plot(x, dy_dx, 'g-', linewidth=1.5, label='Первая производная')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.plot(compute_precise_x(x, dy_dx, p1), 0.0, 'ro', markersize=6)
    ax.plot(compute_precise_x(x, dy_dx, p2), 0.0, 'ro', markersize=6)
    ax.plot(compute_precise_x(x, dy_dx, v), 0.0, 'go', markersize=6)
    
    ax.set_xlabel('λ, нм', fontsize=12)
    ax.set_ylabel(r'$\frac{dR}{d\lambda}$', fontsize=12)
    ax.set_title('Первая производная')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'first_derivative.png', dpi=config.DPI)
    plt.close()

def save_combined_report(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Анализ спектра', fontsize=16, fontweight='bold')

    xv_p1 = compute_precise_x(x, dy_dx, p1)
    xv_p2 = compute_precise_x(x, dy_dx, p2)
    xv_v = compute_precise_x(x, dy_dx, v)

    ax1 = axes[0, 0]
    ax1.plot(x, y, 'b-', alpha=0.5, linewidth=1, label='Исходный спектр')
    ax1.plot(xv_p1, np.interp(xv_p1, x, y_smooth), 'ro', markersize=8)
    ax1.plot(xv_p2, np.interp(xv_p2, x, y_smooth), 'ro', markersize=8)
    ax1.plot(xv_v, np.interp(xv_v, x, y_smooth), 'go', markersize=8)
    ax1.set_title('Исходный спектр')
    ax1.set_xlabel('λ, нм', fontsize=11)
    ax1.set_ylabel('R, %', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize='x-small')

    ax2 = axes[0, 1]
    ax2.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Исходный')
    ax2.plot(x, y_smooth, 'r-', linewidth=1.5, label='Сглаженный спектр', alpha=0.8)
    ax2.plot(x[p1], y_smooth[p1], 'ro', markersize=8, label='Пик')
    ax2.plot(x[p2], y_smooth[p2], 'ro', markersize=8)

    x_line = np.array([x[p1], x[p2]])
    y_line = np.array([y_smooth[p1], y_smooth[p2]])
    ax2.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.8, label='Базовая линия')

    yv = float(np.interp(xv_v, x, y_smooth))
    ax2.plot(xv_v, yv, 'go', markersize=8, label='Долина')

    y_line_at_v = np.interp(xv_v, x_line, y_line)
    ax2.vlines(xv_v, yv, y_line_at_v, colors='darkred', linestyles='-', linewidth=2.5)
    
    width = (x.max() - x.min()) * 0.015
    ax2.hlines([yv, y_line_at_v], xv_v - width, xv_v + width, colors='darkred', linewidth=1.5)
    ax2.text(xv_v + width*2, (yv + y_line_at_v)/2, r'$\Delta R$', color='darkred', fontsize=14, fontweight='bold', va='center')

    info_text = f"$\\Delta R$: {dr:.4f}\n$Q$: {q:.2f}%"
    ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    ax2.set_title('Сглаженный спектр')
    ax2.set_xlabel('λ, нм', fontsize=11)
    ax2.set_ylabel('R, %', fontsize=11)
    ax2.legend(loc='upper right', fontsize='x-small')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(x, dy_dx, 'g-', label='1-я производная')
    ax3.axhline(0, color='black', lw=1, ls='--')
    ax3.plot(xv_p1, 0.0, 'ro', markersize=6, label='Пик')
    ax3.plot(xv_p2, 0.0, 'ro', markersize=6)
    ax3.plot(xv_v, 0.0, 'go', markersize=6, label='Долина')
        
    ax3.set_xlabel('λ, нм', fontsize=11)
    ax3.set_ylabel(r'$\frac{dR}{d\lambda}$', fontsize=14)
    ax3.set_title('Первая производная')
    ax3.legend(fontsize='small', loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis('off') 
    
    text_str = rf"$\Delta R = {dr:.4f}$" + "\n\n" + rf"$Q = {q:.2f}\%$"
    
    ax4.text(0.5, 0.5, text_str, transform=ax4.transAxes, fontsize=40, ha='center', va='center', fontweight='bold', color='#212529')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_dir / "combined_report.png", dpi=config.DPI)
    plt.close()

def generate_all_plots(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, file_name):
    """Вызывает отрисовку 4 картинок (без второй производной)"""
    out_dir = config.OUTPUT_DIR / file_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_raw_plot(x, y, y_smooth, dy_dx, p1, p2, v, out_dir)
    save_geometry_plot(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, out_dir)
    save_first_derivative_plot(x, dy_dx, p1, p2, v, out_dir)
    save_combined_report(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, out_dir)