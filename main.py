import argparse
import sys
from pathlib import Path
import config
from core.data_loader import load_spectral_data
from core.processor import calculate_auto_window_nm, process_spectrum
from core.math_utils import (find_extrema_from_derivative, find_top_two_peaks, find_valley_between_peaks, calculate_physics)
from visualization.plotter import generate_all_plots
from core.exporter import save_json_report, save_summary_csv

def process_single_file(file_path, window_arg, no_plot):
    """Полный цикл обработки одного файла с возвратом словаря результатов"""
    result_dict = {
        "input_file": str(file_path),
        "status": "error",
        "smoothing_parameters": {"window_arg": str(window_arg)},
        "result": {}
    }
    
    try:
        x, y = load_spectral_data(file_path)
        
        result_dict["data_info"] = {
            "num_points": int(len(x)),
            "x_range": [float(x.min()), float(x.max())],
            "y_range": [float(y.min()), float(y.max())]
        }
        
        if str(window_arg).lower() == 'auto':
            win_nm = calculate_auto_window_nm(x, y)
        else:
            win_nm = float(window_arg)
            
        result_dict["smoothing_parameters"]["window_nm_used"] = win_nm
        
        y_smooth, dy_dx = process_spectrum(x, y, win_nm)
        peaks, _ = find_extrema_from_derivative(x, dy_dx)
        
        if len(peaks) >= 2:
            p1, p2 = find_top_two_peaks(x, y_smooth, peaks)
            v = find_valley_between_peaks(y_smooth, p1, p2)
            dr, q = calculate_physics(x, y_smooth, p1, p2, v)
            
            result_dict["extrema"] = {
                "peak1": {"x": float(x[p1]), "y": float(y_smooth[p1])},
                "peak2": {"x": float(x[p2]), "y": float(y_smooth[p2])},
                "valley": {"x": float(x[v]), "y": float(y_smooth[v])}
            }
            
            result_dict["status"] = "success"
            result_dict["result"] = {"difference": dr, "Q_percent": q}
            
            output_dir = config.OUTPUT_DIR / file_path.stem
            
            if not no_plot:
                generate_all_plots(x, y, y_smooth, dy_dx, p1, p2, v, dr, q, file_path.stem)
                
            save_json_report(result_dict, output_dir)
            
            return result_dict
        else:
            result_dict["error"] = "Недостаточно пиков"
            return result_dict
            
    except Exception as e:
        result_dict["error"] = str(e)
        return result_dict


def main():
    parser = argparse.ArgumentParser(description='Анализ спектра отражения (Delta R и Q)')
    parser.add_argument('input_path', type=str, help='Путь к CSV файлу или директории')
    parser.add_argument('--window', type=str, default='auto', help='Ширина окна в нм или "auto" (default: auto)')
    parser.add_argument('--polyorder', type=int, default=3, help='Порядок полинома (default: 3)')
    parser.add_argument('--no-plot', action='store_true', help='Не показывать и не сохранять графики')
    
    args = parser.parse_args()
    
    config.POLYORDER = args.polyorder
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Ошибка: путь '{input_path}' не существует", file=sys.stderr)
        sys.exit(1)
        
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        print(f"Анализ файла: {input_path.name}")
        res = process_single_file(input_path, args.window, args.no_plot)
        if res["status"] == "success":
            print(f"✅ ΔR={res['result']['difference']:.4f}, Q={res['result']['Q_percent']:.2f}%")
        else:
            print(f"❌ Ошибка: {res.get('error')}")
            
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            print(f"В '{input_path}' нет CSV файлов", file=sys.stderr)
            sys.exit(1)

        print(f"Найдено {len(csv_files)} файлов. Начинаю обработку...")
        
        results = []
        for i, f in enumerate(sorted(csv_files), 1):
            res = process_single_file(f, args.window, args.no_plot)
            results.append(res)
            
            if res["status"] == "success":
                print(f"[{i}/{len(csv_files)}] ✅ {f.name}: ΔR={res['result']['difference']:.4f}, Q={res['result']['Q_percent']:.2f}%")
            else:
                print(f"[{i}/{len(csv_files)}] ❌ {f.name}: {res.get('error')}")
                
        summary_file = config.OUTPUT_DIR / 'summary.csv'
        save_summary_csv(results, summary_file)
        print("\n" + "="*70)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Сводная таблица: {summary_file}")
        print("="*70)

if __name__ == "__main__":
    main()