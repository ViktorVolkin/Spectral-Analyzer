import json
import pandas as pd
from pathlib import Path

def save_json_report(result_data, output_dir):
    """Сохраняет детальный JSON-отчет для одного файла"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

def save_summary_csv(results, output_path):
    """Создает сводную таблицу (CSV) для всех обработанных файлов"""
    if not results:
        return
    
    csv_rows = []
    for r in results:
        res_val = r.get("result", {})
        params = r.get("smoothing_parameters", {})
        csv_rows.append({
            "Filename": Path(r.get("input_file")).name,
            "Status": r.get("status"),
            "Auto_Window": params.get("window_arg", ""),
            "Window_nm": params.get("window_nm_used", ""),
            "Delta_R": res_val.get("difference", ""),
            "Q_percent": res_val.get("Q_percent", "")
        })
    
    pd.DataFrame(csv_rows).to_csv(output_path, index=False, encoding='utf-8')