import pandas as pd

input_file = './data_newest.xlsx'
output_file = './data_newest_result.csv'

df = pd.read_excel(input_file)

df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=',')

print(f"✅ Файл успешно сохранен как {output_file}")