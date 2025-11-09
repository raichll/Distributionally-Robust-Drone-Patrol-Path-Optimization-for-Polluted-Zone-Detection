import pandas as pd

# Parquet 文件路径（可为单个文件，也可为整个目录）
parquet_path = r'E:\dwd_truck\publish_date=2024-04-10\part-00000-d0fc4ecd-5d07-47b1-b172-f46be2c37c18.C000'

# 保存的 CSV 文件路径
csv_path = r'E:\dwd_truck\output4.csv'

# 读取 parquet 文件
df = pd.read_parquet(parquet_path)

# 保存为 CSV 文件（不保存索引）
df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig 适合 Excel 打开中文
