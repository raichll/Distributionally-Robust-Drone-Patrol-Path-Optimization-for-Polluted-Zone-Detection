import os
import pandas as pd

# 设置目录路径
folder_path = r'E:\dwd_truck\publish_date=2024-04-10'

# 获取所有 .C000 文件（不区分大小写）
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.c000')]

# 读取并合并数据
df_list = []
for file in file_list:
    try:
        df = pd.read_parquet(file)
        df_list.append(df)
    except Exception as e:
        print(f"读取失败：{file}，错误信息：{e}")

if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"成功读取并合并 {len(df_list)} 个文件，共 {len(combined_df)} 行。")

    # 随机保留三分之一（fraction=1/4）
    sampled_df = combined_df.sample(frac=1/4, random_state=42)  # 设置 random_state 保证可复现

    # 可选：保存为 CSV
    output_csv = r'E:\dwd_truck\output_sampled.csv'
    sampled_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"已保存四分之一数据为 CSV 文件：{output_csv}")
else:
    print("没有成功读取任何 .C000 文件。")
