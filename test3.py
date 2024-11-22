import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# CSVファイルのパスを取得してソート
csv_file_paths = glob.glob("./results/*_0.csv")
csv_file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("_0", '')))

# データフレームを部分的に読み込む関数
def load_and_sample_csv(path):
    df = pd.read_csv(path, names=['x_center', 'y_center', 'width', 'height', 'confidence'])
    return df.iloc[::2]  # 100行ごとにサンプリング

# 並列処理でCSVファイルを読み込む
with ProcessPoolExecutor() as executor:
    data_frames = list(executor.map(load_and_sample_csv, csv_file_paths))

# 全データを連結
combined_df = pd.concat(data_frames, ignore_index=True)

# 連結結果をCSVに書き出し
combined_df.to_csv("combined_results.csv", index=False)

window_size = 5  # 移動平均のウィンドウサイズ

# x_center 列と y_center 列を数値型に変換し、変換できない値は NaN に置き換える
combined_df['x_center'] = pd.to_numeric(combined_df['x_center'], errors='coerce')
combined_df['y_center'] = pd.to_numeric(combined_df['y_center'], errors='coerce')

# 移動平均を各列に適用
combined_df['x_center_ma'] = combined_df['x_center'].rolling(window=window_size).mean()
combined_df['y_center_ma'] = combined_df['y_center'].rolling(window=window_size).mean()

# 前フレームとの差分を計算
combined_df['x_diff'] = combined_df['x_center'].diff()
combined_df['y_diff'] = combined_df['y_center'].diff()

# 移動量のノルム（移動量の大きさ）を計算
combined_df['movement_norm'] = (combined_df['x_diff']**2 + combined_df['y_diff']**2)**0.5
combined_df['movement_norm'] = combined_df['movement_norm'].rolling(window=window_size).mean()



# 移動平均を含めたデータと移動量ノルムを新しいCSVに書き出し
combined_df.to_csv("combined_results_with_moving_average_and_norm.csv", index=False)
