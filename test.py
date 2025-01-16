import os
import pandas as pd
import matplotlib.pyplot as plt

# ディレクトリ内の全てのCSVファイルを処理
def plot_csv_files_in_directory(directory_path):
    # ディレクトリ内のファイル一覧を取得
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        print("CSVファイルが見つかりません。")
        return

    # 各CSVファイルを読み込んでグラフを作成
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)

        try:
            # CSVをデータフレームとして読み込み
            df = pd.read_csv(file_path)

            # 必要なカラムが存在するか確認
            if 'time' not in df.columns or 'norm_bottom_right' not in df.columns:
                print(f"{csv_file} に必要な列がありません。スキップします。")
                continue

            # グラフ作成
            plt.figure()
            plt.plot(df['time'], df['norm_bottom_right'], label=csv_file)
            plt.title(f"Graph of {csv_file}")
            plt.xlabel('Time')
            plt.ylabel('Norm Bottom Right')
            plt.legend()
            plt.grid(True)

            # グラフを表示または保存
            # plt.show()
            # 保存したい場合は以下を使用
            plt.savefig(f"results/{csv_file}_graph.png")

        except Exception as e:
            print(f"{csv_file} の処理中にエラーが発生しました: {e}")

# 実行するディレクトリのパスを指定
directory_path = "./results"  # 例: 現在のディレクトリ内の 'csv_files' フォルダ
plot_csv_files_in_directory(directory_path)
