# 分析ツールドキュメント
分析ツールのドキュメントです。

## ヘッダーファイル

独自のコードを作成するときに便利な関数がまとめられています。

### lib_analyze_base.hpp
webuiから出力されたcsvファイルを高速に読み込むためのツールが含まれています。

- `parse_csv_mmap`: ファイルからデータをメモリに展開します。
- `convert_to_horus_format`: 展開されたデータをformatします。この関数の出力を使用します。

### lib_horus_type.hpp
このプロジェクトで独自に定義された型を定義しています。それに合わせてoperatorのオーバーロードも定義しています。

## 胎動計測

ファイル：`sample_base_analyze.cpp`

```bash
./sample_base_analyze.out CSV_FILE_PATH OUTPUT_DIR
```

- 第一引数：分析結果のCSVファイル
- 第二引数：分析結果を出力するディレクトリ

## 心拍分析

ファイル：`sample_heart_analyze.cpp`

```bash
./sample_hearts_analyze.out result.csv all_video_merge.webm 0 ./out.csv
```

- 第一引数：分析結果のCSVファイル
- 第二引数：分析対象の動画(プロジェクトのフォルダに`all_video_merge.webm`があるのでそれを指定してください。)
- 第三引数：分析index
- 第四引数：出力するcsvファイル
