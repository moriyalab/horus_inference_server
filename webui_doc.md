# WebUI Document

## Upload Video To Database

このタブはデータベースの作成とデータのアップロードのタブです。

1. PCを直接操作して`~/horus_inference_server/INPUT_VIDEOS_COPY_TO_HERE`ディレクトリに分析したい動画を全て入れます。
2. Reload Video Listボタンを押して動画のリストが表示されることを確認します。
3. Start uploadボタンを押してアップロードを行います。内部で動画を処理するのでアップロードには少し時間がかかります。

## Edit Database

プロジェクトの確認と削除を行えます。

## Create Base Dataset

認識したいオブジェクトを決定します。認識したいオブジェクトごとに下記の手順を踏む必要があります。

1. Reload Project Listを押してプロジェクトを選択します。
2. `result image`にサムネイルが表示されることを確認して下のスライドバーを操作して認識したいオブジェクトを囲みます。
(このときオブジェクトはギリギリを選択すると良いです(これには根拠はなく作者の肌感です))
3. Start Inferenceボタンを押して完了です。処理に時間がかかるので待ちます。

## Build Base ML Model

`Create Base Dataset`で全てのオブジェクトを登録したら大規模分析用のモデルを作成します。

1. プロジェクトを選択してbuild base ML Modelボタンを押す。処理に時間がかかるので待ちます。

## Analyze Video

`Build Base ML Model`が完了したらAnaluzeVideoで動画を分析します。

分析が完了するとprojectフォルダの中に`analyze`フォルダが生成されて分析結果がまとめられます。
