# RPS_demo (剪刀石頭布辨識)

Adapted from the AIoT Lab7 (`OldManFalls`跌倒偵測) template, using MediaPipe Hands
to classify Rock(石頭) / Scissors(剪刀) / Paper(布) / Error(其他錯誤手勢).

## 環境安裝

```
pip install -r requirements.txt
```

## 1. 收集訓練資料 (在電腦端執行)

對每個手勢類別各執行一次 (按 q 結束)：

```
python MediapipeDataCollect.py rock
python MediapipeDataCollect.py scissors
python MediapipeDataCollect.py paper
python MediapipeDataCollect.py error
```

每秒擷取一筆手部 21 個關鍵點 (x,y,z) 正規化後的特徵，存到
`./dataSet/<label>.csv`。`error` 類別請做各種「不是石頭/剪刀/布」的手勢
(比讚、OK、張開但角度怪異等)，讓模型學會辨識「這不是合法手勢」。

建議每個類別至少 100~150 筆樣本。

## 2. 訓練兩個模型 (part 3 報告用)

```
python trainSVM.py     # Model 1: SVM (RBF kernel)
python trainRF.py      # Model 2: Random Forest
```

兩支程式都會輸出：
- Accuracy
- Precision / Recall / F1-score (per-class + macro 平均)
- Confusion matrix

把這些數字填進報告 part 3 的表格中即可。`trainSVM.py` 產生 `svm_model.pkl`，
`trainRF.py` 產生 `rf_model.pkl`。

## 3. 在 Raspberry Pi 上執行辨識 (part 1 + part 2)

啟動 Flask 伺服器：

```
python app.py
```

另開一個終端機，執行辨識程式 (選擇分數較高的模型)：

```
python cameraStartRPS.py --model svm
# 或
python cameraStartRPS.py --model rf
```

記得依照 Lab7 的步驟，把 `cameraStartRPS.py` 開頭的 `POST_URL` 改成你的對外 IP：
```
http://<你的對外IP>:5000/post_camera_frame
```

在另一台連到同網路的電腦瀏覽器輸入 `http://<對外IP>:5000/visualize-camera-view`
即可看到即時影像與辨識結果。

## 附錄：用影片作為輸入

```
python cameraStartRPS.py --model svm --video your_demo_video.mp4
```

## 檔案結構

```
RPS_project/
├── MediapipeDataCollect.py   # part1 資料收集 (MediaPipe Hands)
├── dataset_utils.py          # 共用：載入 dataSet/*.csv
├── trainSVM.py                # part3 model 1
├── trainRF.py                 # part3 model 2
├── cameraStartRPS.py          # part1+2 核心：camera -> model -> 分類
├── app.py                     # Flask 伺服器
├── templates/
│   ├── index.html
│   └── camera_view.html
├── dataSet/                    # 收集的 CSV 訓練資料 (執行後產生)
├── svm_model.pkl                # 訓練後產生
├── rf_model.pkl                  # 訓練後產生
└── requirements.txt
```
