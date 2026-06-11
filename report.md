# RPS 手勢辨識系統 — 完整報告

## 目錄
1. [專案概述](#1-專案概述)
2. [系統架構](#2-系統架構)
3. [資料收集](#3-資料收集)
4. [特徵擷取與正規化](#4-特徵擷取與正規化)
5. [模型一：SVM](#5-模型一svm)
6. [模型二：Random Forest](#6-模型二random-forest)
7. [模型比較與選擇](#7-模型比較與選擇)
8. [即時辨識系統](#8-即時辨識系統)
9. [問題排除紀錄](#9-問題排除紀錄)
10. [結論](#10-結論)

---

## 1. 專案概述

本專案實作一套部署於 **Raspberry Pi** 上的即時剪刀石頭布（Rock-Paper-Scissors）手勢辨識系統，包含：

- **Part 1**：使用 MediaPipe Hands 從攝影機影像中擷取手部關鍵點特徵
- **Part 2**：將辨識結果透過 Flask SSE 串流傳送至瀏覽器即時顯示
- **Part 3**：訓練並比較 SVM 與 Random Forest 兩種分類模型

**辨識類別：**

| 類別 | 說明 |
|------|------|
| Rock | 石頭（握拳） |
| Scissors | 剪刀（食中指伸出） |
| Paper | 布（五指展開） |
| Error | 無效手勢（非上述三種） |

**開發環境：**
- 硬體：Raspberry Pi
- OS：Raspberry Pi OS
- Python：3.11.9
- 主要套件：MediaPipe、OpenCV、scikit-learn、Flask

---

## 2. 系統架構

```
攝影機（Pi Camera）
       │
       ▼
MediapipeDataCollect.py / cameraStartRPS.py
  └─ MediaPipe Hands → 擷取 21 個手部關鍵點
  └─ 正規化 → 63 維特徵向量
       │
       ▼
分類模型（svm_model.pkl / rf_model.pkl）
  └─ 預測類別 + 信心值
       │
       ▼
Flask app.py（POST /post_camera_frame）
  └─ SSE /get_camera_stream
       │
       ▼
瀏覽器（camera_view.html）即時顯示影像與辨識結果
```

**檔案結構：**
```
RPS_project/
├── MediapipeDataCollect.py   # 資料收集
├── dataset_utils.py          # 載入 CSV 資料
├── trainSVM.py               # SVM 訓練
├── trainRF.py                # Random Forest 訓練
├── cameraStartRPS.py         # 即時辨識主程式
├── app.py                    # Flask 伺服器
├── templates/
│   ├── index.html
│   └── camera_view.html
├── dataSet/                  # 收集的訓練資料
│   ├── rock.csv
│   ├── scissors.csv
│   ├── paper.csv
│   └── error.csv
├── svm_model.pkl
├── rf_model.pkl
└── requirements.txt
```

---

## 3. 資料收集

### 收集方式
使用 `MediapipeDataCollect.py`，以攝影機即時拍攝手部，每 **0.5 秒**自動擷取一筆特徵資料並寫入 CSV。

```bash
python MediapipeDataCollect.py rock
python MediapipeDataCollect.py scissors
python MediapipeDataCollect.py paper
python MediapipeDataCollect.py error
```

### 資料集統計

| 類別 | 樣本數 |
|------|--------|
| rock（石頭） | 112 |
| scissors（剪刀） | 130 |
| paper（布） | 142 |
| error（無效手勢） | 180 |
| **總計** | **564** |

> `error` 類別收集最多樣本（180 筆），包含比讚、OK、手指彎曲等各種非合法手勢，使模型能更穩健地判斷「這不是有效手勢」。

### 訓練 / 測試分割
- 訓練集：80%（約 451 筆）
- 測試集：20%（113 筆）

---

## 4. 特徵擷取與正規化

### MediaPipe Hands
MediaPipe Hands 可從影像中偵測手部並輸出 **21 個關鍵點**，每個關鍵點包含 x、y、z 三個座標，共 **63 個數值**。

21 個關鍵點對應位置：

```
                8   12  16  20
                |   |   |   |
                7   11  15  19
            4   |   |   |   |
            |   6   10  14  18
            3   |   |   |   |
            |   5---9--13--17
            2  /
             1/
             0  ← wrist（手腕，landmark 0）
```

### 正規化方法
原始座標會因手部位置與距離鏡頭遠近而不同，直接使用會導致模型不穩定。因此進行兩步正規化：

1. **平移**：以 wrist（landmark 0）為原點，所有關鍵點座標減去 wrist 座標
2. **縮放**：以 wrist → 中指 MCP（landmark 9）的距離為 scale，所有座標除以此距離

```python
pts -= pts[0]              # 平移，以 wrist 為原點
scale = np.linalg.norm(pts[9])
if scale > 1e-6:
    pts /= scale           # 縮放，消除手部大小差異
```

正規化後的特徵向量對手部位置與距離具有不變性（translation & scale invariant），大幅提升模型泛化能力。

---

## 5. 模型一：SVM

### 原理
Support Vector Machine（SVM）透過在高維特徵空間中尋找最大間距超平面（maximum-margin hyperplane）進行分類。使用 **RBF（Radial Basis Function）kernel** 處理非線性可分的資料。

### 訓練設定
- Kernel：RBF
- 前處理：`StandardScaler`（將各特徵縮放至均值 0、標準差 1）
- 機率輸出：啟用 `probability=True`，可取得各類別的信心值

### 訓練結果

**整體指標：**

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision（macro） | 0.9095 |
| Recall（macro） | 0.8965 |
| F1-score（macro） | 0.9007 |

**Per-class 詳細數據：**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 0.9583 | 0.8846 | 0.9200 | 26 |
| paper | 0.7941 | 0.9310 | 0.8571 | 29 |
| error | 0.8857 | 0.8611 | 0.8732 | 36 |

**Confusion Matrix（rows = 實際，cols = 預測，順序：rock / scissors / paper / error）：**

|  | rock | scissors | paper | error |
|--|:----:|:--------:|:-----:|:-----:|
| **rock** | **20** | 0 | 1 | 1 |
| **scissors** | 0 | **23** | 2 | 1 |
| **paper** | 0 | 0 | **27** | 2 |
| **error** | 0 | 1 | 4 | **31** |

主要誤判：paper 被預測為 error（4 次）、scissors 被預測為 paper（2 次）。

---

## 6. 模型二：Random Forest

### 原理
Random Forest 是集成學習（Ensemble Learning）方法，由多棵決策樹組成。每棵樹對隨機取樣的資料子集與特徵子集進行訓練，最終以投票方式決定分類結果。

優點：
- 不需要特徵縮放
- 對雜訊有較強的抵抗力
- 可輸出 **feature importance**，解釋模型依賴哪些特徵

### 訓練設定
- 決策樹數量：200 棵
- 機率輸出：使用 `predict_proba()` 取得信心值

### 訓練結果

**整體指標：**

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision（macro） | 0.9167 |
| Recall（macro） | 0.8931 |
| F1-score（macro） | 0.9026 |

**Per-class 詳細數據：**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 1.0000 | 0.8846 | 0.9388 | 26 |
| paper | 0.8621 | 0.8621 | 0.8621 | 29 |
| error | 0.8049 | 0.9167 | 0.8571 | 36 |

**Confusion Matrix（rows = 實際，cols = 預測，順序：rock / scissors / paper / error）：**

|  | rock | scissors | paper | error |
|--|:----:|:--------:|:-----:|:-----:|
| **rock** | **20** | 0 | 0 | 2 |
| **scissors** | 0 | **23** | 1 | 2 |
| **paper** | 0 | 0 | **25** | 4 |
| **error** | 0 | 0 | 3 | **33** |

主要誤判：paper 被預測為 error（4 次）、rock 與 scissors 各有 2 次被誤判為 error。

**Top 10 重要特徵：**

| 排名 | 特徵 | Importance |
|------|------|-----------|
| 1 | landmark 20 y（小指指尖 y） | 0.0606 |
| 2 | landmark 16 y（無名指指尖 y） | 0.0595 |
| 3 | landmark 12 y（中指指尖 y） | 0.0491 |
| 4 | landmark 15 y（無名指第一節 y） | 0.0443 |
| 5 | landmark 9 x（中指 MCP x） | 0.0354 |
| 6 | landmark 4 y（拇指指尖 y） | 0.0335 |
| 7 | landmark 5 x（食指 MCP x） | 0.0329 |
| 8 | landmark 19 y（小指第一節 y） | 0.0321 |
| 9 | landmark 11 y（中指第二節 y） | 0.0295 |
| 10 | landmark 7 y（食指第二節 y） | 0.0274 |

> 重要特徵多為**指尖與關節的 y 軸座標**，符合直覺：石頭、剪刀、布最大的差異在於各手指是否彎曲（y 軸高低）。

---

## 7. 模型比較與選擇

### 整體指標比較

| 指標 | SVM | Random Forest | 勝出 |
|------|:---:|:-------------:|:----:|
| Accuracy | 0.8938 | 0.8938 | 平手 |
| Precision（macro） | 0.9095 | **0.9167** | RF ✓ |
| Recall（macro） | **0.8965** | 0.8931 | SVM ✓ |
| F1-score（macro） | 0.9007 | **0.9026** | RF ✓ |

### Per-class 比較

| Class | SVM F1 | RF F1 | 勝出 |
|-------|:------:|:-----:|:----:|
| rock | 0.9524 | 0.9524 | 平手 |
| scissors | 0.9200 | **0.9388** | RF ✓ |
| paper | 0.8571 | 0.8621 | RF ✓ |
| error | 0.8732 | 0.8571 | SVM ✓ |

### 選擇 Random Forest 的理由

1. **Precision 更高（0.9167 vs 0.9095）**：RF 預測為某類別時，正確率更高，誤判率更低。

2. **F1-score 更高（0.9026 vs 0.9007）**：整體 Precision 與 Recall 的平衡表現較好。

3. **scissors 的 Precision 達到 1.0**：RF 對剪刀手勢完全不會誤判為其他類別；SVM 則有 4.17% 的機率誤判。

4. **SVM 的 paper Precision 僅 0.7941**：代表 SVM 較常把其他手勢錯判為 paper（約 20% 的預測為 paper 是錯的）。

5. **不需要特徵縮放**：RF 不依賴 StandardScaler，推論時步驟更簡單。

6. **可解釋性**：RF 提供 feature importance，可從數據角度驗證「哪些關節最能區分手勢」。

### 結論
**本專案最終採用 Random Forest 作為辨識模型。**

---

## 8. 即時辨識系統

### 執行方式（在 Raspberry Pi 上）

**終端機 1 — 啟動 Flask 伺服器：**
```bash
cd ~/RPS_project
python app.py
```

**終端機 2 — 啟動攝影機辨識：**
```bash
cd ~/RPS_project
python cameraStartRPS.py --model rf
```

**瀏覽器開啟：**
```
http://127.0.0.1:5000/visualize-camera-view
```

### 辨識流程
1. `cameraStartRPS.py` 讀取攝影機影像
2. MediaPipe Hands 偵測手部，擷取 21 個關鍵點
3. 正規化成 63 維特徵向量
4. `rf_model.pkl` 預測類別與信心值
5. 若信心值 ≥ 0.5 → 顯示預測類別；若 < 0.5 → 顯示 "Error"
6. 將影像（base64）與結果 POST 到 Flask `/post_camera_frame`
7. Flask 透過 SSE 串流至瀏覽器即時更新

---

## 9. 問題排除紀錄

### 問題 1：辨識結果顯示 `????`
- **原因**：OpenCV 的 `cv2.putText()` 不支援中文字元，`石頭`、`剪刀`、`布` 無法正確渲染
- **修正**：將 `LABEL_DISPLAY` 改為英文標籤（Rock / Scissors / Paper / Error）

### 問題 2：手勢常被判為 `error`
- **原因**：信心值門檻 `CONF_THRESHOLD = 0.6` 過高，模型略不確定時即回傳 error
- **修正**：降低為 `CONF_THRESHOLD = 0.5`

### 問題 3：取樣間隔過長
- **原因**：`CAPTURE_INTERVAL = 1.0` 秒，每次收集資料需要較長時間，且相鄰樣本差異小
- **修正**：縮短為 `CAPTURE_INTERVAL = 0.5` 秒，讓同樣時間內收集更多樣化的樣本

### 問題 4：GitHub Clone 失敗（Repo not found）
- **原因**：網址輸入錯誤
- **排查方式**：`curl -I https://github.com/wuweian/HW4-Raspberry-Pi` 確認回傳 HTTP 200
- **修正**：重新輸入正確網址

---

## 10. 結論

本專案成功實作了一套從資料收集、模型訓練到即時辨識的完整 RPS 手勢辨識系統。

| 項目 | 結果 |
|------|------|
| 訓練資料 | 564 筆 / 4 類別 / 63 特徵 |
| 最佳模型 | Random Forest |
| 最終 Accuracy | 89.38% |
| 最終 F1-score（macro） | 0.9026 |
| 部署環境 | Raspberry Pi + Flask |
| 即時顯示 | 瀏覽器 SSE 串流 |

Random Forest 在多數指標上優於 SVM，且不需要特徵縮放，適合部署在 Raspberry Pi 這類資源有限的環境。未來若要進一步提升準確率，可從以下方向改進：
- 增加訓練樣本數（目前每類僅 112~180 筆）
- 針對 paper 類別加強資料收集（目前 F1 最低）
- 嘗試更深的模型（如 MLP 神經網路）
