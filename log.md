# RPS 專案開發日誌

## 專案資訊
- **GitHub Repo**: https://github.com/wuweian/HW4-Raspberry-Pi
- **開發環境**: Raspberry Pi（直接接螢幕鍵盤）
- **Python 版本**: 3.11.9

---

## 階段一：專案建立（前一個 Session）

### 完成內容
- 建立完整專案骨架，推送至 GitHub
- **Part 1 資料收集**：`MediapipeDataCollect.py`
  - 使用 MediaPipe Hands 擷取 21 個手部關鍵點
  - 正規化（以 wrist 為原點、用 wrist→中指 MCP 距離縮放）成 63 維特徵向量
  - CLI: `python MediapipeDataCollect.py <rock|scissors|paper|error>`
- **Part 3 模型訓練**：
  - `trainSVM.py`（SVM, RBF kernel + StandardScaler）
  - `trainRF.py`（Random Forest, 200 棵樹 + feature importance）
- **Part 1+2 核心**：
  - `cameraStartRPS.py`：攝影機辨識，POST 結果到 Flask
  - `app.py` + `templates/`：Flask + SSE 即時顯示

---

## 階段二：環境修復與部署（2026-06-11）

### 問題診斷
- 本地 `RPS_project/` 資料夾為空（前一個 session 的檔案建立在暫存 worktree）
- 發現 `C:\Users\weian` 家目錄被設為 git repo，remote 指向 HW4-Raspberry-Pi（異常副作用）
- `index.html` / `camera_view.html` 位於 repo 根目錄，但 `app.py` 預期在 `templates/` 下

### 處理步驟
1. 將 GitHub repo clone 到 `RPS_project/`
2. 建立 `templates/` 資料夾，將 `index.html` 和 `camera_view.html` 移入
3. Commit + push 修正到 GitHub

---

## 階段三：Raspberry Pi 執行（2026-06-12）

### 執行流程

#### 1. Clone 專案到 Pi
```bash
cd ~
git clone https://github.com/wuweian/HW4-Raspberry-Pi.git RPS_project
cd RPS_project
```

#### 2. 安裝套件
```bash
pip install -r requirements.txt
```

#### 3. 收集訓練資料（每類各約 100~180 筆）
```bash
python MediapipeDataCollect.py rock
python MediapipeDataCollect.py scissors
python MediapipeDataCollect.py paper
python MediapipeDataCollect.py error
```

資料筆數：
- rock: 112 samples
- scissors: 130 samples
- paper: 142 samples
- error: 180 samples
- **總計: 564 samples，63 features**

#### 4. 訓練模型
```bash
python trainSVM.py
python trainRF.py
```

---

## 模型訓練結果

### SVM（RBF kernel + StandardScaler）

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision (macro) | 0.9095 |
| Recall (macro) | 0.8965 |
| F1-score (macro) | 0.9007 |

**Per-class report：**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| error | 0.8857 | 0.8611 | 0.8732 | 36 |
| paper | 0.7941 | 0.9310 | 0.8571 | 29 |
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 0.9583 | 0.8846 | 0.9200 | 26 |

**Confusion Matrix（rows=true, cols=pred, order: rock/scissors/paper/error）：**
```
[[20  0  1  1]
 [ 0 23  2  1]
 [ 0  0 27  2]
 [ 0  1  4 31]]
```

---

### Random Forest（200 棵樹）

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision (macro) | 0.9167 |
| Recall (macro) | 0.8931 |
| F1-score (macro) | 0.9026 |

**Per-class report：**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| error | 0.8049 | 0.9167 | 0.8571 | 36 |
| paper | 0.8621 | 0.8621 | 0.8621 | 29 |
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 1.0000 | 0.8846 | 0.9388 | 26 |

**Confusion Matrix（rows=true, cols=pred, order: rock/scissors/paper/error）：**
```
[[20  0  0  2]
 [ 0 23  1  2]
 [ 0  0 25  4]
 [ 0  0  3 33]]
```

**Top 10 重要特徵（landmark_idx*3 + axis）：**
| Feature | Importance |
|---------|-----------|
| landmark 20 y | 0.0606 |
| landmark 16 y | 0.0595 |
| landmark 12 y | 0.0491 |
| landmark 15 y | 0.0443 |
| landmark 9 x | 0.0354 |
| landmark 4 y | 0.0335 |
| landmark 5 x | 0.0329 |
| landmark 19 y | 0.0321 |
| landmark 11 y | 0.0295 |
| landmark 7 y | 0.0274 |

---

### 模型比較結論

兩者 Accuracy 相同（0.8938），Random Forest 在 Precision 和 F1 略勝：
- **建議使用 Random Forest**（`--model rf`）

---

## 問題排除記錄

### 問題 1：`????` 顯示亂碼
- **原因**：OpenCV 的 `cv2.putText()` 不支援中文字元
- **修正**：將 `LABEL_DISPLAY` 改為英文（Rock / Scissors / Paper / Error）

### 問題 2：手勢常被判為 `error`
- **原因**：信心值門檻 `CONF_THRESHOLD = 0.6` 過高
- **修正**：降低為 `0.5`

### 問題 3：取樣間隔太長
- **修正**：`CAPTURE_INTERVAL` 從 `1.0` 秒縮短為 `0.5` 秒，收集更多樣化樣本後重新訓練

---

## 啟動即時辨識

```bash
# 終端機 1
python app.py

# 終端機 2
python cameraStartRPS.py --model rf
```

瀏覽器開啟：`http://127.0.0.1:5000/visualize-camera-view`
