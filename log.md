# RPS 專案開發對話日誌

## 專案資訊
- **GitHub Repo**: https://github.com/wuweian/HW4-Raspberry-Pi
- **開發環境**: Raspberry Pi（直接接螢幕鍵盤，Python 3.11.9）

---

## 2026-06-11　專案環境修復

---

**User：**
> 目前進度（已完成）：已建立完整專案骨架並推送至 GitHub。`index.html` / `camera_view.html` 目前位於 repo 根目錄，建議之後建立 `templates/` 資料夾並把這兩個檔案移過去（app.py 的 Flask 設定預期它們在 `templates/` 下）。

---

**Assistant：**
檢查本地 `RPS_project/` 資料夾，發現是空的。追查後發現：
- 檔案是從 Claude 的暫存 worktree 建立並推送到 GitHub，本地沒有留下
- `C:\Users\weian` 家目錄被設成了 git repo，remote 指向 HW4-Raspberry-Pi（上一個 session 的副作用）
- 本地的 `README.md` 是空檔，會與 GitHub 上的版本衝突

處理步驟：
1. 從 GitHub clone 所有檔案到 `RPS_project/`
2. 建立 `templates/` 資料夾，將 `index.html` 和 `camera_view.html` 移入
3. Commit + push 修正到 GitHub

---

**User：**
> 我的開發環境是 Raspberry Pi 不是在我的電腦

---

**Assistant：**
了解，詢問連線方式與 Python 環境。

---

**User：**
> 直接接螢幕鍵盤。Python 3.11.9。

---

**Assistant：**
提供在 Pi 上的完整執行流程（五個步驟）：
1. Clone repo 到 Pi
2. 安裝套件
3. 收集訓練資料
4. 訓練模型
5. 啟動即時辨識（開兩個終端機）

---

## 2026-06-12　Raspberry Pi 實際執行

---

**User：**
> 第一步出現 REPO not found

---

**Assistant：**
請確認：
1. `ping github.com` 確認網路
2. `curl -I https://github.com/wuweian/HW4-Raspberry-Pi` 確認 repo 是否存在

---

**User：**
> 有 200

---

**Assistant：**
repo 確實存在且公開，請貼出完整指令與錯誤訊息確認是否打錯網址。

---

**User：**
> 第一步完成

---

**Assistant：**
繼續第二步：
```bash
pip install -r requirements.txt
```

---

**User：**
> 好了

---

**Assistant：**
繼續第三步收集訓練資料，先跑 rock：
```bash
python MediapipeDataCollect.py rock
```
對著鏡頭擺石頭手勢，存到約 100 筆再按 `q`。

---

**User：**
> 完成

---

**Assistant：**
提醒繼續收集第四步訓練模型前，但…

---

**User：**
> 我是不是該先收集另外三種的樣本

---

**Assistant：**
對！先把四種手勢都收集完：
```bash
python MediapipeDataCollect.py scissors
python MediapipeDataCollect.py paper
python MediapipeDataCollect.py error
```

---

**User：**
> 做完了，下一步該怎麼做

---

**Assistant：**
現在跑訓練模型：
```bash
python trainSVM.py
python trainRF.py
```
把結果記下來。

---

**User：**
（貼上訓練結果截圖）

---

**Assistant：**
讀取截圖，整理訓練結果如下：

**資料集：** 共 564 筆，63 features
- rock: 112、scissors: 130、paper: 142、error: 180

**SVM 結果：**

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision (macro) | 0.9095 |
| Recall (macro) | 0.8965 |
| F1-score (macro) | 0.9007 |

Per-class：

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| error | 0.8857 | 0.8611 | 0.8732 |
| paper | 0.7941 | 0.9310 | 0.8571 |
| rock | 1.0000 | 0.9091 | 0.9524 |
| scissors | 0.9583 | 0.8846 | 0.9200 |

Confusion Matrix（rock / scissors / paper / error）：
```
[[20  0  1  1]
 [ 0 23  2  1]
 [ 0  0 27  2]
 [ 0  1  4 31]]
```

**Random Forest 結果：**

| 指標 | 數值 |
|------|------|
| Accuracy | 0.8938 |
| Precision (macro) | 0.9167 |
| Recall (macro) | 0.8931 |
| F1-score (macro) | 0.9026 |

Per-class：

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| error | 0.8049 | 0.9167 | 0.8571 |
| paper | 0.8621 | 0.8621 | 0.8621 |
| rock | 1.0000 | 0.9091 | 0.9524 |
| scissors | 1.0000 | 0.8846 | 0.9388 |

Confusion Matrix（rock / scissors / paper / error）：
```
[[20  0  0  2]
 [ 0 23  1  2]
 [ 0  0 25  4]
 [ 0  0  3 33]]
```

兩者 Accuracy 相同，Random Forest 在 Precision 和 F1 略勝，建議使用 RF。

---

**User：**
> 現在比出手勢，有的出現 error，有的出現 ????，取樣的間格時間可能要縮短

---

**Assistant：**
找出兩個問題：

1. **`????`（亂碼）**：OpenCV 的 `cv2.putText()` 不支援中文字元，`石頭`、`剪刀`、`布` 全部顯示為 `????`
2. **`error`**：信心值低於 `CONF_THRESHOLD = 0.6` 門檻，模型不確定時回傳 error

修正內容：
- `cameraStartRPS.py`：`LABEL_DISPLAY` 改為英文（Rock / Scissors / Paper / Error）
- `cameraStartRPS.py`：`CONF_THRESHOLD` 從 `0.6` 降低至 `0.5`
- `MediapipeDataCollect.py`：`CAPTURE_INTERVAL` 從 `1.0` 秒縮短至 `0.5` 秒

Commit message：`Fix Chinese text display, lower confidence threshold to 0.5, halve data capture interval`

Push 完成，告知在 Pi 上執行：
```bash
git pull origin main
```
然後重新收集資料、重新訓練、重啟辨識程式。

---

## 目前專案狀態

### 檔案結構
```
RPS_project/
├── MediapipeDataCollect.py   # 資料收集（CAPTURE_INTERVAL = 0.5s）
├── dataset_utils.py          # 共用：載入 dataSet/*.csv
├── trainSVM.py               # Model 1：SVM
├── trainRF.py                # Model 2：Random Forest
├── cameraStartRPS.py         # 即時辨識（CONF_THRESHOLD = 0.5）
├── app.py                    # Flask 伺服器
├── templates/
│   ├── index.html
│   └── camera_view.html
├── dataSet/                  # 訓練資料 CSV（執行後產生）
├── svm_model.pkl             # 訓練後產生
├── rf_model.pkl              # 訓練後產生
├── requirements.txt
├── report.md                 # 模型比較報告
└── log.md                    # 本對話日誌
```

### Commit 歷程
| Commit | 說明 |
|--------|------|
| `15af045` | Add RPS gesture recognition project（初始推送） |
| `2f2231d` | Move HTML templates into templates/ folder |
| `9a3fdef` | Fix Chinese text display, lower confidence threshold, halve capture interval |
| `f6320cb` | Add development log |
| `1ee83b1` | Add model comparison report |
| `現在` | Update log with full conversation |
