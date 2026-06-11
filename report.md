# Part 3 — 模型比較報告

## 模型介紹

### Model 1：SVM（Support Vector Machine）
使用 RBF（Radial Basis Function）kernel，搭配 `StandardScaler` 對特徵進行標準化。
SVM 透過在高維空間中找到最大間距的超平面來分類，適合特徵數量較多、樣本數量中等的情境。

### Model 2：Random Forest
由 200 棵決策樹組成的集成學習模型，每棵樹對隨機子集進行訓練，最終以投票方式決定分類結果。
不需要特徵縮放，並可輸出 feature importance 了解哪些手部關鍵點對辨識最有影響。

---

## 訓練資料

| 類別 | 樣本數 |
|------|--------|
| rock（石頭） | 112 |
| scissors（剪刀） | 130 |
| paper（布） | 142 |
| error（無效手勢） | 180 |
| **總計** | **564** |

- 特徵維度：63（21 個手部關鍵點 × 3 軸 x/y/z，以 wrist 為原點正規化）
- 訓練/測試比例：80% / 20%

---

## 整體指標比較

| 指標 | SVM | Random Forest |
|------|-----|---------------|
| Accuracy | 0.8938 | 0.8938 |
| Precision（macro） | 0.9095 | **0.9167** |
| Recall（macro） | **0.8965** | 0.8931 |
| F1-score（macro） | 0.9007 | **0.9026** |

---

## Per-class 詳細數據

### SVM

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 0.9583 | 0.8846 | 0.9200 | 26 |
| paper | 0.7941 | 0.9310 | 0.8571 | 29 |
| error | 0.8857 | 0.8611 | 0.8732 | 36 |

### Random Forest

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| rock | 1.0000 | 0.9091 | 0.9524 | 22 |
| scissors | 1.0000 | 0.8846 | 0.9388 | 26 |
| paper | 0.8621 | 0.8621 | 0.8621 | 29 |
| error | 0.8049 | 0.9167 | 0.8571 | 36 |

---

## Confusion Matrix

### SVM
（rows = 實際類別，cols = 預測類別，順序：rock / scissors / paper / error）

|  | rock | scissors | paper | error |
|--|------|----------|-------|-------|
| **rock** | 20 | 0 | 1 | 1 |
| **scissors** | 0 | 23 | 2 | 1 |
| **paper** | 0 | 0 | 27 | 2 |
| **error** | 0 | 1 | 4 | 31 |

### Random Forest
（rows = 實際類別，cols = 預測類別，順序：rock / scissors / paper / error）

|  | rock | scissors | paper | error |
|--|------|----------|-------|-------|
| **rock** | 20 | 0 | 0 | 2 |
| **scissors** | 0 | 23 | 1 | 2 |
| **paper** | 0 | 0 | 25 | 4 |
| **error** | 0 | 0 | 3 | 33 |

---

## 模型選擇理由

兩者 Accuracy 相同（0.8938），但從以下幾點判斷 **Random Forest 較佳**：

1. **Precision 更高（0.9167 vs 0.9095）**：RF 預測為某類別時，正確率更高，誤判率更低。
2. **F1-score 更高（0.9026 vs 0.9007）**：整體 Precision 與 Recall 的平衡表現較好。
3. **rock 和 scissors 的 Precision 達到 1.0**：RF 對這兩個手勢完全不會誤判到其他類別。
4. **SVM 的 paper Precision 僅 0.7941**：代表 SVM 較常把其他手勢錯判為 paper。
5. **不需要特徵縮放**：RF 不依賴 StandardScaler，部署更單純。
6. **可解釋性**：RF 可輸出 feature importance，有助於理解模型依賴哪些手部關鍵點。

### 結論
本專案最終採用 **Random Forest** 作為辨識模型（`rf_model.pkl`），
執行指令為：
```bash
python cameraStartRPS.py --model rf
```
