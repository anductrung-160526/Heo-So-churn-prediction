# 🐷 Dự báo Churn Dịch vụ "Heo Số" — MB Bank

> **Phân tích chân dung khách hàng và xây dựng mô hình dự báo rời bỏ dịch vụ tiết kiệm số trên ứng dụng MBBank**
>
> Dự án thuộc khuôn khổ cuộc thi **FDC 105 – Challenge 3** | Nhóm 5 | 2025

---

## 📌 Tóm tắt

Dự án xây dựng pipeline phân tích & dự báo cho **dịch vụ tiết kiệm số "Heo Số"** của MB Bank, gồm:

1. **Định nghĩa churn từ đầu** trong điều kiện dữ liệu **không có nhãn churn sẵn** — tự thiết kế tiêu chí dựa trên hành vi đăng nhập.
2. **Phân tích chân dung khách hàng** sử dụng và rời bỏ dịch vụ theo 4 chiều: nhân khẩu học, mức độ tương tác, khả năng tài chính và hành vi tiết kiệm.
3. **Xây dựng và so sánh 5 mô hình ML** (Logistic Regression, Decision Tree, Random Forest, LightGBM, XGBoost) — mô hình tốt nhất đạt **ROC-AUC 0.9422**.
4. **Đề xuất 3 giải pháp kinh doanh** cá nhân hóa, cảnh báo sớm, và tích hợp hệ sinh thái — bám sát feature importance của mô hình.

---

## 🎯 Bối cảnh & Vì sao bài toán này quan trọng

**Heo Số** là tính năng tiết kiệm mục tiêu trên app MBBank: gửi tiền linh hoạt, ngắn hạn, không kỳ hạn cố định, vận hành 100% online. Đây là sản phẩm thuộc nhóm **tiền gửi chi phí thấp (CASA)** — loại vốn quý giá nhất với ngân hàng.

Theo Tạp chí Ngân hàng Nhà nước (2024):
- **70–80%** tổng nguồn vốn của NHTM Việt Nam là tiền gửi
- **Trên 25%** tổng huy động đến từ tiền gửi chi phí thấp
- Quy mô và tính ổn định của nguồn vốn này tác động trực tiếp tới khả năng sinh lời

Khách hàng rời bỏ Heo Số đồng nghĩa với việc ngân hàng **mất nguồn vốn giá rẻ** — chi phí cao hơn nhiều so với việc giữ chân. Do đó, **phát hiện sớm khách hàng có nguy cơ churn** có giá trị kinh doanh rõ ràng và đo lường được.

---

## 🧩 Thách thức của bài toán

| Thách thức | Cách xử lý |
|---|---|
| **Không có nhãn churn sẵn** | Tự định nghĩa churn dựa trên hành vi đăng nhập T3 vs T6 (xem bên dưới) |
| **Dữ liệu đã được scale** mà không rõ phương pháp | Phân tích phân phối để giả định Decimal Scaling, giữ nguyên tương quan |
| **Tỷ lệ NULL cao ở nhiều cột** | Phân biệt 2 loại null (không đăng nhập vs đăng nhập nhưng không giao dịch), xử lý theo ngữ cảnh |
| **Mất cân bằng lớp** (71.3% vs 28.7%) | Áp dụng SMOTE trong pipeline train, không leak sang validation/test |
| **Nguy cơ data leakage** từ chính biến đăng nhập dùng để định nghĩa churn | Loại bỏ biến số lần đăng nhập khỏi feature set |

---

## 🏷️ Định nghĩa Churn

Vì không có cột nhãn sẵn, nhóm tự xây tiêu chí từ dữ liệu hành vi đăng nhập:

```
Số lần đăng nhập T6 = Số lần đăng nhập T3   →  Churn = 1
                                              (hệ thống không ghi nhận tương tác mới)

Số lần đăng nhập T6 > Số lần đăng nhập T3   →  Churn = 0
                                              (khách tiếp tục sử dụng app)
```

**Churn ở đây = ngừng/giảm tương tác với tính năng Heo Số**, *không* đồng nghĩa với đóng tài khoản hay rời ngân hàng. Đây là định nghĩa phù hợp với bài toán kinh doanh: ngân hàng muốn giữ khách dùng Heo Số chứ không chỉ giữ khách trong hệ thống.

**Kết quả phân loại:** 71.3% không churn, 28.7% churn — chênh ~2.5 lần.

---

## 🔧 Pipeline xử lý dữ liệu

```
RAW DATA (50 cột, nhiều NULL, mất cân bằng)
    │
    ├─ B1. Chuẩn hóa cấu trúc        → đổi tên cột, chuẩn hóa định tính
    ├─ B2. Kiểm tra giá trị bất thường → phát hiện tuổi đến 800 (lỗi nhập liệu)
    ├─ B3. Xử lý NULL theo 4 chiến lược:
    │      • Fill theo context nhân khẩu học (mode theo thế hệ)
    │      • Kiểm tra logic tích lũy theo thời gian (T3 ≤ T6)
    │      • Điền 0 cho cột giao dịch định lượng
    │      • Bỏ cột ID, phân cấp hành chính, biến không phục vụ modeling
    │
    ▼
DATASET SẠCH (42 cột, 0 NULL)
```

Một điểm đáng lưu ý: dataset có hai loại "không có dữ liệu" mang ý nghĩa khác nhau — `0` là khách *có đăng nhập nhưng không giao dịch*, `NULL` là khách *không đăng nhập*. Việc phân biệt được hai trường hợp này là chìa khóa để xử lý NULL đúng cách (không thay bừa bằng 0 hay mean).

---

## 📊 Phát hiện chính từ EDA

### Chân dung khách hàng tổng quan

| Yếu tố | Đặc điểm chiếm ưu thế |
|---|---|
| Thế hệ | **Gen Y (77.14%)**, Gen Z (21.90%) |
| Giới tính | Nam (55.24%) |
| Tình trạng hôn nhân | **Độc thân** (~64%) |
| Khu vực | **Hà Nội (34.29%)**, TP.HCM, Nam Định |

### 4 insight then chốt phân biệt churn vs non-churn

**1. Tần suất đăng nhập là chỉ báo mạnh nhất.** Nhóm trung thành đăng nhập **gấp ~3.4 lần** nhóm churn (25.7 vs 15.5 lần/tháng). Khi khách hàng đạt mức "mở app gần như mỗi ngày", họ coi Heo Số là nơi lưu trữ tài sản chính.

**2. Hành vi dòng tiền đảo ngược rõ rệt.** Nhóm không churn **gửi thêm** đều đặn các khoản nhỏ ("tích tiểu thành đại"); nhóm churn **rút ròng** ở tất cả phân khúc số dư, nghiêm trọng nhất ở phân khúc số dư rất cao (-308,362 VND so với +17,915 của nhóm trung thành).

**3. Số đối tác giao dịch tương quan thuận với độ trung thành.** Khách giao dịch với nhiều đối tác thường dùng Heo Số như công cụ quản lý tài chính, tận dụng tiền lẻ sau giao dịch để tích lũy → vòng tròn hành vi *"Đối tác nhiều → Nhiều giao dịch → Tiền gửi nhiều"*.

**4. Hình thức thanh toán và loại dịch vụ KHÔNG phân biệt được churn.** Hai nhóm có phân phối tương đồng ở các biến này — đây là phát hiện ngược trực giác nhưng quan trọng để **loại biến nhiễu khỏi mô hình**.

### Chân dung khách hàng churn điển hình

> **Nữ giới, Gen Y, độc thân**, sống ở thành phố lớn (Hà Nội/TP.HCM), đăng nhập app thưa, rút tiền ròng khỏi Heo Số, ít tương tác tổng thể (ít đối tác và khối lượng giao dịch thấp).

---

## 🤖 Xây dựng mô hình

### Feature Engineering

Từ 50 biến gốc → **21 biến** đưa vào mô hình, sau khi loại:

| Biến loại | Lý do |
|---|---|
| Số lần đăng nhập T3, T6 | **Tránh data leakage** (chính 2 biến này định nghĩa nhãn) |
| Lãi suất danh nghĩa & thực tế | Lệch khỏi trọng tâm phân tích đặc điểm khách hàng |
| `loại_kỳ_hạn_vay` | NULL > 50% |
| `số_lượng_loại_giao_dịch_phổ_biến` | Phụ thuộc vào dịch vụ khác, không liên quan Heo Số |
| Cột phân cấp hành chính chi tiết, cột ID | Không phục vụ modeling |

### Quy trình huấn luyện

```
Train / Validation / Test  =  70 / 15 / 15  (stratified)
                    │
                    ▼
    Pipeline: SMOTE → XGBoost (chỉ áp dụng SMOTE trên TRAIN)
                    │
                    ▼
    RandomizedSearchCV  (40 iters, 5-fold Stratified, scoring='roc_auc')
                    │
                    ▼
    Đánh giá trên Validation → Test (không tinh chỉnh thêm)
```

**Lưu ý kỹ thuật:** SMOTE được đặt **trong** `imblearn.Pipeline` để chỉ áp dụng trên train fold của CV, không leak sang validation — đây là lỗi rất phổ biến mà nhiều demo SMOTE mắc phải.

### Kết quả so sánh 5 mô hình

| Metric | Logistic Regression | Decision Tree | Random Forest | LightGBM | **XGBoost** |
|---|---|---|---|---|---|
| **ROC-AUC** | 0.7443 | 0.8362 | 0.9368 | 0.9249 | **0.9422** |
| Precision | 0.64 | 0.73 | 0.83 | 0.80 | **0.83** |
| Recall | 0.67 | 0.79 | 0.83 | 0.83 | **0.83** |
| F1 | 0.61 | 0.72 | 0.83 | 0.81 | **0.83** |
| Accuracy | 0.61 | 0.72 | 0.86 | 0.84 | **0.86** |

**Mô hình lựa chọn: XGBoost** — ROC-AUC cao nhất, cân bằng precision/recall, phù hợp khi có sự đánh đổi chi phí giữa hai loại sai lầm (bỏ sót khách churn vs cảnh báo nhầm khách trung thành).

### Top 10 Feature Importance (XGBoost)

1. **`số_dư_tiết_kiệm_heo_số_tháng_3_2021`** — hành vi gửi tiết kiệm quá khứ là chỉ báo mạnh nhất
2. `thành_phố_Hà_Nội`
3. `tình_trạng_hôn_nhân_Đã_kết_hôn`
4. `số_dư_tiết_kiệm_heo_số_tháng_6_2021`
5. `thành_phố_TP_HCM`
6. `thành_phố_Khác`
7. `tình_trạng_hôn_nhân_Độc_thân`
8. `giới_tính_Nữ`
9. `số_dư_tài_khoản_tiền_gửi`
10. `tổng_số_giao_dịch_tháng_6_2021`

Insight: **Hành vi tài chính trong quá khứ + nhân khẩu học + khu vực địa lý** là ba trụ cột dự báo. Mô hình "kể câu chuyện" nhất quán với EDA, không có biến nào bất thường nhảy lên — dấu hiệu của một mô hình lành mạnh.

---

## 💡 Đề xuất kinh doanh (gắn với mô hình)

Ba giải pháp được suy ra trực tiếp từ feature importance + insight EDA, không phải khuyến nghị chung chung:

### Giải pháp 1 — Cá nhân hóa theo giai đoạn cuộc đời và khu vực
*Nền tảng: Top features là nhân khẩu học + địa phương.*

- Chiến dịch riêng cho **Nữ độc thân Gen Y tại Hà Nội/TP.HCM** với các mục tiêu tiết kiệm gắn lifestyle: du lịch, quà cuối năm, quỹ dự phòng.
- Gói **"Heo Số Gia Đình"** / **"Tiết kiệm cho con"** cho nhóm đã kết hôn — phân khúc ổn định nhất theo mô hình.

### Giải pháp 2 — Hệ thống cảnh báo sớm & ưu đãi duy trì số dư
*Nền tảng: Số dư tiết kiệm là biến quan trọng nhất; rút tiền ròng là tín hiệu đỏ trước churn thực sự.*

- Giám sát thời gian thực: phát hiện khách giảm đăng nhập hoặc số dư sụt 20–30% trong tháng.
- Tự động kích hoạt kịch bản giữ chân: push notification cá nhân hóa, "lãi suất cộng thêm" ngắn hạn, hoàn tiền vào Heo Số khi nạp lại — **can thiệp trước khi khách rút ròng**.

### Giải pháp 3 — Tăng kết nối hệ sinh thái
*Nền tảng: Khách ít đối tác giao dịch & ít khối lượng giao dịch có nguy cơ churn cao.*

- Tích hợp **"Tiết kiệm tự động từ chi tiêu"**: mỗi giao dịch VietQR tự làm tròn và chuyển phần lẻ vào Heo Số → biến Heo Số thành phản xạ chi tiêu hàng ngày, không phải hành động riêng biệt.

---

## 🛠️ Tech Stack

| Hạng mục | Thư viện |
|---|---|
| Xử lý dữ liệu | `pandas`, `numpy` |
| Trực quan hóa | `matplotlib`, `seaborn`, `plotly` |
| Mô hình ML | `scikit-learn` (Logistic, Decision Tree, Random Forest), `xgboost`, `lightgbm` |
| Xử lý mất cân bằng | `imblearn` (SMOTE trong Pipeline) |
| Tinh chỉnh & đánh giá | `RandomizedSearchCV`, `StratifiedKFold`, `roc_auc_score` |

---

## 📁 Cấu trúc repository

```
.
├── README.md                       # File này
├── Group_5_Challenge_3.ipynb       # Notebook đầy đủ: EDA → cleaning → modeling
├── FDC_105_Group_5.pdf             # Slide báo cáo trình bày
└── data/                           # Dữ liệu thô (không commit, theo NDA cuộc thi)
```

---

## 🚀 Cách reproduce

```bash
# 1. Clone repo
git clone <repo-url>
cd heo-so-churn

# 2. Cài môi trường
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm imbalanced-learn jupyter

# 3. Đặt dữ liệu vào data/ rồi chạy notebook
jupyter notebook Group_5_Challenge_3.ipynb
```

---

## ⚠️ Hạn chế & Hướng phát triển

Trung thực về phạm vi dự án:

- **Dữ liệu chỉ có 2 mốc thời gian** (T3 và T6/2021) → định nghĩa churn dựa trên 2 điểm có thể nhạy cảm với nhiễu. Nếu có dữ liệu hàng tháng, có thể định nghĩa churn theo cửa sổ trượt (rolling window) thuyết phục hơn.
- **Dữ liệu đã được scale** trước khi nhận → không trả lại được giá trị tuyệt đối để diễn giải số tiền cụ thể.
- **Chưa triển khai mô hình lên hệ thống** — đây là bài toán phân tích, chưa có MLOps/serving.

Hướng tiếp theo:
- Thử thêm các kỹ thuật giải thích mô hình (SHAP) để hiểu *cách* từng feature tác động lên dự báo của từng khách hàng, không chỉ tổng thể.
- Xây dựng dashboard giám sát churn real-time (tương ứng Giải pháp 2).
- Thử mô hình survival analysis để dự báo *thời gian* đến khi churn, không chỉ phân loại có/không.

---

## 👥 Nhóm thực hiện

**Group 5 — FDC 105 (2025)**

Đóng góp cá nhân: [điền vai trò bạn đảm nhiệm trong nhóm — ví dụ: "phụ trách feature engineering, xây dựng pipeline XGBoost và phần Evaluation"]

---

## 📚 Tham khảo

> Nguyễn, T. T. H., & Trần, T. B. (2024). *Tác động của tiền gửi đến hiệu quả tài chính các ngân hàng thương mại Việt Nam.* Tạp chí Ngân hàng – Ngân hàng Nhà nước Việt Nam. https://tapchinganhang.gov.vn/tac-dong-cua-tien-gui-den-hieu-qua-tai-chinh-cac-ngan-hang-thuong-mai-viet-nam-465.html
