import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🎯 BÀI TOÁN PHÂN LOẠI STRESS SINH VIÊN VỚI PCA")
print("=" * 60)

# =============================================================================
# BƯỚC 1: THU THẬP VÀ MÔ TẢ DỮ LIỆU
# =============================================================================
print("\n🔍 BƯỚC 1: ĐANG TẢI VÀ PHÂN TÍCH DỮ LIỆU...")

# Load dataset
df = pd.read_excel('dataset.xlsx')

print("📊 THÔNG TIN DATASET:")
print(f"- Số lượng mẫu: {df.shape[0]}")
print(f"- Số lượng đặc trưng: {df.shape[1]}")
print(f"- Các đặc trưng: {list(df.columns)}")

# Kiểm tra phân phối target
print(f"\n📈 PHÂN PHỐI STRESS_LEVEL:")
stress_distribution = df['stress_level'].value_counts().sort_index()
print(stress_distribution)

# Hiển thị ý nghĩa các mức độ stress
stress_mapping = {0: "Thấp (Low)", 1: "Trung bình (Moderate)", 2: "Cao (High)"}
for level, count in stress_distribution.items():
    print(f"  - Level {level} ({stress_mapping[level]}): {count} mẫu ({count/len(df)*100:.1f}%)")

# =============================================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================
print("\n" + "=" * 50)
print("⚙️ BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 50)

# 2.1. Kiểm tra và xử lý giá trị thiếu
print("\n🔎 KIỂM TRA GIÁ TRỊ THIẾU:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() > 0:
    print("🔄 Đang xử lý giá trị thiếu...")
    # Thay thế giá trị thiếu bằng median cho numerical features
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print("✅ Đã xử lý giá trị thiếu")
else:
    print("✅ Không có giá trị thiếu")
    df_imputed = df.copy()

# 2.2. Kiểm tra và mã hóa nhãn phân loại
print("\n🔠 KIỂM TRA MÃ HÓA NHÃN:")
print(f"- Giá trị duy nhất trong stress_level: {sorted(df_imputed['stress_level'].unique())}")
print(f"- Kiểu dữ liệu: {df_imputed['stress_level'].dtype}")

# Kiểm tra xem nhãn đã được mã hóa số chưa
if df_imputed['stress_level'].dtype == 'object':
    print("🔄 Đang mã hóa nhãn phân loại...")
    label_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    df_imputed['stress_level'] = df_imputed['stress_level'].map(label_mapping)
    print("✅ Đã mã hóa nhãn phân loại")
else:
    print("✅ Nhãn đã được mã hóa số")

# 2.3. Chuẩn bị features và target
X = df_imputed.drop(columns=['stress_level'])
y = df_imputed['stress_level']

print(f"\n📋 THÔNG TIN SAU TIỀN XỬ LÝ:")
print(f"- Features: {X.shape[1]} đặc trưng")
print(f"- Target: {len(np.unique(y))} classes")

# 2.4. Chuẩn hóa dữ liệu
print("\n📏 CHUẨN HÓA DỮ LIỆU...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Đã chuẩn hóa dữ liệu với StandardScaler")

# 2.5. Feature Extraction với PCA - PC1 và PC2
print("\n🔧 FEATURE EXTRACTION VỚI PCA (PC1, PC2)...")
pca = PCA(n_components=2)  # Chỉ lấy 2 components chính
X_pca = pca.fit_transform(X_scaled)

print("✅ Đã thực hiện PCA:")
print(f"- Số components: {pca.n_components_}")
print(f"- Phương sai được giữ: {pca.explained_variance_ratio_.sum():.3f}")
print(f"- Phương sai PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"- Phương sai PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"- Kích thước dữ liệu sau PCA: {X_pca.shape}")

# Hiển thị các features đóng góp vào PC1 và PC2
print("\n📊 PHÂN TÍCH COMPONENTS:")
feature_names = X.columns
pca_components_df = pd.DataFrame({
    'Feature': feature_names,
    'PC1': pca.components_[0],
    'PC2': pca.components_[1],
    'PC1_Abs': np.abs(pca.components_[0]),
    'PC2_Abs': np.abs(pca.components_[1])
})

print("\n🔝 TOP 5 FEATURES QUAN TRỌNG CHO PC1:")
print(pca_components_df.nlargest(5, 'PC1_Abs')[['Feature', 'PC1']].to_string(index=False))

print("\n🔝 TOP 5 FEATURES QUAN TRỌNG CHO PC2:")
print(pca_components_df.nlargest(5, 'PC2_Abs')[['Feature', 'PC2']].to_string(index=False))

# =============================================================================
# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH
# =============================================================================
print("\n" + "=" * 50)
print("🤖 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
print("=" * 50)

print("\n📊 CHIA DỮ LIỆU TRAIN/TEST...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


print(f"- Tập train (gốc): {X_train.shape[0]} mẫu, {X_train.shape[1]} features")
print(f"- Tập test (gốc): {X_test.shape[0]} mẫu, {X_test.shape[1]} features")
print("✅ Đã chia dữ liệu")

print("\n🎯 HUẤN LUYỆN MÔ HÌNH KNN...")

param_grid = {
    'n_neighbors': range(3, 16, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("🔍 TÌM THAM SỐ TỐI ƯU CHO DỮ LIỆU GỐC...")
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)



best_knn = grid_search.best_estimator_

print(f"\n✅ THAM SỐ TỐI ƯU : {grid_search.best_params_}")

# =============================================================================
## =============================================================================
# BƯỚC 4: ĐÁNH GIÁ
# =============================================================================

print("\n" + "=" * 50)
print("📊 BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH KNN")
print("=" * 50)

# Dự đoán trên tập test
y_pred = best_knn.predict(X_test)

# Tính độ chính xác
test_accuracy = accuracy_score(y_test, y_pred)

print("📈 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH KNN:")
print(f"🔹 Độ chính xác (Accuracy): {test_accuracy:.4f}")

print("\n📋 BÁO CÁO CHI TIẾT:")
print(classification_report(y_test, y_pred, target_names=['Thấp', 'Trung bình', 'Cao']))


# =============================================================================
# TRỰC QUAN HÓA – XÓA HOÀN TOÀN TỶ LỆ PHƯƠNG SAI
# =============================================================================

print("\n🎨 VẼ BIỂU ĐỒ PCA...")

plt.figure(figsize=(15, 10))

# ============================================
# 1️⃣ SCATTER PLOT PCA
# ============================================
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Mức độ Stress')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Phân bố dữ liệu trên PC1 và PC2')
plt.grid(True, alpha=0.3)

# ============================================
# 2️⃣ BIỂU ĐỒ ĐÓNG GÓP THÀNH PHẦN
# ============================================
plt.subplot(2, 2, 2)
top_features_pc1 = pca_components_df.nlargest(8, 'PC1_Abs')

sns.heatmap(
    top_features_pc1[['PC1', 'PC2']],
    annot=True,
    cmap='coolwarm',
    center=0,
    yticklabels=top_features_pc1['Feature']
)
plt.title('Top Features cho PC1 & PC2')

# ============================================
# 3️⃣ CONFUSION MATRIX (KNN GỐC)
# ============================================
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Thấp', 'Trung bình', 'Cao'],
    yticklabels=['Thấp', 'Trung bình', 'Cao']
)

plt.title('Ma trận nhầm lẫn (KNN Gốc)')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')

plt.tight_layout()
plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# BƯỚC 5: LƯU MÔ HÌNH
# =============================================================================
print("\n" + "=" * 50)
print("💾 LƯU MÔ HÌNH")
print("=" * 50)

# Lưu mô hình KNN gốc
joblib.dump(best_knn, 'stress_knn_model.pkl')

# Lưu scaler
joblib.dump(scaler, 'scaler.pkl')

# Lưu tham số tốt nhất
joblib.dump(grid_search.best_params_, 'best_params.pkl')

print("💾 Đã lưu stress_knn_model.pkl, scaler.pkl, best_params.pkl")
print("=" * 60)
print("🎯 HOÀN THÀNH – KHÔNG SỬ DỤNG PCA CHO MÔ HÌNH")
print("=" * 60)
