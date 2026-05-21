import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# Import các class cần thiết từ file để pickle có thể giải tuần tự mô hình thành công
from lib import RandomForest, DecisionTree, Node

# 1. Đọc dữ liệu tập Test đã làm sạch
test_data = pd.read_csv("car_price_test_clean.csv")
feature_cols = [c for c in test_data.columns if c != "Price"]
X_test = test_data[feature_cols].values
y_test = test_data["Price"].values

# 2. Tải mô hình tự viết đã train và lưu thành công
with open('random_forest_model.pkl', 'rb') as f:
    model_custom = pickle.load(f)

# 3. Dự đoán giá xe trên tập Test
y_pred = model_custom.predict(X_test)

# 4. Tính toán các chỉ số đánh giá tương tự mô hình của sklearn
mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
r2    = r2_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
mape  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
n     = len(y_test)
p     = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 5. In kết quả đánh giá chi tiết
print("📊 ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST TỰ VIẾT (CUSTOM)")
print("=" * 50)
print(f"  MAE          : {mae:,.2f} $")
print(f"  MSE          : {mse:,.2f}")
print(f"  RMSE         : {rmse:,.2f} $")
print(f"  Median AE    : {medae:,.2f} $")
print(f"  MAPE         : {mape:.2f} %")
print(f"  R2 Score     : {r2:.4f}")
print(f"  Adjusted R2  : {adj_r2:.4f}")
