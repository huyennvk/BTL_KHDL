import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. BẮT BUỘC IMPORT CLASS CODE TAY TRƯỚC KHI LOAD MODEL
from lib import Node, DecisionTree, RandomForest

st.set_page_config(page_title="Định Giá Xe Nhóm 5", page_icon="🚗", layout="wide")

# 2. HÀM LOAD ASSETS VÀ HOTFIX PHÔNG CHỮ TỰ ĐỘNG
@st.cache_resource
def load_assets():
    # Load mô hình Random Forest code tay
    with open("best_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Load từ điển mã hóa Target Encoding
    with open("target_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    # 🔥 HOTFIX TRÊN WEB: Quét qua toàn bộ từ điển, ép các key lỗi "???" hoặc "სხვა" thành "Other"
    for col in ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels', 'Color', 'Leather interior']:
        if col in encoders:
            cleaned_dict = {}
            for k, v in encoders[col].items():
                if '?' in str(k) or 'სხვა' in str(k) or str(k).strip() == "":
                    cleaned_dict['Other'] = v
                else:
                    cleaned_dict[k] = v
            encoders[col] = cleaned_dict
            
    return model, encoders

# Khởi chạy nạp tài nguyên
try:
    model, encoders = load_assets()
    global_mean = encoders.get('global_mean', 15000)
except FileNotFoundError:
    st.error("❌ Thiếu file 'random_forest_model.pkl' hoặc 'target_encoders.pkl'. Vui lòng kiểm tra lại thư mục!")
    st.stop()

# ==========================================
# GIAO DIỆN NHẬP LIỆU CHÍNH
# ==========================================
st.title("🚗 Định Giá Xe Cũ (Nhóm 5)")
st.markdown("Nhập các thông số chiếc xe của bạn, AI sẽ phân tích xu hướng thị trường và đưa ra mức giá hợp lý nhất!")
st.markdown("---")

st.header("📋 Thông số kỹ thuật")

col1, col2, col3 = st.columns(3)

with col1:
    # Hãng xe (Sắp xếp A-Z)
    mfg_list = sorted(list(encoders.get('Manufacturer', {"TOYOTA": 0}).keys()))
    manufacturer = st.selectbox("Hãng xe", mfg_list)
    
    # DÒNG XE (Tự động lọc theo Hãng xe vừa chọn)
    mapping = encoders.get('mfg_model_mapping', {})
    
    if mapping and manufacturer in mapping:
        raw_models = mapping[manufacturer]
        model_list = sorted(list(set(['Other' if '?' in str(m) or 'სხვა' in str(m) else m for m in raw_models])))
    else:
        raw_models = list(encoders.get('Model', {"Other": 0}).keys())
        model_list = sorted(list(set(['Other' if '?' in str(m) or 'სხვა' in str(m) else m for m in raw_models])))
        
    car_model = st.selectbox("Dòng xe (Model)", model_list)
    
    category_list = sorted(list(encoders.get('Category', {"Sedan": 0}).keys()))
    category = st.selectbox("Phân khúc (Category)", category_list)
    
    prod_year = st.number_input("Năm sản xuất", min_value=1990, max_value=2024, value=2015)
    
    color_list = sorted(list(encoders.get('Color', {"Black": 0}).keys()))
    color = st.selectbox("Màu sắc ngoại thất", color_list)

with col2:
    engine_volume = st.number_input("Dung tích động cơ (Liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    is_turbo = st.selectbox("Động cơ có Turbo tăng áp không?", ["No", "Yes"])
    
    fuel_list = sorted(list(encoders.get('Fuel type', {"Petrol": 0}).keys()))
    fuel_type = st.selectbox("Loại nhiên liệu sử dụng", fuel_list)
    
    leather_list = sorted(list(encoders.get('Leather interior', {"Yes": 0}).keys()))
    leather = st.selectbox("Nội thất bọc da?", leather_list)
    
    cylinders = st.number_input("Số xy-lanh (Cylinders)", min_value=2, max_value=16, value=4)

with col3:
    mileage = st.number_input("Số Kilomet đã đi (Mileage)", min_value=0, value=50000, step=1000)
    
    gear_list = sorted(list(encoders.get('Gear box type', {"Automatic": 0}).keys()))
    gear_box = st.selectbox("Hộp số xe", gear_list)
    
    drive_list = sorted(list(encoders.get('Drive wheels', {"Front": 0}).keys()))
    drive_wheels = st.selectbox("Hệ thống dẫn động", drive_list)
    
    # 🔥 ĐÃ CẬP NHẬT: Thanh trượt (slider) đổi thành Hộp chọn (selectbox) từ 0 đến 16 túi khí
    airbags = st.selectbox("Số lượng túi khí an toàn", list(range(17)), index=4) 
    
    levy = st.number_input("Thuế nhập khẩu / Phí trước bạ (Levy - USD)", min_value=0, value=1000)

st.markdown("---")

# ==========================================
# XỬ LÝ LOGIC KHI BẤM NÚT ĐỰ ĐOÁN
# ==========================================
if st.button("🚀 KÍCH HOẠT AI ĐỊNH GIÁ", use_container_width=True):
    with st.spinner("Mô hình Random Forest đang xử lý dữ liệu..."):
        
        encoded_manufacturer = encoders['Manufacturer'].get(manufacturer, global_mean)
        encoded_model = encoders['Model'].get(car_model, global_mean)
        encoded_category = encoders['Category'].get(category, global_mean)
        encoded_leather = encoders['Leather interior'].get(leather, global_mean)
        encoded_fuel = encoders['Fuel type'].get(fuel_type, global_mean)
        encoded_gear = encoders['Gear box type'].get(gear_box, global_mean)
        encoded_drive = encoders['Drive wheels'].get(drive_wheels, global_mean)
        encoded_color = encoders['Color'].get(color, global_mean)
        
        # Khớp với lựa chọn "Yes"/"No" ở selectbox giao diện
        turbo_val = 1 if is_turbo == "Yes" else 0
        # Tính toán tuổi xe Car_Age
        car_age = 2024 - prod_year
        
        # Sắp xếp đúng 15 cột theo thứ tự train: Levy, Manufacturer, Model, Category, Fuel type, Engine volume, Mileage, Leather interior, Cylinders, Gear box type, Drive wheels, Color, Airbags, is_Turbo, Car_Age
        X_input = np.array([[
            levy,                   # 1. Levy
            encoded_manufacturer,   # 2. Manufacturer
            encoded_model,          # 3. Model
            encoded_category,       # 4. Category
            encoded_fuel,           # 5. Fuel type
            engine_volume,          # 6. Engine volume
            mileage,                # 7. Mileage
            encoded_leather,        # 8. Leather interior
            cylinders,              # 9. Cylinders
            encoded_gear,           # 10. Gear box type
            encoded_drive,          # 11. Drive wheels
            encoded_color,          # 12. Color
            airbags,                # 13. Airbags
            turbo_val,              # 14. is_Turbo
            car_age                 # 15. Car_Age
        ]])
        
        try:
            pred_price = model.predict(X_input)[0]
            st.success("🎉 PHÂN TÍCH THỊ TRƯỜNG HOÀN TẤT!")
            st.markdown(f"<h1 style='text-align: center; color: #ff4b4b;'>${pred_price:,.0f}</h1>", unsafe_allow_html=True)
            st.info("💡 Kết quả được tính toán dựa trên thuật toán Random Forest, phản ánh mức độ khấu hao thực tế.")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý cấu trúc mô hình: {e}")