import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Tiêu đề ứng dụng
st.set_page_config(page_title="DỰ ĐOÁN MỰC ĐỘ STRESS", page_icon="😊", layout="wide")

st.title("🎓 Dự đoán Mức độ Stress cho Sinh viên")
st.markdown("Ứng dụng sử dụng mô hình Machine Learning để dự đoán mức độ stress dựa trên các yếu tố học tập và cuộc sống.")

# Sidebar cho thông tin giới thiệu
with st.sidebar:
    st.header("ℹ️ Giới thiệu")
    st.markdown("""
    **Các mức độ Stress:**
    - 🟢 **0 - Thấp**: Quản lý tốt, ít căng thẳng
    - 🟡 **1 - Trung bình**: Có căng thẳng nhưng trong tầm kiểm soát  
    - 🔴 **2 - Cao**: Cần quan tâm và có biện pháp hỗ trợ
    """)
    
    st.markdown("---")
    st.markdown("**Hướng dẫn:**")  # ĐÃ SỬA LỖI Ở ĐÂY
    st.markdown("1. Điền thông tin vào các ô bên dưới")
    st.markdown("2. Nhấn nút 'Dự đoán'")
    st.markdown("3. Xem kết quả và lời khuyên")

# Tải mô hình và scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('stress_knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("❌ Không tìm thấy mô hình đã huấn luyện. Vui lòng chạy train_model.py trước!")
        return None, None

model, scaler = load_model()

if model is not None:
    # Tạo form nhập liệu
    st.header("📝 Thông tin cá nhân")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧠 Sức khỏe Tâm lý")
        anxiety_level = st.slider("Mức độ lo âu", 0, 21, 10)
        self_esteem = st.slider("Lòng tự trọng", 0, 30, 15)
        mental_health_history = st.selectbox("Tiền sử sức khỏe tâm lý", [0, 1])
        depression = st.slider("Mức độ trầm cảm", 0, 27, 10)
        
    with col2:
        st.subheader("💪 Sức khỏe Thể chất")
        headache = st.slider("Tần suất đau đầu", 0, 5, 2)
        blood_pressure = st.slider("Huyết áp", 1, 3, 2)
        sleep_quality = st.slider("Chất lượng giấc ngủ", 0, 5, 3)
        breathing_problem = st.slider("Vấn đề hô hấp", 0, 5, 2)
        
    with col3:
        st.subheader("🏠 Môi trường sống")
        noise_level = st.slider("Mức độ ồn", 0, 5, 2)
        living_conditions = st.slider("Điều kiện sống", 1, 5, 3)
        safety = st.slider("Cảm giác an toàn", 1, 5, 3)
        basic_needs = st.slider("Nhu cầu cơ bản", 1, 5, 3)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("📚 Học tập")
        academic_performance = st.slider("Kết quả học tập", 0, 5, 3)
        study_load = st.slider("Khối lượng học tập", 0, 5, 3)
        teacher_student_relationship = st.slider("Quan hệ GV-SV", 0, 5, 3)
        
    with col5:
        st.subheader("🎯 Tương lai & Xã hội")
        future_career_concerns = st.slider("Lo lắng nghề nghiệp", 0, 5, 3)
        social_support = st.slider("Hỗ trợ xã hội", 0, 3, 2)
        peer_pressure = st.slider("Áp lực bạn bè", 1, 5, 3)
        
    with col6:
        st.subheader("⚽ Hoạt động khác")
        extracurricular_activities = st.slider("Hoạt động ngoại khóa", 0, 5, 2)
        bullying = st.slider("Bắt nạt", 1, 5, 2)

    # Nút dự đoán
    if st.button("🎯 Dự đoán Mức độ Stress", type="primary", use_container_width=True):
        # Tạo dataframe từ input
        input_data = pd.DataFrame({
            'anxiety_level': [anxiety_level],
            'self_esteem': [self_esteem],
            'mental_health_history': [mental_health_history],
            'depression': [depression],
            'headache': [headache],
            'blood_pressure': [blood_pressure],
            'sleep_quality': [sleep_quality],
            'breathing_problem': [breathing_problem],
            'noise_level': [noise_level],
            'living_conditions': [living_conditions],
            'safety': [safety],
            'basic_needs': [basic_needs],
            'academic_performance': [academic_performance],
            'study_load': [study_load],
            'teacher_student_relationship': [teacher_student_relationship],
            'future_career_concerns': [future_career_concerns],
            'social_support': [social_support],
            'peer_pressure': [peer_pressure],
            'extracurricular_activities': [extracurricular_activities],
            'bullying': [bullying]
        })
        
        # Chuẩn hóa dữ liệu
        input_scaled = scaler.transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Hiển thị kết quả
        st.markdown("---")
        st.header("📊 Kết quả dự đoán")
        
        # Hiển thị theo mức độ
        col_result1, col_result2, col_result3 = st.columns(3)
        
        stress_levels = {
            0: {"name": "THẤP", "emoji": "🟢", "color": "green"},
            1: {"name": "TRUNG BÌNH", "emoji": "🟡", "color": "orange"}, 
            2: {"name": "CAO", "emoji": "🔴", "color": "red"}
        }
        
        level_info = stress_levels[prediction]
        
        with col_result2:
            st.markdown(f"<h1 style='text-align: center; color: {level_info['color']};'>{level_info['emoji']} {level_info['name']}</h1>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Mức độ: {prediction}</h3>", unsafe_allow_html=True)
        
        # Hiển thị xác suất
        st.subheader("📈 Xác suất dự đoán")
        prob_cols = st.columns(3)
        for i, (col, level) in enumerate(zip(prob_cols, stress_levels.values())):
            with col:
                percent = prediction_proba[i] * 100
                col.metric(
                    label=f"{level['emoji']} {level['name']}", 
                    value=f"{percent:.1f}%"
                )
                st.progress(float(prediction_proba[i]))
        
        # Lời khuyên
        st.markdown("---")
        st.header("💡 Lời khuyên")
        
        advice = {
            0: """
            **🎉 Bạn đang quản lý stress rất tốt!**
            - Duy trì lối sống lành mạnh hiện tại
            - Tiếp tục cân bằng giữa học tập và giải trí
            - Chia sẻ kinh nghiệm với bạn bè
            """,
            1: """
            **⚠️ Bạn đang có mức độ stress trung bình**
            - Nghỉ ngơi nhiều hơn, ngủ đủ 7-8 tiếng/ngày
            - Tập thể dục nhẹ nhàng 30 phút mỗi ngày
            - Chia sẻ cảm xúc với người thân, bạn bè
            - Sắp xếp thời gian học tập hợp lý
            """,
            2: """
            **🚨 Bạn đang có mức độ stress cao**
            - **Cần tìm sự giúp đỡ ngay:** Phòng công tác sinh viên, chuyên gia tâm lý
            - Tham gia các hoạt động thư giãn: yoga, thiền
            - Giảm tải khối lượng công việc/học tập
            - Ngủ đủ giấc và ăn uống điều độ
            - Tránh các chất kích thích
            """
        }
        
        st.info(advice[prediction])
        
        # Gợi ý cải thiện dựa trên input
        st.subheader("🎯 Gợi ý cải thiện cụ thể")
        
        improvement_suggestions = []
        if sleep_quality <= 2:
            improvement_suggestions.append("💤 **Cải thiện giấc ngủ:** Ngủ đủ 7-8 tiếng, tránh sử dụng điện thoại trước khi ngủ")
        if anxiety_level >= 15:
            improvement_suggestions.append("🧘 **Giảm lo âu:** Tập hít thở sâu, chia nhỏ công việc lớn")
        if social_support <= 1:
            improvement_suggestions.append("👥 **Tăng kết nối xã hội:** Tham gia câu lạc bộ, trò chuyện với bạn bè")
        if study_load >= 4:
            improvement_suggestions.append("📚 **Giảm tải học tập:** Lập kế hoạch học tập, ưu tiên việc quan trọng")
        if extracurricular_activities <= 1:
            improvement_suggestions.append("⚽ **Tăng hoạt động ngoại khóa:** Tham gia thể thao, sở thích cá nhân")
        
        if improvement_suggestions:
            for suggestion in improvement_suggestions:
                st.write(f"- {suggestion}")
        else:
            st.success("🌟 Các chỉ số của bạn khá cân bằng! Hãy duy trì lối sống hiện tại.")

else:

    st.warning("⚠️ Vui lòng chạy file 'train_model.py' trước để huấn luyện mô hình!")
