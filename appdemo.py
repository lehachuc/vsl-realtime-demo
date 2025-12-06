import os
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import eventlet # Cần thiết cho server

# ================================================================
# 1. KHỞI TẠO ỨNG DỤNG FLASK VÀ SOCKETIO
# ================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here_v5_realtime'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*") 

# ================================================================
# 2. TẢI MÔ HÌNH VÀ CÁC THIẾT LẬP
# ================================================================

try:
    model = load_model("vsl_lstm_model_v2.h5")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# mp_holistic và holistic_video đã được xóa (không cần cho upload)

actions = [
    'ban dang lam gi', 'ban di dau the', 'ban hieu ngon ngu ky hieu khong', 'ban hoc lop may', 'ban khoe khong', 
    'ban muon gio roi', 'ban phai canh giac', 'ban ten la gi', 'ban tien bo day', 'ban trong cau co the', 
    'bo me toi cung la nguoi Diec', 'cai nay bao nhieu tien', 'cai nay la cai gi', 'cam on', 'cap cuu', 'chuc mung', 
    'chung toi giao tiep voi nhau bang ngon ngu ky hieu', 'con yeu me', 'cong viec cua ban la gi', 
    'hen gap lai cac ban', 'mon nay khong ngon', 'toi bi chong mat', 'toi bi cuop', 'toi bi dau dau', 
    'toi bi dau hong', 'toi bi ket xe', 'toi bi lac', 'toi bi phan biet doi xu', 'toi cam thay rat hoi hop', 
    'toi cam thay rat vui', 'toi can an sang', 'toi can di ve sinh', 'toi can gap bac si', 'toi can phien dich', 
    'toi can thuoc', 'toi dang an sang', 'toi dang buon', 'toi dang o ben xe', 'toi dang o cong vien', 
    'toi dang phai cach ly', 'toi dang phan van', 'toi di sieu thi', 'toi di toi Ha Noi', 'toi doc kem', 
    'toi khoi benh roi', 'toi khong dem theo tien', 'toi khong hieu', 'toi khong quan tam', 'toi la hoc sinh', 
    'toi la nguoi Diec', 'toi la tho theu', 'toi lam viec o cua hang', 'toi nham dia chi', 'toi song o Ha Noi', 
    'toi thay doi bung', 'toi thay nho ban', 'toi thich an mi', 'toi thich phim truyen', 'toi viet kem', 'xin chao'
]

# ================================================================
# 3. HÀM HỖ TRỢ (Chỉ giữ lại Resample)
# ================================================================

# (Hàm extract_keypoints đã bị xóa vì không dùng ở backend)

def resample_keypoints(all_keypoints, target_frames=60):
    """Giãn/Nén số keyframes về 60 frames"""
    if len(all_keypoints) == 0:
        return np.zeros((target_frames, 126))
    
    indices = np.linspace(0, len(all_keypoints) - 1, target_frames, dtype=int)
    resampled = [all_keypoints[i] for i in indices]
    return np.array(resampled)

# ================================================================
# 4. ROUTE CHÍNH (Trang web)
# ================================================================

@app.route('/')
def index():
    # Đổi tên file html cho gọn (tùy bạn)
    return render_template('indexxx.html') 

# ================================================================
# 5. API CHO "UPLOAD VIDEO" (ĐÃ XÓA)
# ================================================================
# (Toàn bộ route @app.route('/upload') đã bị xóa)
# ================================================================

# ================================================================
# 6. API CHO "REAL-TIME" (SocketIO) - Giữ nguyên
# ================================================================

# Các hằng số cho máy trạng thái (state machine)    
FPS = 30 # Giả định
PREPARE_SECONDS = 3
RECORDING_SECONDS = 5
PREPARE_FRAMES = PREPARE_SECONDS * FPS
RECORDING_FRAMES = RECORDING_SECONDS * FPS

# Biến toàn cục để lưu trạng thái
user_state = {
    'app_state': 'IDLE',          # IDLE, COUNTDOWN, RECORDING, PREDICTING
    'recording_sequence': [],
    'timer': 0
}

@socketio.on('connect')
def handle_connect():
    global user_state
    user_state = {'app_state': 'IDLE', 'recording_sequence': [], 'timer': 0}
    print(f'Client {request.sid} connected. State reset.')
    emit('status', {'state': 'IDLE', 'message': 'Chờ phát hiện tay...'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client {request.sid} disconnected')

@socketio.on('keypoints')
def handle_keypoints(data):
    """Đây là nơi tái hiện logic v5.0 (Ghi hình 5 giây)"""
    global user_state
    
    keypoints = np.array(data['keypoints'])
    has_hands = np.any(keypoints != 0) # Sửa lỗi 1 tay
    
    app_state = user_state['app_state']
    
    if has_hands:
        if app_state == 'IDLE':
            # 1. Bắt đầu đếm ngược (chuẩn bị)
            user_state['app_state'] = 'COUNTDOWN'
            user_state['timer'] = PREPARE_FRAMES
            emit('status', {'state': 'COUNTDOWN', 'time': PREPARE_SECONDS})
            
        elif app_state == 'COUNTDOWN':
            # 2. Đang đếm ngược
            user_state['timer'] -= 1
            remaining_sec = (user_state['timer'] // FPS) + 1
            emit('status', {'state': 'COUNTDOWN', 'time': remaining_sec})
            
            if user_state['timer'] <= 0:
                # Bắt đầu ghi hình
                user_state['app_state'] = 'RECORDING'
                user_state['timer'] = RECORDING_FRAMES
                user_state['recording_sequence'] = [] # Xóa bộ đệm ghi hình
                emit('status', {'state': 'RECORDING', 'time': RECORDING_SECONDS})
        
        elif app_state == 'RECORDING':
            # 3. Đang ghi hình (5 giây)
            user_state['recording_sequence'].append(keypoints) # Lưu frame
            user_state['timer'] -= 1
            remaining_sec = (user_state['timer'] // FPS) + 1
            emit('status', {'state': 'RECORDING', 'time': remaining_sec})
            
            if user_state['timer'] <= 0:
                # Hết 5 giây -> Bắt đầu dự đoán
                user_state['app_state'] = 'PREDICTING'
                emit('status', {'state': 'PREDICTING', 'message': 'Đang xử lý...'})
                socketio.sleep(0.5) # Cho client kịp nhận trạng thái

                # Lấy mẫu lại 150 frames -> 60 frames
                sequence = resample_keypoints(user_state['recording_sequence'], 60)
                
                # Dự đoán
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_action = actions[np.argmax(res)]
                
                # Gửi kết quả và reset
                emit('prediction', {'sentence': predicted_action})
                user_state['app_state'] = 'IDLE'
                emit('status', {'state': 'IDLE', 'message': 'Chờ phát hiện tay...'})

    else: # Không phát hiện thấy tay
        if app_state == 'COUNTDOWN' or app_state == 'RECORDING':
            # Nếu đang đếm ngược hoặc đang ghi mà hạ tay -> Hủy
            user_state['app_state'] = 'IDLE'
            emit('status', {'state': 'IDLE', 'message': 'Đã hủy. Chờ phát hiện tay...'})

# ================================================================
# 7. CHẠY ỨNG DỤNG
# ================================================================

if __name__ == '__main__':
    print("Starting Flask server... Mở http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)