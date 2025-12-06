import os
import numpy as np
import tensorflow as tf # Vẫn cần TF nhưng chỉ dùng phần Lite
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import eventlet

# ================================================================
# 1. KHỞI TẠO ỨNG DỤNG
# ================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here_v5_realtime'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", max_decode_packets=200)

# ================================================================
# 2. TẢI MÔ HÌNH TFLITE (SIÊU NHẸ)
# ================================================================

try:
    # Tải TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Lấy thông tin input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Đã tải model TFLite thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình TFLite: {e}")
    exit()

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
# 3. HÀM HỖ TRỢ
# ================================================================

def resample_keypoints(all_keypoints, target_frames=60):
    if len(all_keypoints) == 0:
        return np.zeros((target_frames, 126))
    indices = np.linspace(0, len(all_keypoints) - 1, target_frames, dtype=int)
    resampled = [all_keypoints[i] for i in indices]
    return np.array(resampled)

# ================================================================
# 4. ROUTE & SOCKET
# ================================================================

@app.route('/')
def index():
    return render_template('indexxx.html')

FPS = 30
PREPARE_SECONDS = 3
RECORDING_SECONDS = 5
PREPARE_FRAMES = PREPARE_SECONDS * FPS
RECORDING_FRAMES = RECORDING_SECONDS * FPS

user_state = {
    'app_state': 'IDLE',
    'recording_sequence': [],
    'timer': 0
}

@socketio.on('connect')
def handle_connect():
    global user_state
    user_state = {'app_state': 'IDLE', 'recording_sequence': [], 'timer': 0}
    print(f'Client {request.sid} connected. State reset.')
    emit('status', {'state': 'IDLE', 'message': 'Chờ phát hiện tay...'})

@socketio.on('keypoints')
def handle_keypoints(data):
    global user_state
    try:
        keypoints = np.array(data['keypoints'])
        has_hands = np.any(keypoints != 0) 
        app_state = user_state['app_state']
        
        if has_hands:
            if app_state == 'IDLE':
                user_state['app_state'] = 'COUNTDOWN'
                user_state['timer'] = PREPARE_FRAMES
                emit('status', {'state': 'COUNTDOWN', 'time': PREPARE_SECONDS})
            elif app_state == 'COUNTDOWN':
                user_state['timer'] -= 1
                remaining_sec = (user_state['timer'] // FPS) + 1
                emit('status', {'state': 'COUNTDOWN', 'time': remaining_sec})
                if user_state['timer'] <= 0:
                    user_state['app_state'] = 'RECORDING'
                    user_state['timer'] = RECORDING_FRAMES
                    user_state['recording_sequence'] = [] 
                    emit('status', {'state': 'RECORDING', 'time': RECORDING_SECONDS})
            elif app_state == 'RECORDING':
                user_state['recording_sequence'].append(keypoints)
                user_state['timer'] -= 1
                remaining_sec = (user_state['timer'] // FPS) + 1
                emit('status', {'state': 'RECORDING', 'time': remaining_sec})
                
                if user_state['timer'] <= 0:
                    user_state['app_state'] = 'PREDICTING'
                    emit('status', {'state': 'PREDICTING', 'message': 'Đang xử lý...'})
                    socketio.sleep(0.5)

                    # --- DỰ ĐOÁN BẰNG TFLITE ---
                    try:
                        # 1. Chuẩn bị dữ liệu
                        sequence = resample_keypoints(user_state['recording_sequence'], 60)
                        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

                        # 2. Đặt dữ liệu vào TFLite
                        interpreter.set_tensor(input_details[0]['index'], input_data)

                        # 3. Chạy dự đoán
                        interpreter.invoke()

                        # 4. Lấy kết quả
                        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        predicted_index = np.argmax(output_data)
                        predicted_action = actions[predicted_index]
                        
                        print(f"Kết quả TFLite: {predicted_action}", flush=True)
                        emit('prediction', {'sentence': predicted_action})
                        
                    except Exception as e:
                        print(f"LỖI TFLITE: {e}", flush=True)
                        emit('prediction', {'sentence': f"Lỗi: {str(e)}"})
                    
                    user_state['app_state'] = 'IDLE'
                    emit('status', {'state': 'IDLE', 'message': 'Chờ phát hiện tay...'})

        else:
            if app_state == 'COUNTDOWN' or app_state == 'RECORDING':
                user_state['app_state'] = 'IDLE'
                emit('status', {'state': 'IDLE', 'message': 'Đã hủy.'})

    except Exception as e:
        print(f"LỖI CHUNG: {e}", flush=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)