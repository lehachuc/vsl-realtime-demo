import os
import numpy as np
from tqdm import tqdm

# Thêm chữ r ở đầu
DATA_PATH = r"D:\ky1nam4\tttn\Sign-Language-Translation-main\Sign-Language-Translation-main\Sign Language Translator\Data"# ================================================================

# Cài đặt các tham số
# Số lượng sequence cho mỗi hành động

NO_SEQUENCES = 60
# Số lượng frames (file .npy) trong mỗi sequence
NO_FRAMES = 60
# Kích thước của đặc trưng
FEATURE_LENGTH = 126

# Lấy danh sách các hành động (tên 60 thư mục)
# Chúng ta sắp xếp (sort) để đảm bảo thứ tự nhãn nhất quán
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

# Tạo bộ ánh xạ từ tên hành động sang số (0, 1, 2...)
label_map = {action_name: label_index for label_index, action_name in enumerate(actions)}

print(f"Phát hiện {len(actions)} hành động:")
print(label_map)

# # Khởi tạo các danh sách để lưu dữ liệu
# sequences, labels = [], []

# Sử dụng tqdm để xem tiến trình
for action in tqdm(actions, desc="Đang xử lý các hành động"):
    for sequence_index in range(NO_SEQUENCES):
        # Mảng tạm thời để lưu 60 frames của 1 sequence
        window = []
        
        for frame_index in range(NO_FRAMES):
            file_path = os.path.join(DATA_PATH, action, str(sequence_index), f"{frame_index}.npy")
            
            try:
                # Tải file .npy
                res = np.load(file_path)
                window.append(res)
            except Exception as e:
                print(f"Lỗi khi đọc file: {file_path} | Lỗi: {e}")
                # Nếu file lỗi, thêm một mảng 0 để giữ đúng shape
                window.append(np.zeros(FEATURE_LENGTH))
        
        # Thêm chuỗi 60 frames (shape 60, 126) vào danh sách
        sequences.append(window)
        # Thêm nhãn (số) tương ứng
        labels.append(label_map[action])

print(f"\nĐã tải xong {len(sequences)} chuỗi.")

# Chuyển đổi danh sách Python sang mảng NumPy
# X sẽ có shape (3600, 60, 126)
X_data = np.array(sequences)

# y sẽ có shape (3600,)
y_data = np.array(labels)

print(f"Shape của X_data: {X_data.shape}")
print(f"Shape của y_data: {y_data.shape}")

# Lưu 2 mảng này ra file
output_path_X = "X_data.npy"
output_path_y = "y_data.npy"

np.save(output_path_X, X_data)
np.save(output_path_y, y_data)

print(f"\Đã lưu dữ liệu thành công!")
print(f"File features: {output_path_X}")
print(f"File labels: {output_path_y}")
print(label_map)
