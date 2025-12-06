import tensorflow as tf

# 1. Tải model gốc (.h5)
model = tf.keras.models.load_model("vsl_lstm_model_v2.h5")

# 2. Chuyển đổi sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
]
tflite_model = converter.convert()

# 3. Lưu file model mới (.tflite)
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Thành công! Đã tạo file 'model.tflite'")