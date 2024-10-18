# Hàm hỗ trợ để lưu mô hình vào file
import pickle

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

# Hàm hỗ trợ để tải mô hình từ file
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
