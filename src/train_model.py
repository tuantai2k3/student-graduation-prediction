import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# Tạo thư mục 'figures' nếu chưa tồn tại
if not os.path.exists('../figures'):
    os.makedirs('../figures')

# Tạo thư mục 'models' nếu chưa tồn tại
if not os.path.exists('../models'):
    os.makedirs('../models')

# Đọc dữ liệu từ file CSV
data = pd.read_csv(r'C:\Users\Thai Tuan Tai\Desktop\vscode\Python\student-graduation-prediction\data\student_data.csv')

# Chuẩn bị dữ liệu (giả định các cột có tên 'average_grade', 'credits_earned', 'failed_subjects')
X = data[['average_grade', 'credits_earned', 'failed_subjects']]
y = data['graduated']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

# Huấn luyện mô hình cây quyết định
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

# Lưu mô hình cây quyết định
with open(r'C:\Users\Thai Tuan Tai\Desktop\vscode\Python\student-graduation-prediction\models\decision_tree_model.pkl', 'wb') as f:
    pickle.dump(decision_tree, f)

# Huấn luyện mô hình rừng ngẫu nhiên
random_forest = RandomForestClassifier(n_estimators=100, random_state=150)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)

# Lưu mô hình rừng ngẫu nhiên
with open(r'C:\Users\Thai Tuan Tai\Desktop\vscode\Python\student-graduation-prediction\models\random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest, f)

# Đánh giá mô hình
def evaluate_model(model_name, y_true, y_pred):
    print(f"\nModel: {model_name}")
    print(f"Độ chính xác: {accuracy_score(y_true, y_pred)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Báo cáo phân loại:")
    print(classification_report(y_true, y_pred))

evaluate_model('Cây quyết định', y_test, y_pred_tree)
evaluate_model('Rừng ngẫu nhiên', y_test, y_pred_forest)

# Vẽ confusion matrix thay thế cho plot_confusion_matrix

# Confusion matrix - Cây quyết định
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix_tree, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Cây quyết định")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('../figures/confusion_matrix_tree.png')
plt.show()

# Confusion matrix - Rừng ngẫu nhiên
conf_matrix_rf = confusion_matrix(y_test, y_pred_forest)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Rừng ngẫu nhiên")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('../figures/confusion_matrix_rf.png')
plt.show()

# Vẽ ROC Curve cho rừng ngẫu nhiên
y_pred_proba_rf = random_forest.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Rừng ngẫu nhiên')
plt.legend(loc="lower right")
plt.savefig('../figures/roc_curve_rf.png')
plt.show()
