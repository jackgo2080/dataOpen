from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据清洗：这里我们假设数据集已经比较干净，不需要额外处理

# 缺失值处理
imputer = SimpleImputer(strategy='mean')  # 用均值填充缺失值
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 特征选择：这里我们暂时使用所有特征，不进行特征选择

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train, X_test)

#------------------------选择模型---------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 决策树模型
dt = DecisionTreeClassifier(random_state=42)

# 随机森林模型
rf = RandomForestClassifier(random_state=42)

# 支持向量机模型
svm = SVC(random_state=42)

#------------------------训练模型---------------------------
from sklearn.model_selection import GridSearchCV

# 决策树参数调优
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)

# 选择最佳参数的决策树模型
best_dt = grid_search_dt.best_estimator_

#--------------------评估模型------------------------
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# 使用测试集评估模型
y_pred_dt = best_dt.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred_dt)
recall = recall_score(y_test, y_pred_dt, average='macro')
f1 = f1_score(y_test, y_pred_dt, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 打印分类报告
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))
