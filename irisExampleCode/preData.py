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
