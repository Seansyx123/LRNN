# 导入所需的库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # 以鸢尾花数据集为例
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 类别标签

# 数据预处理：标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 创建并训练LDA分类器
lda = LDA(n_components=None)  # 若不指定n_components，则默认为类别数-1
lda.fit(X_train, y_train)

# 使用LDA模型对测试集进行预测
y_pred = lda.predict(X_test)

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"LDA模型的预测准确率为: {accuracy}")

# 如果需要降维可视化（对于二维或三维数据）
# 可以使用transform方法将高维数据投影到LDA空间
X_lda = lda.transform(X_scaled)