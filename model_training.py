import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report  

# 加载清洗后的数据  
data = pd.read_csv('cleaned_sample_train.csv')  

# 分离特征和标签  
X = data.drop(columns=['Label'])  
y = data['Label']  

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# 训练模型  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# 评估模型  
y_pred = model.predict(X_test)  
print(classification_report(y_test, y_pred))  

# 加载测试数据  
test_data = pd.read_csv('test_noLabel.csv')  

# 进行相同的预处理  
test_data = pd.get_dummies(test_data, drop_first=True)  

# 确保测试数据与训练数据的特征一致  
test_data = test_data.reindex(columns=X.columns, fill_value=0)  

# 进行预测  
test_predictions = model.predict(test_data)  

# 输出结果  
result = pd.DataFrame({'ID': test_data['ID'], 'Label': test_predictions})  
result.to_csv('result.csv', index=False)  
print("预测结果已保存为 result.csv")