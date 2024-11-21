import pandas as pd  
from sklearn.preprocessing import StandardScaler  

# 加载数据  
data = pd.read_csv('Sample_train.csv')  

# 检查缺失值  
print("缺失值统计：")  
print(data.isnull().sum())  

# 处理缺失值（假设没有缺失值）  

# 类别编码  
data = pd.get_dummies(data, drop_first=True)  

# 异常值检测  
print("MonthlyIncome 描述性统计：")  
print(data['MonthlyIncome'].describe())  

# 删除异常值（假设收入超过 20000 的记录为异常）  
data = data[data['MonthlyIncome'] < 20000]  

# 标准化  
scaler = StandardScaler()  
data[['Age', 'DistanceFromHome', 'MonthlyIncome']] = scaler.fit_transform(data[['Age', 'DistanceFromHome', 'MonthlyIncome']])  

# 输出清洗后的数据  
data.to_csv('cleaned_sample_train.csv', index=False)  
print("清洗后的数据已保存为 cleaned_sample_train.csv")