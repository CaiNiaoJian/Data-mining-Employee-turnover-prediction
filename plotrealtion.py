import pandas as pd  

# 加载数据  
data = pd.read_csv('Sample_train.csv')  # 请根据实际文件名调整  

# 打印列名  
print("数据集的列名：", data.columns)  

# 选择除 Label 之外的其他属性  
features = data.columns[data.columns != 'Label']  
print("除 Label 之外的其他属性：")  
print(features.tolist())