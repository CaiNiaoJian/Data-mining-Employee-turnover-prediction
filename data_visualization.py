import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

# 加载数据  
data = pd.read_csv('Sample_train.csv')  # 请根据实际文件名调整  

# 打印列名  
print("数据集的列名：", data.columns)  

# 选择数值型列  
numeric_data = data.select_dtypes(include=['float64', 'int64'])  

# 计算相关性矩阵  
correlation_matrix = numeric_data.corr()  

# 生成热力图  
plt.figure(figsize=(12, 10))  
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})  
plt.title('Correlation Heatmap')  
plt.show()  

# 选择与离职（Label）相关性高的特征  
high_correlation_features = correlation_matrix['Label'].abs().sort_values(ascending=False)  
print("与离职相关性高的特征：")  
high_corr_features = high_correlation_features[high_correlation_features > 0.3].index.tolist()  
print(high_corr_features)  

# 绘制高相关性特征与离职状态的散点图  
for feature in high_corr_features:  
    if feature != 'Label':  # 排除离职标签本身  
        plt.figure(figsize=(10, 6))  
        sns.scatterplot(data=data, x=feature, y='Label', alpha=0.6)  
        plt.title(f'{feature} vs. Employee Status')  
        plt.xlabel(feature)  
        plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
        plt.ylim(-0.1, 1.1)  # 设置y轴范围，确保0和1都在可视范围内  
        plt.axhline(0.5, color='red', linestyle='--')  # 添加一条水平线，帮助识别离职状态  
        plt.show()