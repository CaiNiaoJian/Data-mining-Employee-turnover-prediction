import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

# 加载清洗后的数据  
data = pd.read_csv('cleaned_sample_train.csv')  

# 打印列名  
print("数据集的列名：", data.columns)  

# 设置绘图风格  
sns.set(style="whitegrid")  

# 1. 年龄与离职的关系  
plt.figure(figsize=(10, 6))  
sns.scatterplot(data=data, x='Age', y='Label', alpha=0.6)  
plt.title('Age vs. Employee Status')  
plt.xlabel('Age')  
plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
plt.show()  

# 2. 性别与离职的关系  
plt.figure(figsize=(10, 6))  
sns.countplot(data=data, x='Gender_Male', hue='Label', palette='Set2')  
plt.title('Employee Status by Gender')  
plt.xlabel('Gender (0: Female, 1: Male)')  
plt.ylabel('Count')  
plt.legend(title='Label', labels=['Not Left', 'Left'])  
plt.show()  

# 3. 工作环境满意度与离职的关系  
plt.figure(figsize=(10, 6))  
sns.boxplot(data=data, x='EnvironmentSatisfaction', y='Label', palette='Set2')  
plt.title('Environment Satisfaction vs. Employee Status')  
plt.xlabel('Environment Satisfaction (1-4)')  
plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
plt.show()  

# 4. 员工职位与离职的关系  
plt.figure(figsize=(10, 6))  
sns.countplot(data=data, x='JobRole_Human Resources', hue='Label', palette='Set2')  
plt.title('Employee Status by Job Role')  
plt.xlabel('Job Role (0: Not HR, 1: HR)')  
plt.ylabel('Count')  
plt.legend(title='Label', labels=['Not Left', 'Left'])  
plt.show()  

# 5. 绩效评分与离职的关系  
plt.figure(figsize=(10, 6))  
sns.boxplot(data=data, x='PerformanceRating', y='Label', palette='Set2')  
plt.title('Performance Rating vs. Employee Status')  
plt.xlabel('Performance Rating (1-4)')  
plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
plt.show()  

# 6. 升职年限与离职的关系  
plt.figure(figsize=(10, 6))  
sns.scatterplot(data=data, x='YearsSinceLastPromotion', y='Label', alpha=0.6)  
plt.title('Years Since Last Promotion vs. Employee Status')  
plt.xlabel('Years Since Last Promotion')  
plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
plt.show()  

# 7. 工龄与离职的关系  
plt.figure(figsize=(10, 6))  
sns.boxplot(data=data, x='TotalWorkingYears', y='Label', palette='Set2')  
plt.title('Total Working Years vs. Employee Status')  
plt.xlabel('Total Working Years')  
plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
plt.show()