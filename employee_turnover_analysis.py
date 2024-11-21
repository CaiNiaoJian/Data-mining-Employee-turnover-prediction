import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.cluster import KMeans  

# 设置绘图风格  
sns.set(style="whitegrid")  

def load_data(file_path):  
    """加载数据并返回数据框"""  
    data = pd.read_csv(file_path)  
    return data  

def preprocess_data(data):  
    """数据预处理，包括缺失值处理"""  
    print("数据集的基本信息：")  
    print(data.info())  
    
    # 检查缺失值  
    print("缺失值统计：")  
    print(data.isnull().sum())  
    
    # 填充数值型缺失值  
    for column in data.select_dtypes(include=['float64', 'int64']).columns:  
        data[column].fillna(data[column].mean(), inplace=True)  
    
    # 填充非数值型缺失值  
    for column in data.select_dtypes(include=['object']).columns:  
        data[column].fillna(data[column].mode()[0], inplace=True)  # 用众数填充  
    
    return data  

def analyze_correlation(data):  
    """计算并打印特征与离职标签的相关性"""  
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  
    correlation_matrix = numeric_data.corr()  
    
    print("与离职（Label）相关性：")  
    print(correlation_matrix['Label'].sort_values(ascending=False))  
    
    return correlation_matrix  

def perform_clustering(data):  
    """使用 K-means 聚类分析"""  
    features_for_clustering = data.select_dtypes(include=['float64', 'int64']).drop(columns=['Label'])  
    kmeans = KMeans(n_clusters=3)  # 假设选择 3 个聚类  
    data['Cluster'] = kmeans.fit_predict(features_for_clustering)  
    
    print("聚类结果：")  
    print(data[['Cluster', 'Label']].head())  
    
    return data  

def plot_pie_chart(data):  
    """绘制离职与未离职的饼图"""  
    plt.figure(figsize=(8, 6))  
    data['Label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)  
    plt.title('Employee Status Distribution')  
    plt.ylabel('')  
    plt.show()  

def plot_scatter_chart(data):  
    """绘制散点图（例如：年龄与离职状态）"""  
    plt.figure(figsize=(10, 6))  
    sns.scatterplot(data=data, x='Age', y='Label', hue='Cluster', alpha=0.6)  
    plt.title('Age vs. Employee Status by Cluster')  
    plt.xlabel('Age')  
    plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
    plt.show()  

def plot_bubble_chart(data):  
    """绘制气泡图（例如：工龄与离职状态，气泡大小为月收入）"""  
    plt.figure(figsize=(10, 6))  
    sns.scatterplot(data=data, x='TotalWorkingYears', y='Label', size='MonthlyIncome', sizes=(20, 500), alpha=0.5)  
    plt.title('Total Working Years vs. Employee Status (Bubble Size: Monthly Income)')  
    plt.xlabel('Total Working Years')  
    plt.ylabel('Employee Status (0: Not Left, 1: Left)')  
    plt.show()  

def main():  
    # 加载数据  
    file_path = 'Sample_train.csv'  # 请根据实际文件名调整  
    data = load_data(file_path)  
    
    # 数据预处理  
    data = preprocess_data(data)  
    
    # 相关性分析  
    correlation_matrix = analyze_correlation(data)  
    
    # 相似度分析  
    data = perform_clustering(data)  
    
    # 可视化  
    plot_pie_chart(data)  
    plot_scatter_chart(data)  
    plot_bubble_chart(data)  

if __name__ == "__main__":  
    main()