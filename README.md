# Employee Turnover Prediction Project
## Overview
Welcome to the Employee Turnover Prediction Project! 🎉  
In this project, we dive deep into the world of employee data to predict who might be packing their bags and heading for the exit. With the power of data analysis and machine learning, we aim to uncover the hidden patterns that lead to employee turnover. 
## 项目概述
欢迎来到员工离职预测项目！🎉  
在这个项目中，我们深入研究员工数据，以预测谁可能会打包行李，准备离开。借助数据分析和机器学习的力量，我们旨在揭示导致员工离职的潜在模式。
## 数据集属性介绍
- **ID**: 员工的唯一标识符。
- **Age**: 员工的年龄。
- **BusinessTravel**: 员工的出差状态（例如：偶尔出差、频繁出差）。
- **Department**: 员工所在的部门（如研发、销售等）。
- **DistanceFromHome**: 员工居住地到工作地点的距离。
- **Education**: 员工的教育程度（通过一个编号来表示）。
- **EducationField**: 员工的教育领域（如生命科学、医疗等）。
- **EmployeeNumber**: 员工的编号。
- **EnvironmentSatisfaction**: 员工对工作环境的满意度（通过一个编号来表示）。
- **Gender**: 员工的性别（男或女）。
- **JobInvolvement**: 员工在工作中的参与度（通过一个编号来表示）。
- **JobLevel**: 员工的职位等级（通过一个编号来表示）。
- **JobRole**: 员工的职位角色（如软件工程师、经理等）。
- **JobSatisfaction**: 员工对工作的满意度（通过一个编号来表示）。
- **MaritalStatus**: 员工的婚姻状况（已婚、单身等）。
- **MonthlyIncome**: 员工的月收入。
- **NumCompaniesWorked**: 员工在职期间工作的公司数量。
- **Over18**: 员工是否超过 18 岁（通常为 Y/N）。
- **OverTime**: 员工是否加班（通常为 Y/N）。
- **PercentSalaryHike**: 员工薪水提升百分比。
- **PerformanceRating**: 员工的绩效评分。
- **RelationshipSatisfaction**: 员工对人际关系的满意度（通过一个编号来表示）。
- **StandardHours**: 标准工作小时数。
- **StockOptionLevel**: 员工的股票期权级别（通过一个编号来表示）。
- **TotalWorkingYears**: 员工的总工作年限。
- **TrainingTimesLastYear**: 员工去年接受培训的次数。
- **WorkLifeBalance**: 员工的工作与生活平衡情况（通过一个编号来表示）。
- **YearsAtCompany**: 员工在公司工作的年限。
- **YearsInCurrentRole**: 员工在当前职位上的年限。
- **YearsSinceLastPromotion**: 距离上一次晋升的年数。
- **YearsWithCurrManager**: 员工与当前经理共事的年数。
- **Label**: 目标标签，通常用于分类（如离职与否）。
## Features
- **Data Preprocessing**: Clean and prepare your data like a pro! 🧹
- **Correlation Analysis**: Discover which factors are most correlated with employee turnover. 🔍
- **Clustering**: Group similar employees together using K-means clustering. 🤝
- **Visualizations**: Create stunning pie charts, scatter plots, and bubble charts to visualize your findings. 📊
## 功能
- **数据预处理**: 像专业人士一样清理和准备您的数据！🧹
- **相关性分析**: 发现哪些因素与员工离职最相关。🔍
- **聚类分析**: 使用 K-means 聚类将相似的员工分组。🤝
- **可视化**: 创建令人惊叹的饼图、散点图和气泡图来可视化您的发现。📊
## Getting Started
To get started with this project, follow these simple steps:
1. **Clone the repository**: 
   ```bash
   git clone https://github.com/CaiNiaoJian/Data-mining-Employee-turnover-prediction.git
   cd employee-turnover-prediction
   ```
2. **Install the required packages**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis**: 
   ```bash
   python employee_turnover_analysis.py
   ```
4. **Enjoy the insights**: Sit back, relax, and watch as the magic unfolds! ✨
## 快速开始
要开始这个项目，请按照以下简单步骤操作：
1. **克隆仓库**: 
   ```bash
   git clone https://github.com/CaiNiaoJian/Data-mining-Employee-turnover-prediction.git
   cd employee-turnover-prediction
   ```
2. **安装所需的包**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **运行分析**: 
   ```bash
   python employee_turnover_analysis.py
   ```
4. **享受洞察**: 坐下来，放松，观看魔法的展开！✨
## Contributing
We welcome contributions! If you have ideas, suggestions, or just want to say hi, feel free to open an issue or submit a pull request. Let's make this project even better together! 🤗
## Employee Turnover Prediction Project
### Overview
Welcome to the Employee Turnover Prediction Project! 🎉  
In this project, we dive deep into the world of employee data to predict who might be packing their bags and heading for the exit. With the power of data analysis and machine learning, we aim to uncover the hidden patterns that lead to employee turnover.
---
### 数据分析与模型预测过程总结
#### 数据处理
1. **清洗数据**  
   - 删除了与预测无关的特征，例如 `EmployeeNumber`、`StandardHours` 和 `Over18`，以减少噪声。
   - 将分类特征（如 `BusinessTravel` 和 `OverTime`）编码为数值。
   - 对标称型特征进行了独热编码（One-Hot Encoding）。
   - 创建了新的特征以增强模型表现，例如 `JobInvolvementPerPercentSalaryHike` 和 `AverageWorkYearsPerCompany`。
   - 对比率型特征进行了Z-分数标准化，对序数型特征进行了最小-最大规范化。
2. **清洗后的数据保存**  
   - 生成了清洗后的数据文件 `cleaned_train.csv` 和 `cleaned_test.csv`。
#### 模型训练
1. **模型选择**  
   - 使用了 `RandomForestClassifier`（随机森林分类器）进行预测。
2. **超参数优化**  
   - 使用 `GridSearchCV` 对超参数进行网格搜索优化。
   - 搜索范围包括 `n_estimators`（树的数量）、`max_depth`（树的最大深度）、`min_samples_split`（内部节点再划分所需的最小样本数）和 `min_samples_leaf`（叶子节点最少样本数）。
3. **模型评估**  
   - 使用训练集的20%作为验证集进行模型评估。
   - 通过准确率和分类报告（包括 `precision`、`recall` 和 `f1-score`）衡量模型性能。
#### 预测与结果比较
1. **预测结果**  
   - 使用清洗后的测试数据集 `cleaned_test.csv` 进行预测，并生成预测文件 `analysis_result.csv`。
   - 输出文件包含员工 `ID` 和预测的离职标签 `Label`。
2. **结果比较**  
   - 将 `result.csv` 和 `analysis_result.csv` 进行对比。
   - 计算相似度（准确率）以及两个数据集中离职比（离职员工占总员工的比例）。
---
### 分析结果
#### 相似度（准确率）
- `Similarity (Accuracy): 0.43`
#### 离职比
- `Result dataset attrition rate: 1.71%`  
- `Analysis result dataset attrition rate: 59.14%`
#### 分类报告
```plaintext
             precision    recall  f1-score   support
         0       0.54      0.50      0.52       117
         1       0.47      0.50      0.49       103
  accuracy                           0.50       220
 macro avg       0.50      0.50      0.50       220
weighted avg    0.51      0.50      0.51       220
```
### 可视化
**混淆矩阵**
用于展示预测值与实际值的差异，直观反映分类模型的表现。
预测可靠性图
通过柱状图展示模型对每个类别预测的可靠性。
#### 总结
- 本项目采用了清洗数据、优化模型、对比预测结果的完整流程。
- 从结果可以看出，模型需要进一步优化以提高准确率。
- 混淆矩阵和可靠性分析为模型改进提供了方向。
- 如果有更多数据或特征可以加入分析，将可能进一步提升模型性能。
## 贡献
我们欢迎贡献！如果您有想法、建议，或者只是想打个招呼，请随时打开问题或提交拉取请求。让我们一起让这个项目变得更好！🤗
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## 许可证
该项目根据 MIT 许可证授权 - 详情请参见 [LICENSE](LICENSE) 文件。
---
Thank you for checking out our project! We hope you find it useful and maybe even a little fun. Happy coding! 🚀
感谢您查看我们的项目！我们希望您觉得它有用，甚至有点有趣。祝您编码愉快！🚀
```