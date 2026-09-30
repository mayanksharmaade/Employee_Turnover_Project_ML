Employee Turnover Analytics
Overview
Employee Turnover Analytics is a machine-learning classification project developed from a course-end HR analytics problem statement for Portobello Tech.
The project builds an end-to-end workflow for understanding and predicting employee turnover. It combines data-quality checks, exploratory data analysis, employee clustering, class-imbalance handling with SMOTE, cross-validated model comparison, test-set evaluation, and risk-based retention strategy design.
The central business objective is to help an HR team identify patterns associated with employee attrition and use machine-learning probabilities to prioritise retention actions.
The notebook follows the project brief closely and demonstrates both technical modelling and business interpretation.
Business Problem
Employee turnover refers to workers leaving an organisation over time. The source problem statement asks the ML developer supporting HR to analyse employee work characteristics such as project count, average monthly hours, tenure, promotion history, salary level, satisfaction, and performance evaluation, then build models that can help predict which employees are more likely to leave.
The project specifically requires:
1. Data-quality checks for missing values.
2. Exploratory analysis of factors related to turnover.
3. K-means clustering of employees who left.
4. SMOTE to address class imbalance.
5. Five-fold cross-validation.
6. Comparison of Logistic Regression, Random Forest, and Gradient Boosting.
7. Model selection using ROC/AUC and confusion-matrix metrics.
8. Probability-based employee risk zones.
9. Targeted retention strategies.
Dataset
The notebook uses:
HR_comma_sep.csv
with:
14,999 employees
10 columns
The target variable is:
left
where:
0 = employee stayed
1 = employee left
The original sales column is renamed to:
Department
for clarity.
Main features include:
- satisfaction_level
- last_evaluation
- number_project
- average_montly_hours
- time_spend_company
- Work_accident
- promotion_last_5years
- Department
- salary
Project Workflow
HR Employee Dataset
        │
        ▼
Data Quality Checks
        │
        ▼
Exploratory Data Analysis
        │
        ├── Correlation Heatmap
        ├── Satisfaction Distribution
        ├── Evaluation Distribution
        ├── Monthly Hours Distribution
        └── Project Count vs Turnover
        │
        ▼
K-Means Clustering of Leavers
        │
        ▼
Categorical Encoding
        │
        ▼
Stratified 80/20 Train-Test Split
        │
        ▼
SMOTE on Training Data
        │
        ▼
5-Fold Cross-Validation
        │
        ├── Logistic Regression
        ├── Random Forest
        └── Gradient Boosting
        │
        ▼
Final Test-Set Evaluation
        │
        ├── ROC / AUC
        ├── Confusion Matrices
        ├── Precision
        ├── Recall
        └── F1 Score
        │
        ▼
Best Model
        │
        ▼
Turnover Probability
        │
        ▼
Employee Risk Zones
        │
        ▼
Retention Strategy
Data Quality
The dataset contains:
14,999 rows
10 columns
0 missing values
No imputation or row removal is required for missing data.
The target distribution is imbalanced:
Stayed: 11,428 employees
Left:    3,571 employees
which corresponds to approximately:
Stayed: 76.19%
Left:   23.81%
Because the minority class is the group of greatest interest, the project later applies SMOTE only to the training data.
Exploratory Data Analysis
Correlation Analysis
A correlation heatmap is generated for the numerical features.
A key observation is that satisfaction_level has the strongest negative relationship with employee turnover, with a correlation of approximately:
-0.39
This indicates that lower employee satisfaction is strongly associated with leaving.
Other variables such as tenure, project count, and average monthly hours show weaker relationships when considered individually.
The analysis also highlights relationships between project load, monthly working hours, and employee evaluation.
Distribution Analysis
The project examines:
- Employee satisfaction
- Last evaluation score
- Average monthly working hours
Satisfaction
The satisfaction distribution is multi-modal.
A noticeable group of employees has very low satisfaction, while another large group has relatively high satisfaction.
The low-satisfaction cluster represents an important turnover-risk population.
Last Evaluation
Evaluation scores show a bi-modal pattern, with groups around moderate and high performance.
This suggests that turnover is not limited to poor performers; some highly rated employees also leave.
Average Monthly Hours
Working hours also show multiple behavioural groups.
Employees at both low and high workload levels appear among leavers, suggesting that both disengagement and burnout may contribute to turnover.
Project Count and Employee Turnover
The notebook compares project count for employees who stayed and employees who left.
The observed pattern is approximately U-shaped:
- Employees with 2 projects show substantial turnover.
- Employees with 3–4 projects have the strongest retention.
- Turnover rises again among employees with 6 projects.
- In this dataset, employees with 7 projects all belong to the leaver group.
This suggests that both under-utilisation and excessive workload may contribute to attrition.
K-Means Clustering
The project isolates employees who left and performs K-means clustering using:
satisfaction_level
last_evaluation
with:
n_clusters = 3
The resulting cluster centres are approximately:
Cluster Profile	Satisfaction	Evaluation
High performance / low satisfaction	0.111	0.869
Lower performance / medium satisfaction	0.410	0.517
High performance / high satisfaction	0.809	0.912


Cluster sizes are approximately:
1,650
977
944
The notebook interprets these groups as:
High-Performance / Low-Satisfaction
Strong performers with very low satisfaction.
This group may represent burnout or excessive workload and is particularly costly to lose.
Lower-Performance / Medium-Satisfaction
Employees with moderate satisfaction but weaker evaluations.
This group may reflect disengagement or performance-management issues.
High-Performance / High-Satisfaction
Employees who appear both satisfied and highly rated but still leave.
These employees may be responding to external opportunities, career progression, or better compensation.
The clustering step is important because it shows that not all leavers have the same profile and therefore should not receive the same retention intervention.
Data Pre-Processing
Categorical variables are separated and converted to numerical features using:
pd.get_dummies()
The final feature matrix contains:
14,999 rows
20 predictors
The dataset is then split using a stratified:
80% training
20% testing
random_state = 123
Resulting sizes:
Training: 11,999 rows
Testing:   3,000 rows
Before SMOTE, the training target contains:
Stayed: 9,142
Left:   2,857
Handling Class Imbalance with SMOTE
SMOTE is applied only to the training data.
After oversampling:
Stayed: 9,142
Left:   9,142
This balances the training set while leaving the test set untouched.
Keeping the test set in its original imbalanced distribution is important because final evaluation should reflect the real data rather than an artificially balanced test population.
Model Training
The project trains three classifiers:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
Five-fold cross-validation is applied using:
StratifiedKFold
Cross-Validation Results
Logistic Regression
Approximate cross-validation results:
Accuracy: 0.808
Precision (left): 0.801
Recall (left):    0.820
F1 (left):        0.811
Random Forest
Approximate cross-validation results:
Accuracy: 0.985
Precision (left): 0.994
Recall (left):    0.975
F1 (left):        0.985
Gradient Boosting
Approximate cross-validation results:
Accuracy: 0.963
Precision (left): 0.976
Recall (left):    0.948
F1 (left):        0.962
Final Test-Set Evaluation
The models are trained on the full SMOTE-balanced training set and then evaluated against the original untouched test set.
Logistic Regression
Accuracy:  0.784
Precision for left: 0.534
Recall for left:    0.723
F1 for left:        0.614
Random Forest
Accuracy:  0.990
Precision for left: 0.979
Recall for left:    0.979
F1 for left:        0.979
Gradient Boosting
Accuracy:  0.964
Precision for left: 0.920
Recall for left:    0.929
F1 for left:        0.924
ROC-AUC Comparison
The test-set ROC-AUC scores are:
Random Forest:       0.9952
Gradient Boosting:   0.9856
Logistic Regression: 0.8209
Random Forest achieves the strongest overall discrimination between employees who stay and employees who leave.
Best Model
The notebook selects:
Random Forest Classifier
as the best-performing model.
It achieves the strongest combination of:
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 score
for the left class.
For this business problem, recall is especially important because a false negative means predicting that an at-risk employee will stay when that person actually leaves.
Missing a likely leaver may prevent HR from intervening before departure.
Precision remains important as well because unnecessary retention interventions consume time and resources, but the project prioritises recall while maintaining strong precision.
Employee Turnover Risk Zones
The Random Forest model predicts employee turnover probabilities for the test set.
Employees are grouped into four risk zones.
Safe Zone          < 20%
Low-Risk Zone      20%–60%
Medium-Risk Zone   60%–90%
High-Risk Zone     > 90%
The test-set distribution is:
Safe Zone:         2,185 employees
Low-Risk Zone:       109 employees
Medium-Risk Zone:     56 employees
High-Risk Zone:      650 employees
Retention Strategy
Safe Zone — Green
Employees with turnover probability below 20%.
Recommended actions:
- Maintain normal engagement
- Provide recognition
- Continue career-development discussions
- Avoid unnecessary intervention
Low-Risk Zone — Yellow
Employees between 20% and 60%.
Recommended actions:
- Periodic manager check-ins
- Monitor project workload
- Maintain a healthy project allocation
- Provide learning and development opportunities
- Watch for declining satisfaction
Medium-Risk Zone — Orange
Employees between 60% and 90%.
Recommended actions:
- Proactive one-on-one discussions
- Review compensation
- Review promotion eligibility
- Rebalance workload
- Provide concrete career-growth opportunities
High-Risk Zone — Red
Employees above 90%.
Recommended actions:
- Immediate personalised retention discussions
- Leadership involvement
- Role-change opportunities
- Compensation review
- Burnout reduction
- Workload adjustment
- Succession planning if departure appears unavoidable
Key Business Insights
The notebook identifies several notable patterns:
- Low satisfaction is strongly associated with turnover.
- Employees can leave because of both under-utilisation and overwork.
- A project allocation around 3–4 projects appears to be a retention sweet spot in this dataset.
- Some high performers have extremely low satisfaction, suggesting burnout risk.
- Some highly satisfied and highly evaluated employees still leave, indicating that external opportunities may matter.
- Promotion history and salary should be considered when designing targeted retention strategies.
- Employee attrition should not be treated as a single homogeneous behaviour.
Visual Outputs
The notebook saves charts under:
graphs/
Generated visualisations include:
01_correlation_heatmap.jpg
02_dist_satisfaction_level.jpg
02_dist_last_evaluation.jpg
02_dist_average_montly_hours.jpg
03_project_count_bar.jpg
04_kmeans_clusters.jpg
05_report_logreg.jpg
06_report_rf.jpg
07_report_gb.jpg
08_roc_curves.jpg
09_confusion_matrices.jpg
10_risk_zones.jpg
These charts document the full analytical and modelling workflow.
Technology Stack
- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- K-Means
- SMOTE
- Logistic Regression
- Random Forest
- Gradient Boosting
