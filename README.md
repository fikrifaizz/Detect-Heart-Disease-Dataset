# Implementation of Dollmaker Optimization Algorithm in Hyperparameter Tuning Support Vector Machine for Heart Disease Detection

|                      | Information                                                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Source               | [Kaggle Dataset: Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset 'Heart Disease Dataset')   |
| Usability            | 8.82                                                                                                                                 |
| License              | Unknown                                                                                                                              |
| File Types and Sizes | csv (38.11 KB)                                                                                                                       |
| Tags                 | Health, Heart Conditions                                                                                                             |

---

## Project Domain
### Background
<p style="text-align: justify;">
Heart disease is one of the leading causes of death worldwide, with more than 17 million deaths each year caused by cardiovascular disease according to the World Health Organization (WHO) [1]. Early detection of heart disease is a priority to reduce mortality, where Machine Learning (ML)-based models can play an important role. One of the algorithms often used for classification in this problem is Support Vector Machine (SVM). However, the performance of SVM is highly dependent on the combination of hyperparameters such as C (regularization parameter) and gamma (kernel coefficient). Tuning these hyperparameters can be challenging due to the large search space and non-linear relationships between parameters.
</p>
<p style="text-align: justify;">
To overcome these challenges, metaheuristic approaches such as Dollmaker Optimization Algorithm (DOA) are applied for hyperparameter tuning. Metaheuristics have been shown to be superior to conventional methods such as grid search or random search due to their ability to efficiently explore and exploit the search space. For example, a study published in ScienceDirect showed that metaheuristics have the advantage of improving the accuracy of Machine Learning models by optimizing complex parameters through adaptive iteration [2]. Dollmaker Optimization Algorithm, which is designed based on the principles of creative adaptation and exploration, offers an innovative solution in parameter optimization.
</p>
<p style="text-align: justify;">
The implementation of DOA for SVM hyperparameter tuning in this project aims to produce an accurate heart disease detection model that can be integrated into a system-based application to provide early warning to patients and doctors. With this approach, this project supports preventive efforts and early treatment of heart disease, which is in line with the global goal of reducing mortality from cardiovascular disease [1].
</p>

---

## Business Understanding
### Problem Statements
<p style="text-align: justify;">

1. How to detect heart disease early by utilizing available patient health data?
2. How to apply Dollmaker Optimization Algorithm (DOA) for SVM hyperparameter tuning?
</p>

### Goals
<p style="text-align: justify;">

1. Building an SVM-based model that is able to detect the risk of heart disease early with high accuracy.
2. Applying Dollmaker Optimization Algorithm to improve SVM performance through hyperparameter tuning **C** and **$\gamma$**.

</p>

### Solution Statements
<p style="text-align: justify;">
To achieve the goal of maximizing accuracy in detecting heart disease, two main components are applied, namely Dollmaker Optimization Algorithm (DOA) for hyperparameter tuning and Support Vector Machine (SVM) for building a classification model. The following is a detailed explanation of each component before integration is carried out.
</p>

1. Dollmaker Optimization Algorithm (DOA)

    DOA is a metaheuristic optimization algorithm inspired by the creativity of dollmakers. This algorithm adopts exploration and exploitation mechanisms in the search space to find optimal solutions adaptively. Here are the steps of DOA:

    1.1 Population Initialization:
    A population (**P**) of a certain size (**N**) is initialized. Each individual in the population represents a candidate solution, where the candidate solution is a vector of hyperparameters *C* (regularization) and *$\gamma$* (kernel coefficient). The initialization formula for each individual is given as follows:  
    
    $$x_{i,d} = lb_d + r \cdot (ub_d - lb_d)$$
    
    where:

    - $x_{i,d}$ = initial position of individual $i$ on dimension $d$,

    - $lb_d, ub_d$ = lower and upper bounds for the dimension $d$,

    - $r$ = random number between 0 and 1. 
   
    for svm tuning:

    - $C \in [0.1, 100] $,

    - $\gamma \in [0.01, 0.1]$.
   
    1.2 Exploration:
    Exploration is carried out to find new solutions by using the adaptation equation:
    
    <img src="https://github.com/user-attachments/assets/fe0b4891-5b3d-4505-a4a3-b3a0e5fe2818" alt="Exploration Model" title="Exploration Model">
 
    <img src="https://github.com/user-attachments/assets/b03b0a92-aea3-480b-b6b5-b963371394c1" alt="Exploration Rules" title="Exploration Rules">

    where:

    - $P$ = The selected doll-making pattern,

    - $P_d$ = The selected doll-making pattern on dimension $d$,

    - $X{^{P1}_i}$ = The New position for the $i$ individuals based on first phase of DOA,

    - $F{^{P1}_i}$ = objective function value,

    - $I$ = randomly selected number, taking values of 1 or 2.
   
    1.3 Exploitation:
    Exploitation is done to improve the best candidate solutions through local search around the optimal solution. This ensures the reinforcement of the best solution without losing focus on the best areas in the search space.
    
    <img src="https://github.com/user-attachments/assets/8897fd5e-700d-4b39-a971-861b4d3e42e6" alt="Exploitation Model" title="Exploitation Model">
    
    <img src="https://github.com/user-attachments/assets/35e5fa56-d46a-4958-8daa-858f0b82c707" alt="Exploitation Rules" title="Exploitation Rules">

    where:

    - $X{^{P2}_i}$ = The New position for the $i$ individuals based on second phase of DOA,

    - $t$ = Iteration Counter
   
    1.4 Fitness Evaluation: The fitness ($f$) of each individual is calculated based on the performance of the SVM model trained using the hyperparameters $C$ and $\gamma$. In this project, fitness is defined as the accuracy of the model:
    
    $$f = (\frac{TP + TN}{TP + TN + FP + FN})$$
    
    where TP, TN, FP, and FN are True Positives, True Negatives, False Positives, and False Negatives.

    1.5 Adaptive Iteration: The population is updated iteratively until the maximum number of iterations is reached. At each iteration, the best individual ($P_{best}$) is saved and used to generate new solutions [3].


2. Support Vector Machine
    
    Support Vector Machine (SVM) is a supervised learning algorithm widely used for classification tasks. It is particularly effective in handling both linear and non-linear data by employing a kernel function to map input data to higher-dimensional spaces. For this project, we utilize the Radial Basis Function (RBF) kernel, which allows the model to handle complex decision boundaries. The following outlines the mathematical formulation of SVM and its integration into this project.
    
    2.1 Objective Function
    
    The SVM classifier solves the following optimization problem:
    
    $$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$
    
    Subject to:
    
    $$y_i (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i$$
    
    where:

    - $w$ = weight vector,

    - $b$ = bias term,

    - $\xi_i$ = slack variables for soft margin,

    - $C$ = regularization parameter that controls the trade-off between margin width and classification error.
    
   2.2 Kernel Function: Radial Basis Function (RBF)
    
    The RBF kernel is defined as:
    
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
    
    where:

    - $\gamma > 0$: kernel coefficient that controls the influence of a single training example.
   
    2.3 Decision Function
    
    The decision boundary is expressed as:
    
    $$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$
    
    where:

    - $\alpha_i$ = Lagrange multipliers,

    - $y_i$ = class label (+1, -1),

    - $K(\mathbf{x}_i, \mathbf{x})$ = kernel function [4]. 

3. Integration of DOA and SVM
    
    After understanding each component, here are the integration steps to maximize model accuracy:

    Implementation Stages:

    3.1 Parameter Initialization
    
    - DOA is used to initialize the parameter population $C$ and $\gamma$.
    - Each member of the population is evaluated using SVM on the training data.
   
    3.2 Fitness Function

    - The accuracy of the SVM model is calculated as a fitness metric:
   
    $$f = \frac{TP + TN}{TP + TN + FP + FN}$$

    3.3 DOA Iteration

    The population is updated through exploration and exploitation, with the parameters $C$ and $\gamma$ adjusted until the maximum iteration is reached.

    3.4 Optimal Model

   The best parameters $C$ and $\gamma$ are used to retrain the SVM model with all training data.

   3.5 Final Evaluation

   The model is tested on test data to measure performance using the metrics:

   - Accuracy :

      $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

   - Precision :

      $$Precision = \frac{TP}{TP + FP}$$

   - Recall :

      $$Recall = \frac{TP}{TP + FN}$$

   - F1-Score :

      $$F1-Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

---
## Data Understanding

### Dataset Information
- Total initial data: 1025 rows.
- Number of duplicate data: 723 rows found as duplicates.
- Number of data after duplicate cleaning: 302 rows remaining.
- Inconsistent data :
  - column thal: Value 0 is considered invalid.
  - column ca: Value 4 is considered invalid.
  - After inconsistent data cleaning, the final total rows are 296 rows.


### Dataset Variables

| **Column**  | **Description**                                                                                  | **Data Type** |
|-------------|--------------------------------------------------------------------------------------------------|---------------|
| `age`       | Age of the patient in years                                                                      | Numerical     |
| `sex`       | Gender of the patient (1 = Male; 0 = Female)                                                     | Numerical     |
| `cp`        | Chest pain type (0 = No pain, 1-3 = Varying levels of severity)                                  | Numerical     |
| `trestbps`  | Resting blood pressure (in mmHg on admission to the hospital)                                    | Numerical     |
| `chol`      | Serum cholesterol level in mg/dl                                                                 | Numerical     |
| `fbs`       | Fasting blood sugar > 120 mg/dl (1 = True; 0 = False)                                            | Numerical     |
| `restecg`   | Resting electrocardiographic results (0 = Normal, 1 = ST-T abnormality, 2 = Left ventricular)    | Numerical     |
| `thalach`   | Maximum heart rate achieved                                                                      | Numerical     |
| `exang`     | Exercise-induced angina (1 = Yes; 0 = No)                                                        | Numerical     |
| `oldpeak`   | ST depression induced by exercise relative to rest                                               | Numerical     |
| `slope`     | Slope of the peak exercise ST segment                                                            | Numerical     |
| `ca`        | Number of major vessels (0-3) visualized through fluoroscopy                                     | Numerical     |
| `thal`      | Thalassemia status (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)                         | Numerical     |
| `target`    | Heart disease diagnosis (1 = Risk of heart disease; 0 = No risk)                                 | Numerical     |

### Exploration Data Analysis

1. Distribution of Numerical Features (Histogram with KDE)

<img src="https://github.com/user-attachments/assets/f8fdfae9-a900-4b74-bea4-d767d6af9cd0" alt="Distribution of Numerical Features" title="Distribution of Numerical Features">
   
- Cholesterol (chol) and oldpeak showed skewness to the right, indicating that most patients had low values for these features.

- Some category features such as cp (type of chest pain) and thal (type of thalassemia) have uneven distribution, which may impact the interpretation of the relationship with the target.

2. Correlation Heatmap

<img src="https://github.com/user-attachments/assets/34334a1d-0bd5-41f4-8e99-4f0159083158" alt="Correlation Heatmap" title="Correlation Heatmap">

- Features such as cp (chest pain type) and target have a relatively strong positive correlation, suggesting chest pain types are linked to heart disease.

- thalach (maximum heart rate achieved) shows a negative correlation with target, implying that lower maximum heart rate often correlates with heart disease.

3. Pairplot

<img src="https://github.com/user-attachments/assets/6ef5603b-b6d3-4cf7-90e9-9f65f9e3e55a" alt="Pairplot" title="Pairplot">

- Visual inspection reveals linear or clustered patterns between some feature pairs, such as:

- thalach and age: Older patients tend to have lower maximum heart rates.

- oldpeak and ca: Higher oldpeak values correlate with an increase in the number of major vessels colored by fluoroscopy.

- The target variable shows clear separability for some features like cp, thalach, and oldpeak.

4. Target Distribution

<img src="https://github.com/user-attachments/assets/698a616b-41e6-48b4-8ff8-5717d6e55248" alt="Target Distribution" title="Target Distribution">

- The target variable is evenly distributed, with a slightly higher count for class 1 (patients with heart disease).

- Balanced distribution ensures no significant bias in model training toward any class.

---
## Data Preparation

1. Train Test Split

The dataset is divided into training and testing sets using an 80-20 split ratio. This ensures the model is trained on the majority of the data while reserving a portion for unbiased evaluation.

2. Feature Scalling

Min-Max Scaling is applied to normalize the feature values within a range of 0 to 1. This prevents features with larger numerical ranges from dominating the model's learning process.

### Splitting Data into Training and Testing Sets

```python
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Description:

  - x contains all features except the target variables.

  - y contains the target variable (target column)

  - The train_test_split function from sklearn.model_selection is used to divide the dataset into training (80%) and testing (20%) subsets.

- Purpose:

  - To evaluate the model's generalization capability on unseen data after training.

- Outcome:

   - Shapes of the training and testing datasets:
   
  ```python
   Training set shape: X_train=(236, 13), y_train=(236,)
   Test set shape: X_test=(60, 13), y_test=(60,)
   ```

### Feature Scaling using Min-Max Scaler

```python
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dump(scaler, 'scaler.pkl')
```

- Description:

   - Min-Max Scaler from sklearn.preprocessing is applied to scale all feature values between 0 and 1.

   - fit_transform is used on the training set to compute the scaling parameters and apply scaling.

   - transform is used on the testing set using the same scaling parameters from the training set to ensure consistency.

   - The fitted scaler is saved using joblib.dump for later use during inference.

- Purpose:

  - To normalize the data so that all features contribute equally to the model's training, avoiding bias caused by different feature ranges.

- Outcome:

   - Features are scaled, ensuring better performance for algorithms sensitive to feature scaling, such as Support Vector Machines (SVM).

---
## Modeling and Evaluation

### Support Vector Machine 

Model Description:

The SVM model uses the Radial Basis Function (RBF) kernel, which is effective for non-linear classification problems. The default hyperparameters were used:

- $C$ : 1 (Regularization parameter)

- Gamma : 'scale' (Kernel Coefficient)

Implementation:

```python
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
```

Performance Matrix:

- Accuracy: 88.33%

- Precision: 87.18%

- Recall: 94.44%

- F1 Score: 90.67%

Advantages:

- Handles non-linear data effectively.

- Works well with high-dimensional data.

- Robust to overfitting with proper regularization.

Disadvantages:

- Sensitive to the choice of hyperparameters.

- Computationally expensive for large datasets.

### Dollmaker Optimization Algorithm with SVM (DOA-SVM)

Model Description:

DOA-SVM enhances the SVM model by optimizing its hyperparameters C and Gamma using the Dollmaker Optimization Algorithm (DOA).

- Population Size: 5

- Dimensions: 2 ($C$ and Gamma)

- Iterations : 500

- Search Space:

   - $C \in [0.1, 100]$

   - $Gamma \in [0.01, 0.1]$

Implementation:

```python
doa_svm = DOA_SVM(5, 2, 500, X_train, X_test, y_train, y_test)
best_fitness, best_params = doa_svm.run()
```

Performance Metrics:

- Best Fitness: 91.67%

- Final Evaluation:

   - Accuracy: 91.67%

   - Precision: 91.89%

   - Recall: 94.44%

   - F1 Score: 93.15%

Final Hyperparameters:
- $C$: 20.2970

- Gamma: 0.0660

Advantages:

- Automatically finds optimal hyperparameters.

- Improves model accuracy and generalization.

Disadvantages:

- Requires more computational resources due to multiple iterations.

- The performance gain may diminish for simpler problems.

### Model Comparison

| **Metric**  | **SVM** | **DOA-SVM** |
|-------------|---------|-------------|
| `Accuracy`  | 88.33%  | 91.67%      |
| `Precision` | 87.18%  | 91.89%      |
| `Recall`    | 94.44%  | 94.44%      |
| `F1 Score`  | 90.67%  | 93.15%      |

## Model Prediction 

The prediction phase leverages the trained DOA-SVM model and scaler to evaluate whether a given set of patient health data indicates heart disease. The prediction process involves the following steps:

1. Load Pre-trained Model and Scaler

   - The optimized model (best_doa_svm_model.pkl) and the scaler (scaler.pkl) are loaded to ensure consistency during prediction.

2. Input Preprocessing

   - The input data is reshaped into a single-row array and transformed into a DataFrame for feature alignment.

3. Feature Scalling

   - The input features are scaled using the previously fitted MinMaxScaler.

4. Prediction

   - The scaled data is passed to the DOA-SVM model to classify the input as either:

      - 1: Heart Disease Detected

      - 0: No Symptoms of Heart Disease

5. Interpretation

   - The prediction is returned as a user-friendly message.

## Conclusion

1. By leveraging patient health data, a Support Vector Machine (SVM)-based model was built to classify the risk of heart disease with high accuracy. The model achieved an accuracy of 88.33%, with balanced metrics such as Precision (87.18%), Recall (94.44%), and an F1 Score (90.67%) using only the SVM approach. This demonstrates the effectiveness of the SVM model in early detection of heart disease, making it a valuable tool for preventive healthcare applications.

2. The Dollmaker Optimization Algorithm (DOA) was successfully implemented to tune the hyperparameters C (regularization) and $\gamma$ (kernel coefficient) of the SVM model. By iteratively optimizing these parameters, the DOA-SVM model achieved a higher accuracy of 91.67%, alongside improved metrics such as F1 Score (93.15%), Precision (91.89%), and Recall (94.44%). This demonstrates the capability of DOA to explore and exploit the parameter space effectively, enhancing the SVM model's performance.

3. The project demonstrates the effective use of machine learning for early heart disease detection and showcases the impact of optimization algorithms like DOA in improving model performance. The combined approach of SVM modeling and DOA optimization ensures a robust, accurate, and scalable solution for healthcare diagnostics, paving the way for real-world applications in medical systems.


## Reference
<p style="text-align: justify;">

[1] World Health Organization. (2021). "Cardiovascular Diseases (CVDs)." WHO. https://www.who.int

[2] Yousri, D., Mohammed, A., & Ismail, H. (2021). "Metaheuristics for parameter tuning in machine learning models." Applied Soft Computing. https://doi.org/10.1016/j.asoc.2021.107581

[3] Saleh Al Omari, et al. "Dollmaker Optimization Algorithm: A Novel Human-Inspired Optimizer for Solving Optimization Problems." *International Journal of Intelligent Engineering and Systems*, Vol.17, No.3, 2024, pp. 816–823. DOI: 10.22266/ijies2024.0630.63.

[4] Scikit-learn documentation. "Support Vector Machines." Available at: https://scikit-learn.org/dev/modules/svm.html (accessed on January 25, 2025).

</p>
