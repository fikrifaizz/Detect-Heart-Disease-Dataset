# Implementation of Dollmaker Optimization Algorithm in Hyperparameter Tuning Support Vector Machine for Liver Disease Detection

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
    
    $$x{^{P1}_{i,d}} = x_{i,d} + r \cdot (P_d - I \cdot x_{i,d})$$
    
    $$X_i = \begin{cases} X_i^{P_1}, & F_i^{P_1} \leq F_i, \\ X_i, & \text{else} \end{cases}$$
    
    where:

    - $P$ = The selected doll-making pattern,

    - $P_d$ = The selected doll-making pattern on dimension $d$,

    - $X{^{P1}_i}$ = The New position for the $i$ individuals based on first phase of DOA,

    - $F{^{P1}_i}$ = objective function value,

    - $I$ = randomly selected number, taking values of 1 or 2.
   
    1.3 Exploitation:
    Exploitation is done to improve the best candidate solutions through local search around the optimal solution. This ensures the reinforcement of the best solution without losing focus on the best areas in the search space.
    
    $x{^{P2}_{i,d}} = x_{i,d} + (1 - 2r_{i,d}) \cdot (\frac{ub_j - lb_j}{t})$

    $X_i = \begin{cases} X_i^{P_2}, & F_i^{P_2} \leq F_i, \\ X_i, & \text{else} \end{cases}$
    
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

## Data Understanding



## Reference
<p style="text-align: justify;">

[1] World Health Organization. (2021). "Cardiovascular Diseases (CVDs)." WHO. https://www.who.int

[2] Yousri, D., Mohammed, A., & Ismail, H. (2021). "Metaheuristics for parameter tuning in machine learning models." Applied Soft Computing. https://doi.org/10.1016/j.asoc.2021.107581

[3] Saleh Al Omari, et al. "Dollmaker Optimization Algorithm: A Novel Human-Inspired Optimizer for Solving Optimization Problems." *International Journal of Intelligent Engineering and Systems*, Vol.17, No.3, 2024, pp. 816–823. DOI: 10.22266/ijies2024.0630.63.

[4] Scikit-learn documentation. "Support Vector Machines." Available at: https://scikit-learn.org/dev/modules/svm.html (accessed on January 25, 2025).

</p>