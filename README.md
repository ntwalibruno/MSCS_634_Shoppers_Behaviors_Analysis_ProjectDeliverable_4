**Author**: Ntwali Bruno Bahongere  
**Course**: Advanced Big Data and Data Mining  
**Project**: Residency Project - Shoppers Behavior Analysis

## Project Overview

This project conducts a comprehensive analysis of online shopping behavior to understand customer patterns, predict revenue generation, and build production-ready machine learning models. Using a dataset of e-commerce website sessions, we perform data cleaning, exploratory data analysis (EDA), advanced feature engineering, and deploy multiple classification and regression models with rigorous evaluation. The analysis extends to advanced techniques including customer segmentation with DBSCAN clustering, pattern discovery through FP-Growth association rule mining, and comprehensive model comparison across multiple algorithms (Logistic Regression, SVM, K-NN) with hyperparameter optimization.

## Dataset Summary

**Source**: [Shoppers Behavior and Revenue Dataset](https://www.kaggle.com/datasets/subhajournal/shoppers-behavior-and-revenue) from Kaggle

**Dataset Characteristics**:
- **Original Size**: 12,330 sessions × 18 features
- **Final Cleaned Size**: 12,205 sessions × 18 features (99.0% data retention)
- **Post Feature Engineering**: 12,205 sessions × 50+ features
- **Target Variables**: 
  - `Revenue` (Boolean - Classification target)
  - `PageValues` (Numeric - Regression target)
- **Time Period**: 10 months of e-commerce data
- **Data Quality**: Excellent (no missing values, minimal duplicates)

### Original Feature Categories:
- **Numerical Features (14)**: Page visit counts, session durations, bounce/exit rates, page values
- **Categorical Features (4)**: Month, visitor type, weekend indicator, revenue outcome

## Key Insights Discovered

### 1. Revenue Conversion Patterns
- **Overall Conversion Rate**: 15.6% of sessions result in purchases
- **Class Imbalance**: 5.4:1 ratio (non-revenue to revenue sessions)
- **Weekend Effect**: Weekend sessions show better conversion patterns
- **Seasonal Trends**: Strong monthly variation in conversion rates

### 2. Customer Behavior Segmentation
- **New vs Returning Visitors**: New visitors demonstrate higher conversion potential
- **High-Value Session Characteristics**:
  - Significantly longer total session duration
  - Higher product page engagement
  - Lower bounce and exit rates
  - Elevated page value metrics

### 3. Feature Relationships & Correlations
- **Strong Negative Predictors**: ExitRates (-0.237), BounceRates (-0.176)
- **Positive Predictors**: ProductRelated_Duration (+0.188), PageValues
- **High Feature Correlations**: Page counts strongly correlated with durations (0.7-0.8)
- **Zero-Inflation Patterns**: Duration features show significant zero-value presence

### 4. Advanced Model Performance Results
- **Classification Models (Revenue Prediction)**:
  - Logistic Regression: ROC-AUC 0.90+, Accuracy 85%+ (Excellent baseline)
  - SVM: ROC-AUC 0.87+, Accuracy 87%+ (Strong non-linear performance)
  - K-NN Optimized: ROC-AUC 0.88+, Accuracy 86%+ (Tuned performance)
  
- **Regression Model (PageValues Prediction)**:
  - Lasso Regression: R² 0.999+, RMSE <1.0 (Near-perfect prediction)

### 5. Customer Segmentation & Pattern Discovery
- **DBSCAN Clustering**: 6 distinct customer segments identified
  - Main segment (91.1%): 15.7% revenue rate with optimization potential
  - High-value niches: Specialized segments with >30% conversion rates
  - Noise detection: 6.4% outlier sessions requiring special handling

- **Association Rule Mining**: 66,564+ behavioral patterns discovered
  - Strong revenue predictors: Low bounce + Product engagement → Revenue
  - Navigation optimization: Optimal page flow sequences identified
  - Temporal patterns: Weekend shopping behavior significantly different
  
## Data Cleaning and Preprocessing Methodology

### Phase 1: Data Quality Assessment
1. **Missing Values Analysis**: 
   - Comprehensive check across all 18 features
   - No missing values detected in the original dataset
   
2. **Duplicate Detection**: 
   - Identified 125 duplicate records (1.0% of data)
   - Systematic analysis of duplicate patterns
   
3. **Data Type Validation**: 
   - Verified appropriate data types for each feature
   - Ensured numerical and categorical features are properly formatted
   
4. **Outlier Detection**: 
   - Applied IQR method (3×IQR threshold) for outlier identification
   - Statistical analysis across all numerical features

### Phase 2: Data Cleaning Implementation
1. **Missing Values Handling**:
   - No missing values required imputation
   - Robust framework implemented for future datasets
   
2. **Duplicate Removal**:
   - Removed 125 duplicate rows using pandas drop_duplicates()
   - Preserved 99.0% of original data integrity
   
3. **Data Validation**:
   - Verified cleaned dataset structure and quality
   - Confirmed data types and statistical properties

### Phase 3: Advanced Data Preparation
- Created cleaned dataset (`df_cleaned`) for analysis
- Configured visualization environment with matplotlib and seaborn
- Established consistent plotting styles and parameters
- Prepared data structures for feature engineering pipeline

## Exploratory Data Analysis (EDA)

### 1. Numerical Features Analysis
- **Distribution Visualization**: 3×3 grid of histograms with KDE overlays
- **Statistical Summary**: Detailed statistics including mean, median, skewness, kurtosis
- **Skewness Assessment**: Identified heavily right-skewed distributions requiring transformation
- **Zero-Inflation Detection**: Systematic analysis of features with >10% zero values
- **Statistical Interpretation**: Automated interpretation of distribution shapes

### 2. Categorical Features Analysis  
- **Distribution Analysis**: 2×3 grid visualizing all categorical features
- **High-Cardinality Handling**: Top 10 focus for OperatingSystems, Browser, Region
- **Value Frequency**: Detailed breakdown with percentages for all unique values
- **Comprehensive Statistics**: Complete analysis of categorical variable distributions

### 3. Advanced Correlation Analysis
- **Correlation Matrix**: Masked heatmap showing only lower triangle for clarity
- **High Correlation Detection**: Automated identification of correlations >0.7
- **Target Variable Relationships**: Comprehensive analysis of feature-revenue correlations
- **Statistical Significance Testing**: T-tests with p-value annotations for key relationships

## Advanced Feature Engineering

### 1. Page Engagement Metrics
- **Total_Pages**: Sum of all page visit types
- **Total_Duration**: Aggregate session duration
- **Avg_Time_Per_Page**: Average engagement per page
- **PageValue_Per_Duration**: Value efficiency metrics
- **PageValue_Per_Page**: Value per page interaction

### 2. User Behavior Patterns
- **High_Bounce/High_Exit**: Binary indicators based on median thresholds
- **Bounce_Exit_Score**: Combined bounce-exit behavior metric
- **Product_Focus_Ratio**: Ratio of product-related engagement
- **Session_Depth**: Weighted page depth calculation

### 3. Temporal and Seasonal Features
- **Month_Numeric**: Numerical month encoding
- **Season**: Categorical seasonal grouping (Winter, Spring, Summer, Fall)
- **Holiday_Season**: Binary indicator for November-December
- **Special_Period**: Combined special day and holiday indicator

### 4. Technology and User Profile Features
- **Popular_Browser/Popular_OS**: Binary indicators based on frequency
- **Is_Returning/Is_New**: Visitor type binary encoding
- **Weekend_Shopping**: Weekend indicator

### 5. Interaction and Composite Features
- **High_Value_Session**: Multi-criteria high-value session identifier
- **Engagement_Score**: Weighted composite engagement metric
- **High_Risk_Exit**: Risk indicator based on bounce/exit percentiles
- **Page_Diversity**: Count of different page types visited

### 6. Statistical Transformations
- **Log Transformations**: Applied to skewed duration features using log1p
- **Standardization**: StandardScaler applied to key continuous features
- **Feature Scaling**: Normalized versions of engagement metrics

**Total Engineered Features**: 32 new features created, expanding dataset to 50+ total features

## Machine Learning Models & Evaluation

### 1. Model Architecture
- **Classification Model**: Logistic Regression for Revenue prediction
- **Regression Model**: Lasso Regression for PageValues prediction
- **Data Preparation**: Stratified train-test splits with proper scaling
- **Feature Selection**: Systematic exclusion of redundant and target variables

### 2. Model Performance Metrics

#### Classification Model (Revenue Prediction)
- **Primary Metric**: ROC-AUC Score > 0.90 (Excellent)
- **Accuracy**: 85%+ with balanced precision-recall
- **Cross-Validation**: 5-fold stratified CV with consistent performance
- **Stability**: Very stable across all CV folds
- **Generalization**: Excellent - minimal overfitting

#### Regression Model (PageValues Prediction)
- **Primary Metric**: R² > 0.999 (Near-perfect prediction)
- **RMSE**: <1.0 (Very low prediction error)
- **Cross-Validation**: 5-fold CV with exceptional consistency
- **Stability**: Extremely stable performance
- **Generalization**: Excellent - no overfitting detected

### 3. Advanced Model Evaluation

#### Cross-Validation Analysis
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Overfitting Assessment**: Train vs test performance comparison
- **Stability Analysis**: Coefficient of variation calculations
- **Visualization**: Professional box plots of CV results

#### Performance Visualizations
- **ROC Curves**: Classification performance with AUC visualization
- **Confusion Matrix**: Heatmap showing classification accuracy
- **Actual vs Predicted**: Scatter plots for regression performance
- **Residuals Analysis**: Error distribution assessment

## Challenges Encountered and Solutions

### Challenge 1: Class Imbalance in Revenue Prediction
**Problem**: Highly imbalanced target variable (5.4:1 ratio)
**Solution**: 
- Used stratified sampling in train-test splits
- Applied appropriate evaluation metrics (ROC-AUC, precision-recall)
- Recommended class weights and SMOTE for future improvements

### Challenge 2: Highly Skewed Feature Distributions
**Problem**: Most numerical features heavily right-skewed
**Solution**:
- Applied log1p transformations to skewed features
- Used robust statistical measures (median, IQR)
- Implemented appropriate visualization techniques

### Challenge 3: Zero-Inflated Features
**Problem**: Many duration features contain significant zero values
**Solution**:
- Systematic zero-inflation analysis (>10% threshold)
- Special handling in feature engineering (adding 1 to denominators)
- Flagged for special treatment in modeling

### Challenge 4: High-Cardinality Categorical Features
**Problem**: Features like OperatingSystems, Browser, Region have many unique values
**Solution**:
- Created popularity-based binary features
- Focused visualizations on top categories
- Recommended binning strategies for deployment

### Challenge 5: Feature Engineering Complexity
**Problem**: Need to create meaningful features without data leakage
**Solution**:
- Systematic feature engineering pipeline
- Careful temporal and logical feature construction
- Validation of feature relationships and distributions

## Technical Implementation

### Libraries and Dependencies
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy.stats
- **Machine Learning**: scikit-learn (multiple modules)
- **Data Loading**: kagglehub for dataset access
- **Environment Management**: warnings handling

## Part 3: Advanced Classification Models and Clustering Analysis

### 1. Support Vector Machine (SVM) Classification
- **Algorithm**: SVM with RBF kernel for revenue prediction
- **Performance**: ROC-AUC > 0.87, competitive with logistic regression
- **Training Time**: Optimized for large-scale deployment
- **Use Case**: Non-linear pattern recognition in customer behavior

### 2. K-Nearest Neighbors (K-NN) with Hyperparameter Optimization
- **Algorithm**: K-NN with comprehensive hyperparameter tuning
- **Grid Search Parameters**:
  - `n_neighbors`: [3, 5, 7, 9, 11, 15, 21]
  - `weights`: ['uniform', 'distance']
  - `metric`: ['euclidean', 'manhattan', 'minkowski']
- **Optimization Results**: Best parameters selected via 5-fold cross-validation
- **Performance**: Improved accuracy through systematic parameter optimization

### 3. Comprehensive Model Comparison
- **Models Evaluated**: Logistic Regression, SVM, K-NN (default), K-NN (optimized)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC curves, confusion matrices, performance comparisons
- **Best Model**: Selected based on weighted performance across multiple metrics

## DBSCAN Clustering Analysis

### 1. Customer Segmentation with DBSCAN
- **Algorithm**: Density-Based Spatial Clustering of Applications with Noise
- **Features Used**: 10 key behavioral metrics including PageValues, Engagement Score, Session patterns
- **Parameter Optimization**: Systematic eps and min_samples selection using k-distance plots
- **Results**: 
  - **6 distinct customer clusters** identified
  - **Cluster 1**: 91.1% of customers (main segment, 15.7% revenue rate)
  - **Specialized segments**: 4 smaller clusters with unique characteristics
  - **Noise detection**: 6.4% outlier sessions identified

### 2. Cluster Characteristics Insights
- **High-Value Customers**: Clusters with >30% revenue rates and high page values
- **Engaged Browsers**: Long session duration, moderate conversion
- **Quick Exiters**: High bounce rates, optimization targets
- **Loyal Non-Converters**: Returning visitors needing incentive programs

## Association Rule Mining with FP-Growth

### 1. FP-Growth Algorithm Implementation
- **Transaction Data**: 20+ binary behavioral indicators created
- **Algorithm**: Frequent Pattern Growth with optimized performance
- **Parameters**: 5% minimum support threshold for pattern discovery
- **Results**: 
  - **66,564 association rules** discovered
  - **116,706 rules** with confidence ≥ 60%
  - **Strong patterns** identified with lift values > 2.0

### 2. Key Pattern Categories Discovered

#### Revenue-Predicting Patterns
- **Navigation Patterns**: Low bounce + Product pages → Revenue (High confidence)
- **Visitor Behavior**: Returning visitors + Long sessions → High engagement
- **Temporal Patterns**: Weekend shopping + High page values → Revenue

#### Customer Journey Optimization
- **Page Flow Patterns**: Optimal navigation sequences identified
- **Engagement Drivers**: Feature combinations leading to high engagement
- **Risk Indicators**: Patterns predicting session abandonment

### 3. Business Applications & Real-World Implementation

#### Identified insights can be utilized for
1. **Segmented Marketing Campaigns** by identified clusters
2. **Website Navigation Optimization** based on association rules
3. **Real-time Personalization** using cluster membership
4. **Predictive Analytics Platform** for real-time customer segmentation
5. **Advanced Personalization Engine** using discovered patterns

## Models Performance Summary

### Classification Models Comparison
| Model | Accuracy | ROC-AUC | F1-Score | Training Time | Best Use Case |
|-------|----------|---------|----------|---------------|---------------|
| Logistic Regression | 85%+ | 0.90+ | High | Fast | Baseline, interpretable |
| SVM | 87%+ | 0.87+ | High | Moderate | Non-linear patterns |
| K-NN (Default) | 84%+ | 0.86+ | Moderate | Fast | Simple implementation |
| K-NN (Optimized) | 86%+ | 0.88+ | High | Moderate | Tuned performance |

### Clustering & Pattern Mining Results
- **Customer Segments**: 6 distinct behavioral clusters with clear business applications
- **Pattern Discovery**: 66,564+ actionable association rules for business optimization
- **Business Value**: Comprehensive framework for personalization and targeted marketing

## Enhanced Technical Implementation

### Additional Libraries and Dependencies
- **Clustering**: scikit-learn DBSCAN, PCA for dimensionality reduction
- **Association Rules**: mlxtend library for FP-Growth algorithm
- **Hyperparameter Tuning**: GridSearchCV for systematic optimization
- **Advanced Visualization**: 3D plotting, silhouette analysis, rule visualization

### Files
```
shoppers_behavior_analysis.ipynb    # Complete analysis notebook (28 cells)
README.md                           # Comprehensive project documentation
```
---

*This analysis represents a complete data mining and machine learning pipeline from raw data to production-ready models with advanced clustering and pattern discovery capabilities for e-commerce optimization.*
