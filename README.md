# Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/dandrandandran2093/competitions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🎯 Competition Overview

Kaggle's legendary **Titanic: Machine Learning from Disaster** competition - predicting passenger survival using machine learning. This project combines comprehensive exploratory data analysis (EDA) with advanced ensemble techniques for maximum performance.

**Kaggle Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

**My Notebook:** [View on Kaggle](https://www.kaggle.com/code/dandrandandran2093/the-titanic-project)

## 📊 Project Highlights

- **Advanced Feature Engineering**: 18+ engineered features including family size, titles, age/fare groups
- **Ensemble Learning**: Voting classifier combining Random Forest, XGBoost, and Gradient Boosting
- **Strong Regularization**: Optimized hyperparameters to prevent overfitting
- **Interactive Visualizations**: Plotly-powered EDA and model performance analysis
- **10-Fold Cross-Validation**: Reliable performance estimation (83%+ accuracy)

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **Cross-Validation (10-Fold)** | **0.8340 ± 0.0324** |
| Validation Accuracy | 0.8324 |
| Training Accuracy | 0.8540 |
| Generalization Gap | 0.0216 ✅ |
| ROC-AUC Score | 0.89+ |

**Key Achievement:** Minimal overfitting through aggressive regularization and careful feature engineering.

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

**Dataset Statistics:**
- Training: 891 passengers
- Test: 418 passengers  
- Missing data: Age (177), Cabin (687), Embarked (2)

**Key Insights:**
- Female survival rate: **74%** vs Male: **19%**
- 1st class survival: **63%** vs 3rd class: **24%**
- Children under 12: Higher survival rate
- Family size impact: Sweet spot at 2-4 members

### 2. Feature Engineering

Created 18+ features from 11 original columns:

**Engineered Features:**
- `FamilySize`: SibSp + Parch + 1
- `IsAlone`: Binary indicator for solo travelers
- `Title`: Extracted from Name (Mr, Miss, Mrs, Master, Rare)
- `AgeGroup`: Binned ages (Child, Teen, Adult, Middle-aged, Senior)
- `FareGroup`: Quartile-based fare categories
- `HasCabin`: Cabin availability indicator

**Transformations:**
- Log-transformed Fare (handles outliers)
- Age imputation by Title median
- Clipped extreme values (Age: 0.5-80)

### 3. Model Selection & Ensemble

**Base Models:**

| Model | CV Score | Strategy |
|-------|----------|----------|
| Random Forest | 0.8285 | Robust baseline, feature importance |
| XGBoost | 0.8327 | Gradient boosting, regularization |
| Gradient Boosting | 0.8272 | Sequential ensemble |
| Logistic Regression | 0.8104 | Linear benchmark |

**Final Ensemble:**
- **Voting Classifier** (Soft voting)
- Combines: RF + XGBoost + GB
- Weights: Equal (best performance)

### 4. Regularization Strategy

Applied aggressive regularization to prevent overfitting:

```python
Random Forest:
- max_depth=5 (reduced from 10)
- min_samples_split=15 (increased from 10)
- min_samples_leaf=5 (increased from 4)

XGBoost:
- learning_rate=0.03 (reduced from 0.1)
- max_depth=3 (reduced from 6)
- reg_alpha=0.1, reg_lambda=1.0 (L1/L2 regularization)

Gradient Boosting:
- learning_rate=0.03
- subsample=0.8 (additional regularization)
```

## 📈 Visualizations

### Survival Analysis
- Gender distribution by survival
- Class-based survival rates
- Age distribution comparison
- Family size impact

### Model Evaluation
- Cross-validation comparison
- Confusion matrix heatmap
- ROC curve (AUC=0.89+)
- Feature importance ranking

**Top 5 Most Important Features:**
1. Sex (Gender)
2. Fare
3. Age
4. Title_Mr / Title_Mrs
5. Pclass_3

## 🚀 Quick Start

### Option 1: Kaggle (Recommended)

**[Open Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/the-titanic-project)** - Run instantly with pre-installed libraries!

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/tutkufurkan/Kaggle-Competitions.git
cd Kaggle-Competitions/Titanic-Survival-Prediction

# Install dependencies (if needed)
pip install numpy pandas scikit-learn xgboost matplotlib seaborn plotly jupyter

# Download dataset from Kaggle
# https://www.kaggle.com/competitions/titanic/data

# Run notebook
jupyter notebook the-titanic-project.ipynb
```

## 📊 Dataset

**Download from Kaggle:** [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

The dataset contains passenger information including age, sex, ticket class, fare, cabin, and survival status.

## 📁 Project Structure

```
Titanic-Survival-Prediction/
├── the-titanic-project.ipynb    # Main notebook
├── the-titanic-project.py       # Python script version
├── README.md                    # This file
└── LICENSE                      # Apache 2.0 license
```

## 🎓 Learning Outcomes

**Machine Learning Concepts:**
- Feature engineering techniques
- Handling missing data strategically
- Ensemble learning (Voting Classifier)
- Cross-validation for model evaluation
- Overfitting prevention through regularization

**Technical Skills:**
- Pandas for data manipulation
- Scikit-learn for ML pipeline
- XGBoost for gradient boosting
- Plotly for interactive visualizations
- Stratified K-Fold cross-validation

## 🔧 Usage Example

```python
# Load and prepare data
train_df = pd.read_csv('train.csv')

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Train voting ensemble
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(max_depth=5)),
        ('xgb', XGBClassifier(learning_rate=0.03)),
        ('gb', GradientBoostingClassifier(learning_rate=0.03))
    ],
    voting='soft'
)

# Cross-validation
cv_scores = cross_val_score(voting, X, y, cv=10)
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

## 💡 Key Takeaways

**What Worked:**
✅ Title extraction from names (significant predictor)  
✅ Aggressive regularization (max_depth=3-5)  
✅ Log-transform on Fare (handles skewness)  
✅ Soft voting ensemble (better than hard voting)  
✅ 10-fold CV (reliable performance estimate)

**What Didn't Work:**
❌ Deep trees (max_depth > 6) - overfitting  
❌ Complex feature interactions - noise  
❌ Deleting outliers - data loss  
❌ Cabin feature engineering - too many missing values

## 🔗 Related Projects

**My Machine Learning Series:**

- 🎯 **Classification Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Classifications-Models)

- 📈 **Regression Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Regression-Models)

- 🔍 **Clustering Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Clustering-Models)

- 🚀 **Advanced Topics** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Advanced-Topics)

## 📚 References

- **Kaggle Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Inspired by:** [DataiTeam Titanic EDA](https://www.kaggle.com/code/kanncaa1/dataiteam-titanic-eda)

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 📞 Connect

- **Kaggle:** [@dandrandandran2093](https://www.kaggle.com/dandrandandran2093)
- **GitHub:** [@tutkufurkan](https://github.com/tutkufurkan)
- **Website:** [tutkufurkan.com](https://tutkufurkan.com)

---

⭐ **Star this repo if it helped you learn!**

🌐 [tutkufurkan.com](https://tutkufurkan.com)
