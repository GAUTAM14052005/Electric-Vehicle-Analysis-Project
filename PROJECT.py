# ============================================
# 🚀 1. BUSINESS UNDERSTANDING (CRISP-DM)
# ============================================
# Goal:
# Analyze Electric Vehicle population trends and predict future values

# ============================================
# 📚 2. IMPORT LIBRARIES
# ============================================
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================
# 📂 3. DATA LOADING
# ============================================
df = pd.read_csv(r"C:\Users\ahuja\Downloads\EV_data.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)

# ============================================
# 🏷️ 4. RENAMING COLUMNS
# ============================================
df.columns = df.columns.str.lower().str.replace(" ", "_")

print("\nColumns after renaming:\n", df.columns)

# ============================================
# 🔍 5. DATA UNDERSTANDING
# ============================================
print("\nInfo:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

# ============================================
# 🧹 6. DATA CLEANING
# ============================================

# Drop duplicates
df = df.drop_duplicates()
if 'date' in df.columns:
    df=df.drop(columns=['date'])
# Fill missing values
for col in df.select_dtypes(include='number').columns:
    df[col]=df[col].fillna(df[col].mean())
for col in df.columns:
    if df[col].dtype=='object':
       df[col]=df[col].fillna(df[col].mode()[0])

# ============================================
# 📊 7. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# Histogram (numeric)
df.hist(figsize=(12,8),color='orange',edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.tight_layout()
#facecolor='#f5f7fa'
plt.show()

# ============================================
# 📊 BAR CHART (Categorical)
# ============================================
if 'county' in df.columns:
    df['county'].value_counts().head(10).plot(kind='bar',color='coral')
    plt.title("Top 10 Counties")
    plt.tight_layout()
    plt.show()

# ============================================
# 🥧 PIE CHART
# ============================================
if 'county' in df.columns:
    df['county'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
    plt.title("Top 5 Counties Distribution")
    plt.ylabel("")
    plt.show()

# ============================================
# 📈 SCATTER PLOT
# ============================================
numeric_cols = df.select_dtypes(include='number').columns

if len(numeric_cols) >= 2:
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title("Scatter Plot")
    plt.show()
# ============================================
# 🚨 OUTLIER DETECTION & TREATMENT (IQR METHOD)
# ============================================

numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

    # Option 1: Capping (Recommended ✅)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Verify after treatment
print("\nOutlier treatment completed.")
fig,ax=plt.subplots(figsize=(10,6))
ax.set_facecolor('beige')
sns.set_style("whitegrid")
sns.boxplot(data=df,ax=ax)
#plt.figure(figsize=(10,5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot Before/After Outlier Treatment")
plt.xticks(rotation=90)
plt.show()

# ============================================
# 🔥 HEATMAP (Correlation)
# ============================================
plt.figure(figsize=(10,6))

sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================
# ⚙️ 8. FEATURE ENGINEERING
# ============================================

# Example: growth rate (if year column exists)
if 'year' in df.columns and 'county' in df.columns:
    numeric_cols=df.select_dtypes(include='number').columns
    df['growth'] = df.groupby('county')[numeric_cols[0]].pct_change()
    df['growth']=df['growth'].fillna(0)

# ============================================
# 🔄 9. ENCODING (Categorical → Numeric)
# ============================================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype=='object':
        try:
            df[col]=le.fit_transform(df[col].astype(str))
        except Exception as e:
                print(f"Dropping column {col}:{e}")
                df=df.drop(columns[col])

# ============================================
# 🎯 10. FEATURE & TARGET SELECTION
# ============================================

# Choose last numeric column as target (you can modify)
target = 'percent_electric_vehicles'

X = df.drop(columns=[target])
y = df[target]
X=X.select_dtypes(include='number')
X=X.replace([np.inf, -np.inf],np.nan)
X=X.fillna(X.mean())
y=y.fillna(y.mean())
print("\nNaN in X:",X.isnull().sum().sum())
print("NaN in y:",y.isnull().sum())
print("X dtypes:\n",X.dtypes)

# ============================================
# ✂️ 11. TRAIN TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 🤖 12. MACHINE LEARNING (LINEAR REGRESSION)
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)
#model = RandomForestRegressor()
#model.fit(X_train, y_train)

# ============================================
# 🔮 13. PREDICTION
# ============================================
y_pred = model.predict(X_test)

# ============================================
# 📉 14. EVALUATION
# ============================================
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# ============================================
# 📊 15. VISUALIZE PREDICTION
# ============================================
plt.scatter(y_test, y_pred,color='green',alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()

# ============================================
# 🧠 16. CONCLUSION (PRINT INSIGHTS)
# ============================================
print("\nModel trained successfully!")
print("You can improve using Random Forest or XGBoost.")
