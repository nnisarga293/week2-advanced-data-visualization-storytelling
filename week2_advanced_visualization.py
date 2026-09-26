import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load public Wine Recognition dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["class"] = df["target"].map(dict(enumerate(wine.target_names)))
df = df.drop(columns=["target"])

features = wine.feature_names
print(df.head())
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().sum())

# Class distribution
df["class"].value_counts().sort_index().plot(kind="bar")
plt.title("Wine Classes: Sample Distribution")
plt.xlabel("Wine class")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()

# Proline box plot
df.boxplot(column="proline", by="class")
plt.title("Proline Concentration by Wine Class")
plt.suptitle("")
plt.xlabel("Wine class")
plt.ylabel("Proline")
plt.tight_layout()
plt.show()

# Scatter plot
for cls in wine.target_names:
    sub = df[df["class"] == cls]
    plt.scatter(sub["alcohol"], sub["flavanoids"], label=cls, alpha=0.8)
plt.title("Alcohol vs Flavanoids")
plt.xlabel("Alcohol")
plt.ylabel("Flavanoids")
plt.legend()
plt.tight_layout()
plt.show()

# Correlation matrix
corr = df[features].corr()
plt.imshow(corr.values, aspect="auto")
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=70)
plt.yticks(range(len(features)), features)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# PCA
X = StandardScaler().fit_transform(df[features])
pc = PCA(n_components=2, random_state=42).fit_transform(X)
for cls in wine.target_names:
    mask = df["class"].values == cls
    plt.scatter(pc[mask, 0], pc[mask, 1], label=cls, alpha=0.8)
plt.title("PCA Projection of Wine Profiles")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()
