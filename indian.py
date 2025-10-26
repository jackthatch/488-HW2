import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

indianR = loadmat("indianR.mat")
indianGth = loadmat("indian_gth.mat")

x = indianR['X'].T 
y = indianGth['gth'].flatten()

# remove background class
mask = y > 0
x = x[mask]
y = y[mask] 

x_scaled = StandardScaler().fit_transform(x)

pca = PCA().fit(x_scaled)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10,10))
plt.bar(range(1, 203), explained_variance, width=0.5)
plt.xticks(range(1, 203, 10))
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Indian Pines — Explain Variance per Component")
plt.tight_layout()
plt.show()
## END 1A ##

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(10,10))
scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, s=6, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Indian Pines — PCA 2D Projection")
plt.tight_layout()
plt.show()

## END 1B ##

lda = LDA(n_components=2)
x_lda = lda.fit_transform(x_scaled, y)

plt.figure(figsize=(10,10))
scatter = plt.scatter(x_lda[:,0], x_lda[:,1], c=y, s=15, alpha=0.7)
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Indian Pines — LDA 2D Projection")
plt.tight_layout()
plt.show()

## END 1C ## 

training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
classifications = [
    ("Naive Bayes", GaussianNB()),
    ("SVM Linear", SVC(kernel="linear")),
    ("SVM RBF", SVC(kernel="rbf"))
]

def supervised_classification(x, y, training_size):
    res = {}

    for name, model in classifications:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_size, stratify=y, random_state=0
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        res[name] = round(accuracy, 4)
    
    return res

def supervised_classification_classwise(x, y, training_size):
    res = {}

    for name, model in classifications:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_size, stratify=y, random_state=0
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        classes = np.unique(y_test)
        class_acc = {}

        for c in classes:
            mask = (y_test == c)
            class_acc[c] = np.mean(y_pred[mask] == y_test[mask])

        res[name] = class_acc
    
    return res

results_base = {name: [] for name, _ in classifications}
results_pca = {name: [] for name, _ in classifications}
results_lda = {name: [] for name, _ in classifications}

for size in training_sizes:
    base_res = supervised_classification(x_scaled, y, size)
    pca_res  = supervised_classification(x_pca,  y, size)
    lda_res  = supervised_classification(x_lda,  y, size)

    for name, _ in classifications:
        results_base[name].append(base_res[name])
        results_pca[name].append(pca_res[name])
        results_lda[name].append(lda_res[name])

    if (size == 0.4):
        classwise_base_res = supervised_classification_classwise(x_scaled, y, size)
        classwise_pca_res  = supervised_classification_classwise(x_pca,  y, size)
        classwise_lda_res  = supervised_classification_classwise(x_lda,  y, size)

## just printing classwise data and will convert to table in doc

df_base = pd.DataFrame(classwise_base_res).round(3)
df_pca  = pd.DataFrame(classwise_pca_res).round(3)
df_lda  = pd.DataFrame(classwise_lda_res).round(3)

print("classwise no dim")
print(df_base)

print("\nclasswise pca")
print(df_pca)

print("\nclasswise lda")
print(df_lda)


sizes_percent = [int(s*100) for s in training_sizes]

plt.figure(figsize=(10,10))
for name in results_pca:
    plt.plot(sizes_percent, results_pca[name], marker='o', label=name)
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Indian Pines – PCA Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,10))
for name in results_lda:
    plt.plot(sizes_percent, results_lda[name], marker='o', label=name)
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Indian Pines – LDA Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,10))
for name in results_base:
    plt.plot(sizes_percent, results_base[name], marker='o', label=name)
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Indian Pines – No Dimensionality Reduction Accuracy")
plt.legend()
plt.grid(True)
plt.show()