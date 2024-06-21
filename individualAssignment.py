import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import stats

log_reg_accuracies = []
ann_accuracies = []
dataPath = "HR_data.csv"
data = pd.read_csv(dataPath)

features = ["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC"]
X = data[features]
y = data["Frustrated"]
groups = data["Individual"]

gkf = GroupKFold(n_splits=5)

for train_index, test_index in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_train_onehot, y_test_onehot = tf.keras.utils.to_categorical(y_train, num_classes=11), tf.keras.utils.to_categorical(y_test, num_classes=11)
    
    #Log Reg model
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_train, y_train)
    y_prediction_model1 = model1.predict(X_test)
    accuracy_model1 = accuracy_score(y_test, y_prediction_model1)
    log_reg_accuracies.append(accuracy_model1)
    
    #ANN model
    tf.random.set_seed(42)
    model2 = tf.keras.models.Sequential([tf.keras.layers.Dense(16, input_dim=X_train.shape[1], activation="relu"),
                                         tf.keras.layers.Dense(16, activation="relu"),
                                         tf.keras.layers.Dense(11, activation="softmax")])
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    model2.fit(X_train, y_train_onehot, epochs=100, batch_size=32)
    loss, accuracy_model2 = model2.evaluate(X_test, y_test_onehot, verbose=0)
    ann_accuracies.append(accuracy_model2)
    
log_reg_mean = np.mean(log_reg_accuracies)
log_reg_std = np.std(log_reg_accuracies)
ann_mean = np.mean(ann_accuracies)
ann_std = np.std(ann_accuracies)

print(f'Logistic Regression - Mean Accuracy: {log_reg_mean}, Std: {log_reg_std}')
print(f'ANN - Mean Accuracy: {ann_mean}, Std: {ann_std}')

# Paired t-test
t_stat, p_value = stats.ttest_rel(log_reg_accuracies, ann_accuracies)
print(f'Paired t-test - t-statistic: {t_stat}, p-value: {p_value}')


differences = np.array(log_reg_accuracies) - np.array(ann_accuracies)

shapiro_test = stats.shapiro(differences)
print(f'Shapiro-Wilk Test - Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}')

kolmogorov_test = stats.kstest(differences, "norm", args=(np.mean(differences), np.std(differences)))
print(f'Kolmogorov-Smirnov Test - Statistic: {kolmogorov_test.statistic}, p-value: {kolmogorov_test.pvalue}')

stat, p_value_wilcoxon = stats.wilcoxon(log_reg_accuracies, ann_accuracies)
print(f'Wilcoxon signed-rank test - Statistic: {stat}, p-value: {p_value_wilcoxon}')

print("LOG", log_reg_accuracies)
print("ANN", ann_accuracies)
plt.figure(figsize=(8,6))
stats.probplot(differences, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()