# Date: 04/05/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8 

# References
# https://www.tensorflow.org/tutorials/keras/regression

# Modules
import myconfigs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tf_docs 
import tensorflow_docs.plots
import tensorflow_docs.modeling
sns.set()                                                    

#%% (Modified) Tensorflow Deep Learning Linear Regression Model Example

# data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
# data_path = tf.keras.utils.get_file("auto-mpg.data", data_url) 

# Reading data
features = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
df = pd.read_csv(myconfigs.dltsf2_1, names=features, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
df.head()

# Removing missign data
df.isna().sum()
df = df.dropna()

# One-hot encoding of dummy variables
df["Origin"] = df["Origin"].map({1:"USA",2:"Europe",3:"Japan"})
df = pd.get_dummies(df, prefix="", prefix_sep="")

# Splitting Train/Test data
train_data = df.sample(frac=0.8, random_state=42)
train_targets = train_data.pop("MPG")
test_data = df.drop(train_data.index)
test_targets = test_data.pop("MPG")

# Inspecting feature relationships
sns.pairplot(train_data[["MPG","Cylinders","Displacement","Weight","Acceleration"]], diag_kind="kde")
plt.show()

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data[features[1:-1]])
train_targets = scaler.fit_transform(train_targets.values.reshape(-1,1))
test_data = scaler.fit_transform(test_data[features[1:-1]])
test_targets = scaler.fit_transform(test_targets.values.reshape(-1,1))

# Hyperparametrs
model_hist = {}
epochs = 100
hidden_units = 32
loss_function = "mse"
learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Model structure
net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=hidden_units,
                          activation="relu",
                          use_bias=True,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="zeros",
                          input_shape=[len(pd.DataFrame(train_data).keys())]),
    tf.keras.layers.Dense(units=hidden_units,
                          activation="relu"),
    tf.keras.layers.Dense(1)])

# Model Optimizer                       
net.compile(loss=loss_function, optimizer=optimizer, metrics=["mse", "mae"])

# Model Training
model_hist["fit"] = net.fit(train_data,
                            train_targets,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=1,
                            use_multiprocessing=True,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

# Checking for over/underfitting
plotter = tf_docs.plots.HistoryPlotter(smoothing_std=2)
plt.subplot(1,2,1)
plotter.plot(model_hist, metric="mse")
plt.title("Deep Learning Model - Loss Function Evaluation")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plotter = tf_docs.plots.HistoryPlotter(smoothing_std=2)
plt.subplot(1,2,2)
plotter.plot(model_hist, metric="mse")
plt.title("Deep Learning Model - Loss Function Evaluation")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolut Error")
plt.show()

#%% Fitting traditional OLS regression

from sklearn.linear_model import LinearRegression
ols = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=False)
ols.fit(train_data, train_targets)

#%% Comparing both models

# Model evaluation metrics
dl_loss, dl_mse, dl_mae = model_hist["fit"].model.evaluate(test_data, test_targets, verbose=0)
print("Deep Learning Model MSE: {}".format(round(dl_mse, 5)))
print("Deep Learning Model MAE: {}".format(round(dl_mae, 5)))

from sklearn.metrics import mean_squared_error, mean_absolute_error
ols_pred = ols.predict(test_data)
print("Traditional OLS Model MSE: {}".format(round(mean_squared_error(test_targets, ols_pred),5)))
print("Traditional OLS Model MAE: {}".format(round(mean_absolute_error(test_targets, ols_pred),5)))

# Comparing predictions on both models
dl_pred = model_hist["fit"].model.predict(test_data, use_multiprocessing=True).flatten()
plt.subplot(1,2,1)
plt.scatter(dl_pred, test_targets, c="green", marker="o", alpha=0.7)
plt.title("Deep Learning Model - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.subplot(1,2,2)
plt.scatter(ols_pred, test_targets, c="blue", marker="o", alpha=0.7)
plt.title("OLS Regression - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.show()

# Checking normality assumption on both models
from scipy.stats import shapiro
dl_resid = (test_targets - dl_pred)
ols_resid = (test_targets - ols_pred)
plt.subplot(1,2,1)
sns.distplot(dl_resid, color="orange")
plt.title("Deep Learning Regression - Residuals")
plt.subplot(1,2,2)
sns.distplot(ols_resid, color="red")
plt.title("OLS Regression - Residuals")
plt.show()
shapiro(dl_resid)
shapiro(ols_resid)