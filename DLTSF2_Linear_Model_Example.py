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

#%% Tensorflow Deep Learning Linear Regression Model Example

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

# Inspecting feature relationships
sns.pairplot(df[["MPG","Cylinders","Displacement","Weight","Acceleration"]], diag_kind="kde")
plt.show()

# Splitting Train/Test data
train_data = df.sample(frac=0.8, random_state=42)
train_targets = train_data.pop("MPG")
test_data = df.drop(train_data.index)
test_targets = test_data.pop("MPG")

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
train_targets = scaler.fit_transform(train_targets.values.reshape(-1,1))
test_data = scaler.fit_transform(test_data)
test_targets = scaler.fit_transform(test_targets.values.reshape(-1,1))

# Hyperparametrs
epochs = 1000
hidden_units = 20 
loss_function = "mse"
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Model structure
net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=hidden_units,
                          activation="relu",
                          use_bias=True,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="zeros",
                          input_shape=[train_data.shape[1]]),
    tf.keras.layers.Dense(units=1, activation="linear")])

# Model Optimizer                       
net.compile(loss=loss_function, optimizer=optimizer, metrics=["mse", "mae"])

# Model Training
model = net.fit(train_data,
                train_targets,
                epochs=epochs,
                validation_data=(test_data, test_targets),
                verbose=1,
                use_multiprocessing=True,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Model summary 
model.model.summary()

# Checking for overfitting or underfitting
plt.subplot(1,2,1)
plt.plot(model.history["loss"], c="blue", label="Train MSE")
plt.plot(model.history["val_loss"], c="orange", label="Test MSE")
plt.title("Deep Learning Model - Loss Function Evaluation")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.subplot(1,2,2)
plt.plot(model.history["mae"], c="green", label="Train MAE")
plt.plot(model.history["val_mae"], c="red", label="Test MAE")
plt.title("Deep Learning Model - Loss Function Evaluation")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Absolut Error")
plt.show()

# Model evaluation metrics
dl_loss, dl_mse, dl_mae = model.model.evaluate(test_data, test_targets, verbose=0)
print("Model Mean Squared Error: {}".format(round(dl_mse, 5)))
print("Model Mean Absolut Error: {}".format(round(dl_mae, 5)))

# Comparing predictions with targets
dl_pred = model.model.predict(test_data, use_multiprocessing=True).flatten()
plt.scatter(dl_pred, test_targets, c="green", marker="o", alpha=0.7)
plt.title("Deep Learning Model - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.show()

# Checking normality assumption
from scipy.stats import shapiro
dl_resid = (test_targets - dl_pred)
sns.distplot(dl_resid, color="purple")
plt.title("Deep Learning Regression Model - Residuals")
plt.show()
shapiro(dl_resid)
