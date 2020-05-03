# Date: 29/04/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8 

# Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tf_docs 
import tensorflow_docs.plots
import tensorflow_docs.modeling
sns.set()                                                    

#%% Tensorflow Linear Model Example - https://www.tensorflow.org/tutorials/keras/regression

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#data_path = tf.keras.utils.get_file("auto-mpg.data", data_url)
data_path = "C:\\Users\\pedro\\.keras\\datasets\\auto-mpg.data" 

features = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
df = pd.read_csv(data_path, names=features, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
df.head()
df.shape

df.isna().sum()
df = df.dropna()
df.shape

df["Origin"] = df["Origin"].map({1:"USA",2:"Europe",3:"Japan"})
df = pd.get_dummies(df, prefix="", prefix_sep="")
df.columns

train_data = df.sample(frac=0.8, random_state=42)
train_targets = train_data.pop("MPG")
test_data = df.drop(train_data.index)
test_targets = test_data.pop("MPG")

sns.pairplot(train_data[["MPG","Cylinders","Displacement","Weight","Acceleration"]], diag_kind="kde")
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data[features[1:-1]])
train_targets = scaler.fit_transform(train_targets.values.reshape(-1,1))
test_data = scaler.fit_transform(test_data[features[1:-1]])
test_targets = scaler.fit_transform(test_targets.values.reshape(-1,1))

model_hist = {}

epochs = 1000
hidden_units = 32
loss_function = "mse"
learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate)

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
                          
net.compile(loss=loss_function,
            optimizer=optimizer,
            metrics=["mse"])

model_hist["fit2"] = net.fit(train_data,
                            train_targets,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=1,
                            use_multiprocessing=True,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

plotter = tf_docs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot(model_hist, metric="mse")
plt.title("Model Loss Function Evaluation")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()

loss, mse, mae = model_hist["fit2"].model.evaluate(test_data, test_targets, verbose=2)
print("Testing set Mean Squared Error: {:5.2f} MPG".format(mse))
print("Testing set Mean Absolut Error: {:5.2f} MPG".format(mae))

predictions = model_hist["fit2"].model.predict(test_data, use_multiprocessing=True).flatten()
plt.scatter(predictions, test_targets, c="green", marker="o", alpha=0.8)
plt.title("Scatterplot - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.show()

from scipy.stats import shapiro
residuals = test_targets - predictions
sns.distplot(residuals, color="orange")
plt.show()
norm_test = shapiro(residuals) # H0: Non-Normality x H1: Normality
norm_test # [Test Statistic, P-value]

#%% 
