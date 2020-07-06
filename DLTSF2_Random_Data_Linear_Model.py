# Date: 15/05/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8                                                     

# References:
# . https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

# Modules
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Random data generation
from sklearn.datasets import make_regression
inputs, targets = make_regression(n_samples=10000,
                                  n_features=20,
                                  n_informative=15,
                                  n_targets=1,
                                  bias=5,
                                  shuffle=True,
                                  random_state=42,
                                  noise=0.1)  

# Mean Squared Error
# . Default loss function for regression problems.
# . Preffered under the inference framework of Maximum Likelihood estimation.
# . If residuals are gaussian and non-serial correlated, OLS = MLE.
# . The squaring penalizes harder the model for making bigger mistakes.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(scaler.fit_transform(inputs),
                                                    scaler.fit_transform(targets.reshape(-1,1)),
                                                    test_size=0.2,
                                                    random_state=42)

epochs = 1000
learning_rate = 0.01
loss_function = "mean_squared_error"
optmizer = tf.keras.optimizers.Adam(lr=learning_rate)

net = tf.keras.Sequential([
      tf.keras.layers.Dense(units=50,
                            activation="relu",
                            use_bias=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer="zeros",
                            input_shape=[inputs.shape[1]]),
      tf.keras.layers.Dense(units=1, activation="linear")])

net.compile(optimizer=optmizer, loss=loss_function, metrics=["mse"])

mse_model = net.fit(x_train,
                    y_train,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    use_multiprocessing=True,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=20)])

# Mean Squared Logarithmic Error
# . MAY be more suitable when the target variable has more spread out values.
# . It has the effect of relaxing the punishing effect of large differences in large predicted values.
# . Instead, one can take the log of the features and then calculate MSE.
# . May be more suitable when predicting on unscaled data.

u_x_train, u_x_test, u_y_train, u_y_test = train_test_split(inputs, targets, test_size=0.8, random_state=42)

epochs = 1000
learning_rate = 0.01
loss_function = "mean_squared_logarithmic_error"
optmizer = tf.keras.optimizers.Adam(lr=learning_rate)

net = tf.keras.Sequential([
      tf.keras.layers.Dense(units=50,
                            activation="relu",
                            use_bias=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer="zeros",
                            input_shape=[inputs.shape[1]]),
      tf.keras.layers.Dense(units=1, activation="linear")])

net.compile(optimizer=optmizer, loss=loss_function, metrics=["mse"])

msle_model = net.fit(u_x_train,
                     u_y_train,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(u_x_test, u_y_test),
                     use_multiprocessing=True,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=20)])

# Mean Absolute Error Loss Function
# . More suitable when either the target variable or feature set has outiliers.

epochs = 1000
learning_rate = 0.01
loss_function = "mean_absolute_error"
optmizer = tf.keras.optimizers.Adam(lr=learning_rate)

net = tf.keras.Sequential([
      tf.keras.layers.Dense(units=50,
                            activation="relu",
                            use_bias=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer="zeros",
                            input_shape=[inputs.shape[1]]),
      tf.keras.layers.Dense(units=1, activation="linear")])

net.compile(optimizer=optmizer, loss=loss_function, metrics=["mae"])

mae_model = net.fit(x_train,
                    y_train,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    use_multiprocessing=True,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=20)])

# Comparing the learning curves of the models
plt.plot(mse_model.history['loss'], label='Train MSE', c="blue")
plt.plot(mse_model.history['val_loss'], label='Test MSE', c="orange")
plt.title("Mean Squared Error Model")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(msle_model.history['loss'], label='Train MSLE', c="purple")
plt.plot(msle_model.history['val_loss'], label='Test MSLE', c="orange")
plt.title("Mean Squared Logarithmic Error Model")
plt.xlabel("Epochs")
plt.ylabel("MSLE")
plt.legend()
plt.show()

plt.plot(mae_model.history['loss'], label='Train MAE', c="green")
plt.plot(mae_model.history['val_loss'], label='Test MAE', c="orange")
plt.title("Mean Absolut Error Model")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()

# Comparing the predictive performance of the models
mse_pred = mse_model.model.predict(x_test, verbose=0, use_multiprocessing=True).flatten()
plt.scatter(mse_pred, y_test, c="blue", marker="o", alpha=0.7)
plt.title("MSE Model - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Target") 
plt.show()

msle_pred = msle_model.model.predict(u_x_test, verbose=0, use_multiprocessing=True).flatten()
plt.scatter(msle_pred, u_y_test, c="purple", marker="o", alpha=0.7)
plt.title("MSLE Model - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Target") 
plt.show()

mae_pred = mae_model.model.predict(x_test, verbose=0, use_multiprocessing=True).flatten()
plt.scatter(mae_pred, y_test, c="green", marker="o", alpha=0.7)
plt.title("MAE Model - Predicted x Targets")
plt.xlabel("Predicted")
plt.ylabel("Target") 
plt.show()

# Checking normality assumption for the models
from scipy.stats import shapiro
mse_resid = (mse_pred - y_test)
sns.distplot(mse_resid, color="blue")
plt.title("Mean Squared Error Model Residuals")
plt.show()
print("MSE normality p-value: %.4f" % shapiro(mse_resid)[1])

msle_resid = (msle_pred - u_y_test)
sns.distplot(msle_resid, color="purple")
plt.title("Mean Squared Logarithmic Error Model Residuals")
plt.show()
print("MSLE normality p-value: %.4f" % shapiro(msle_resid)[1])

mae_resid = (mae_pred - y_test)
sns.distplot(mae_resid, color="green")
plt.title("Mean Absolut Error Model Residuals")
plt.show()
print("MAE Model normality p-value: %.4f" % shapiro(mae_resid)[1])