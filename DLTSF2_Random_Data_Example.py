
#%% Binary Classification Loss Functions

# Random data generation
from sklearn.datasets import make_circles
inputs, targets = make_circles(n_samples=10000, shuffle=True, noise=0.1, random_state=42)

# Data preview
for i in range(2):
  samples_ix = np.where(targets == i)
  plt.scatter(inputs[samples_ix, 0], inputs[samples_ix, 1], label=str(i))
plt.title("Binary Classification Example - Random Circles")
plt.legend()
plt.show()

# Binary Cross-Entropy
# . Default loss function for most binary classification problems.
# . Intended for use when the binary labels are in the set {0,1}.
# . Like MSE, its the preffered loss function under maximum likelihood estimation.
# . Calculates the average difference between the actual and predicted probabilty
#   distribution for predicting class 1. A perfect cross-entropy value is 0.

x_test, x_train, y_test, y_train = train_test_split(inputs, targets, test_size=0.8, random_state=42)

epochs = 1000
learning_rate = 0.01
loss_function = "binary_crossentropy"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

bc_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_shape=[inputs.shape[1]]),
  tf.keras.layers.Dense(units=1, activation="sigmoid")])

bc_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

bc_model = bc_net.fit(x=x_train,
                      y=y_train,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test,y_test),
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Hinge Loss
# . Alternative to the binary cross-entropy loss function, developed for SVM models.
# . Inteded for use when the labels are for set {-1,1}.
# . Encourages samples to have the correct sign, penalizing greater for differences in signs.
# . Requires an modification to the activation function due to the change in domain.
# . Reports on its performance are somewhat mixed, it may perform worse the cross-entropy.

hinge_y_train = np.copy(y_train)
hinge_y_test = np.copy(y_test)
hinge_y_train[np.where(hinge_y_train == 0)] = -1
hinge_y_test[np.where(hinge_y_test == 0)] = -1

epochs = 1000
learning_rate = 0.01
loss_function = "hinge"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

hinge_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_dim=inputs.shape[1]),
  tf.keras.layers.Dense(units=1, activation="tanh")])

hinge_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

hinge_model = hinge_net.fit(x=x_train,
                            y=hinge_y_train,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, hinge_y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Squared Hinge Loss
# . The Hinge loss functions has many extensions, one of which is the Squared Hinge Loss.
# . Has the effect of smoothing out the error function and makes it numerically more stable.
# . At times, if the Hinge Loss functions does not perform well, Squared Hinge Loss might.

epochs = 1000
learning_rate = 0.01
loss_function = "squared_hinge"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

sqr_hinge_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_dim=inputs.shape[1]),
  tf.keras.layers.Dense(units=1, activation="tanh")])

sqr_hinge_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

sqr_hinge_model = sqr_hinge_net.fit(x=x_train,
                                    y=hinge_y_train,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test, hinge_y_test),
                                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Comparing the learning curves of the models
plt.plot(bc_model.history['loss'], label='Train MSE', c="blue")
plt.plot(bc_model.history['val_loss'], label='Test MSE', c="orange")
plt.title("Binary Cross-Entropy Model")
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.legend()
plt.show()

plt.plot(hinge_model.history['loss'], label='Train MSLE', c="purple")
plt.plot(hinge_model.history['val_loss'], label='Test MSLE', c="orange")
plt.title("Hinge Model")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.legend()
plt.show()

plt.plot(sqr_hinge_model.history['loss'], label='Train MAE', c="green")
plt.plot(sqr_hinge_model.history['val_loss'], label='Test MAE', c="orange")
plt.title("Squared Hinge Model")
plt.xlabel("Epochs")
plt.ylabel("Square Hinge Loss")
plt.legend()
plt.show()

# Comparing evaluation metrics throughout the models
def compute_metrics(model, test_data, test_targets):
  """ 
  -----------
  Description
  -----------
    Given an classification model, computes various model evaluation metrics.

  ----------
  Parameters
  ----------
    . model: Tensorflow.python.keras.callbacks.History object
    . test_data: Model test set.
    . test_targets: Model test targets.

  -------
  Returns
  -------
    . predictions: Model predictions.
    . baseline_model: Associated Null-Accuracy "dumb" model.
    . confusion_matrix: Confusion-Matrix plot of the model.
    . metrics_df: Dataframe containing all the computed evalutaions metrics.
    . ROC_plot: The ROC plot of the model.
    . AUC_plot: The AUC plot of the model.
  """

  predictions = model.predict(test_data, verbose=0, use_multiprocessing=True).flatten()
  null_accuracy_model = np.count_nonzero(test_targets)/len(test_targets)
  accuracy = 
  true_positives = np.count_nonzero((predictions * ))
  true_negatives
  false_negatives
  false_positives
  f1_score

  # True positives = 
  tp = 
  tn = np.count_nonzero(((predict_labels-1)*(true_labels-1)))

bc_pred = bc_model.predict(x_test, verbose=0, use_multiprocessing=True).flatten()
 
    FP = tf.count_nonzero(prediction_integer * (labels - 1),
        name="False_Positives", dtype=tf.int32)
 
    FN = tf.count_nonzero((prediction_integer - 1) * labels,
        name="False_Negatives", dtype=tf.int32)



#%% Multi-class Classification Loss Functions