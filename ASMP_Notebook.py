############################################################################
# Date : 08/04/2020                                                        #
# Author : Pedro H. Puntel                                                 #
# Email : pedro.puntel@gmail.com                                           #
# Topic : 365 Data Science Course - Advanced Statistical Methods in Python #
# Ecoding : UTF-8                                                          #
############################################################################

#######
# Setup
#######
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set() # overrides matplotlib's plot style

#################################################################
# Linear Regression (Example 1) - Predicting GPA using SAT scores
#################################################################
filename = "D:\Python\Courses\365_Data_Science\Advanced Statistcal Methods in Python\Data\sat_gpa.csv"
df = pd.read_csv(filename)
df.head()
df.describe()

y = df["GPA"]
x1 = df["SAT"]

plt.scatter(x1, y, c="blue", marker="o", alpha=0.7)
plt.title("SAT x GPA")
plt.xlabel("SAT Scores")
plt.ylabel("GPA Income")
plt.show()

x = sm.add_constant(x1)
fit1 = sm.OLS(y, x).fit()
fit1.summary()
yhat = 0.0017*x1 + 0.275

plt.scatter(x1, y, c="blue", marker="o", alpha=0.7)
fig = plt.plot(x1, yhat, lw=4, c="orange")
plt.title("SAT x GPA - Ordinary Least Squares Fit")
plt.legend(loc="upper left")
plt.xlabel("SAT Scores")
plt.ylabel("GPA Income")
plt.show()

#######################################################################################
# Linear Regression (Example 2) - Predicting GPA using SAT scores and Attendance Ratios
#######################################################################################
# Introduction of a Dummy variable such that:
# . Yes represents >= 75% attendance
# . No represents < 75% attendance

filename = "D:/Python/Courses/365_Data_Science/Advanced Statistcal Methods in Python/Data/sat_gpa_attd.csv"
df2 = pd.read_csv(filename)
df2.head()

df2["Attendance"] = df2["Attendance"].map({"Yes":1, "No":0})
df2.head()
df2.describe()

y = df2["GPA"]
x = df2[["SAT","Attendance"]]

plt.scatter(df2.SAT[df2.Attendance == 1], df2.GPA[df2.Attendance == 1], c="Green", marker="o", alpha=0.7, label=">= 75%")
plt.scatter(df2.SAT[df2.Attendance == 0], df2.GPA[df2.Attendance == 0], c="Red", marker="o", alpha=0.7, label="< 75%")
plt.title("SAT x GPA - With Attendance Effects")
plt.legend(loc="lower right")
plt.xlabel("SAT Scores")
plt.ylabel("GAP Income")
plt.show()

x = sm.add_constant(x)
fit2 = sm.OLS(y,x).fit()
fit2.summary()
yhat_1 = 0.0014*x["SAT"] + (0.6439 + 0.2226)
yhat_0 = 0.0014*x["SAT"] + 0.6439

plt.scatter(df2.SAT[df2.Attendance == 1], df2.GPA[df2.Attendance == 1], c="Green", marker="o", alpha=0.7, label=">= 75%")
plt.scatter(df2.SAT[df2.Attendance == 0], df2.GPA[df2.Attendance == 0], c="Red", marker="o", alpha=0.7, label="< 75%")
plt.plot(df2.SAT, yhat_1, lw=2, c="Green", label=">= 75% fit")
plt.plot(df2.SAT, yhat_0, lw=2, c="Red", label="< 75% fit")
plt.title("SAT x GPA - OLS Fit with Attendance Effects")
plt.legend(loc="lower right")
plt.xlabel("SAT Scores")
plt.ylabel("GPA Income")
plt.show()

###################################################
# Linear Regression (Example 3) - Practical Example
###################################################
# Notes
# . There are variables (such as Price) with missing values.
# . Too many unique models of cars, making it impossible to make a dummy for it.
# . Numerical variables are very skewed. Must check for outliers!
# . There is a strange maximum value in the EngineV variable (99.9 is a old way of stating NA's).
# . Almost half of the vehicles are of Engine Type Diesel and of Body sedan.
# . Price variable resembles an exponential distribution, hence a candidate to transformation.
# . Multicolinearity is expected in our model since Mileage, Year and EngineV are naturally are
#   vehicle characteristics correlated to one another. Must check the VIF of those varialbles.
# . VIF suggests Year and EngineV to be the most colinear variables. We should drop the bigger one (Year).
# . Drop the first dummy to not introduce multicolinearity. For a variable with N categorical
#   classes, one should include N-1 dummies.
# . We want the residuals to be ~ N(0,Sigma). Many negative residuals (far from the mean) imply
#  that predictions are much higher than the targets. Definetly a place for improving of the model.
# . Dummies do count as varibales and must be taken into account when calculating adjusted R-Squared.
# . When interpreting the coefficiets for the dummy variables, we must always look for the category
#   which was omitted. For example, for the Brand dummy, we see no 'Brand_Audi', so 'Brand_BMW' being
#   equal to 0.01, means that, on average, BMW's are 0.01 percent expensive than Audi. The same applies
#   to the other dummies as well.

# OLS Assumptions
# . Linearity - The response variable can be modeled as a linear combination of the predictors
# . Endogenity - The errors of the model are uncorrelated to the predictor variables
# . Normality and Homocedasticity - The error term of the model is ~ N(0, Sigma), where Sigma = I*sigma2
# . Autocorrelation - The errors of them model are uncorrelated with each other
# . Multicolinearity - The predictor variables are not perfectly correlated

filename = "D:/Python/Courses/365_Data_Science/Advanced Statistcal Methods in Python/Data/cars_dataset.csv"
raw_data = pd.read_csv(filename)
raw_data.rename(columns={"Engine Type":"EngineTyp"}, inplace=True)
raw_data.head()
raw_data.columns
raw_data.describe(include="all")

raw_data.isnull().sum()
data_no_null = raw_data.dropna(axis=0)
data_no_null.shape # removed 320 rows
data_no_null.describe()

data_of_interest = data_no_null.drop(["Model"], axis=1)
data_of_interest.shape

sns.distplot(data_of_interest["Price"], color="green")
data_of_interest = data_of_interest[data_of_interest["Price"] < data_of_interest["Price"].quantile(0.99)]
sns.distplot(data_of_interest["Price"], color="green")
data_of_interest.describe()
data_of_interest.shape # removed 41 rows

sns.distplot(data_of_interest["Mileage"], color="blue")
data_of_interest = data_of_interest[data_of_interest["Mileage"] < data_of_interest["Mileage"].quantile(0.99)]
sns.distplot(data_of_interest["Mileage"], color="blue")
data_of_interest.describe()
data_of_interest.shape # removed 40 rows

sns.distplot(data_of_interest["EngineV"], color="red")
data_of_interest = data_of_interest[data_of_interest["EngineV"] < 6.5] # domain-knowledge cutting point
sns.distplot(data_of_interest["EngineV"], color="red")
data_of_interest.describe()
data_of_interest.shape # removed 50 rows

sns.distplot(data_of_interest["Year"], color="orange")
data_of_interest = data_of_interest[data_of_interest["Year"] > data_of_interest["Year"].quantile(0.01)]
sns.distplot(data_of_interest["Year"], color="orange")
data_of_interest.describe()
data_of_interest.shape  # removed 77 rows

data_final = data_of_interest.reset_index(drop=True)
data_final.shape
data_final.describe(include="all")

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_final['Year'], data_final['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_final['EngineV'], data_final['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_final['Mileage'], data_final['Price'])
ax3.set_title('Price and Mileage')

sns.distplot(data_final["Price"], color="green")
data_final["Price"] = np.log(data_final["Price"])
sns.distplot(data_final["Price"], color="green")
data_final.describe()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_final['Year'], data_final['Price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_final['EngineV'], data_final['Price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_final['Mileage'], data_final['Price'])
ax3.set_title('Log Price and Mileage')

from statsmodels.stats.outliers_influence import variance_inflation_factor
exog_vars = data_final[["Mileage","Year","EngineV"]]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(exog_vars.values, i) for i in range(exog_vars.shape[1])]
vif["Feature"] = exog_vars.columns

data_final = data_final.drop("Year", axis=1)
data_final.describe()

data_final = pd.get_dummies(data_final, drop_first=True)
data_final.head()
data_final.columns

from sklearn.preprocessing import StandardScaler
response = data_final["Price"]
features = data_final.drop("Price", axis=1)
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state=42)

reg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
reg.fit(x_train, y_train)
yhat = reg.predict(x_train)

plt.scatter(y_train, yhat, c="blue", marker="o", alpha=0.7)
plt.title("Vehicle Prices - Predicted x True values")
plt.xlabel("Predicted Prices")
plt.ylabel("True Prices")
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

residuals = (y_train - yhat)
sns.distplot(residuals, color="orange")

r2 = reg.score(x_train, y_train)
n = data_final.shape[0]
k = data_final.shape[1]
adj_r2 = 1 - (((1-r2)*(n-k-1)) / (n-k-1))

reg_summary = pd.DataFrame(data_final.drop("Price", axis=1).columns.values, columns=["Features"])
reg_summary["Weights"] = reg.coef_
reg_summary
set(data_of_interest.Brand)     # Banchmark Brand is Audi
set(data_of_interest.Body)      # Benchmark Body is Crossover
set(data_of_interest.EngineTyp) # Benchmark EgineTyp is Diesel
set(data_of_interest.Body)      # Benchmark Registration is No

yhat_test = reg.predict(x_test)

plt.scatter(y_test, yhat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

predict_summary = pd.DataFrame(np.exp(yhat_test), columns=["Prediction"])
predict_summary["Target"] = np.exp(y_test.reset_index(drop=True)) # needs to reset indexes of test data
predict_summary["Residual"] = predict_summary["Target"] - predict_summary["Prediction"]
predict_summary["Abs_Diff_%"] = np.absolute(predict_summary["Residual"]/predict_summary["Target"])*100
predict_summary.sort_values(by=["Abs_Diff_%"], ascending=True)
predict_summary.head()
predict_summary.describe()

################################################################
# Logistic Regression Example - Client term deposit subscription
################################################################
# Model Assumptions
# >> https://www.lexjansen.com/wuss/2018/130_Final_Paper_PDF.pdf
# . Appropiate Outcome Structure
# . Observation Idependence
# . Abscence of Multicolinearity
# . Linearity between independent variables and its log-odds
# . Large Sample Size

# Statsmodels Summary Notes
# >> https://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/
# >> https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/
# . converged: Important to see if the model converged indeed
# . Covariance Type: 'nonrobust' warns about the specification of the covariance matrix
# . McFadden's Pseudo R-Squared: How much better the model performs against the LL-Null
# . Log-Likelihood: Can be negative since it's the log of a probability. The bigger the better.
# . LL-Null: Log-Likelihood of what is considered to be the worst model as possible
# . LLR p-value: {H0: LLc worse than LL-Null; H1: LLC better than LL-Null}
# . Quasi-Separation: Bad thing. States that there was almost completly seperation of the data
# into categories. In the case of complete separation, the MLE does not exists and the validty
# of the model is questionable.

# Notes
# . GOAL: An bank wanted to see if it's marketing campaing had a positive effect on
#   persuing an client to make a term deposit.
# . Target variable is 'SUBSCRIBED', which indicates the subscription of a client to a term deposit.
# . The variables 'MAY' and 'MARCH' didn't seem very clear on its meaning so we chose to drop it.
# . The removal of the constant term did not affected the overall significance of the model.
# . The 'IR' variable has a clear bimodal PDF. Meaning that data contained customers
#   which had either very low or very high interest rate. Yet it added considerably to the
#   predictive power of the model. This makes sense since if one has an high interest rate
#   it is more likely that it doesn't sign up for a term deposit.
# . The 'DR' variable clearly has some big-valued outliers. Although not specified the
#   its associated unit of measure, if one supposes it do be 'days', it's quite unlikely that
#   a marketing campaing would last 3000 days (8.2 years). Therefore, seems reaosnable to discard
#   such extreme values.
# . The model had an overall training/test accuracy of 85%. It may definetly have room for improvement,
#   since 2 other variables were disconsidered.
# . The model had pretty much consistent evaluation metrics for both training and test data sets. This
#   is actually a good thing since it proofs that there was no overfitting neither underfitting, although
#   overall accuaracy was not the best.

# Interpretation of the coefficients
# log(pi/1-pi) = exp(-0.673 - 0.7*IR + 0.006*DR + 2.92*CR + 1.93*PV)
# . Unit increase in IR reduces in 27% the odds of a client subscribing to a term deposit
# . To give credit in CR, increases in 947% the odds of a client (...)
# . Participating in previous marketing campain, increases in 354% the odds of a client (...)
# . Unit increase in DR increases increases in 51% the odds of a client (...)

filename = "D:/Python/Courses/365_Data_Science/Advanced Statistcal Methods in Python/Data/bank_data.csv"
bank_data = pd.read_csv(filename)
bank_data.shape
bank_data.head()

bank_data = bank_data.drop(["MAY","MARCH"], axis=1)
bank_data = bank_data.reset_index(drop=True)

sns.distplot(bank_data.IR, color="blue")
plt.show()
sns.distplot(bank_data.DR, color="green")
plt.show()

bank_data = bank_data[bank_data["DR"] < bank_data["DR"].quantile(0.95)]
bank_data = bank_data.reset_index(drop=True)
bank_data["SUBSCRIBED"] = bank_data["SUBSCRIBED"].map({"yes":1,"no":0})
bank_data.shape # removed 37 observations (5% loss of data)

from sklearn.model_selection import train_test_split
y_test, y_train, x_test, x_train = train_test_split(bank_data.SUBSCRIBED, bank_data.drop("SUBSCRIBED", axis=1), test_size=0.8, random_state=42)

bank_logit = sm.Logit(endog=y_train, exog=sm.add_constant(x_train)).fit()
bank_logit.summary()

from sklearn.metrics import f1_score
train_acc = pd.DataFrame(bank_logit.pred_table())
train_acc.columns = ["Predicted No", "Predicted Yes"]
train_acc = train_acc.rename(index={0:"Actual No",1:"Actual Yes"})
train_acc
print(f"Training Accuracy: {(np.sum(np.mat(train_acc).diagonal()))/np.sum(np.mat(train_acc))}")
print(f"Training Precision: {np.mat(train_acc)[1,1]/np.sum(np.mat(train_acc)[1,:])}")
print(f"Training Sensitivity: {np.mat(train_acc)[0,0]/np.sum(np.mat(train_acc)[0,:])}")
print(f"Training F1-Score: {f1_score(y_true=y_train, y_pred=np.round(bank_logit.predict(exog=sm.add_constant(x_train))))}")

sns.regplot(x=sm.add_constant(x_train).IR, y=y_train, logistic=True)
plt.show() # This plot meets our interpretation of interest rates and client subscription
sns.regplot(x=sm.add_constant(x_train).DR, y=y_train, logistic=True)
plt.show() # Marketing campaign with more duration, tends to drive the client to subscription

bank_logit_pred = bank_logit.predict(exog=sm.add_constant(x_test)) # Unfortnately, the predict() method returns a series object and thus it has no pred_table()
pred_acc = np.histogram2d(y_test, bank_logit_pred, np.array([0, 0.5, 1]))[0] # If an predicted values lies between 0.5 or 1, it rounds up to 1 otherwise zero.
print(f"Predictive Accuracy: {np.sum(np.mat(pred_acc).diagonal())/np.sum(np.mat(pred_acc))}")
print(f"Predictive Precision:{np.mat(pred_acc)[1,1]/np.sum(np.mat(pred_acc)[1,:])}")
print(f"Predictive Sensitivity: {np.mat(pred_acc)[0,0]/np.sum(np.mat(pred_acc)[0,:])}")
print(f"Predictive F1-Score: {f1_score(y_true=y_test, y_pred=np.round(bank_logit_pred))}")