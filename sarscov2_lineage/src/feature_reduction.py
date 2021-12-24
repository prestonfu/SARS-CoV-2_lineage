from sklearn import linear_model
import numpy as np


def features_used(lm):
  coefficients = lm.coef_[0]
  n_possible_features = len(coefficients)
  n_features_used = sum(coefficients != 0)
  print("The original logistic regression model used {used} out of a possible {possible} features"
        .format(used=n_features_used, possible=n_possible_features))


def accuracy_compare(lm, X_train, X_test, y_train, y_test):
  y_pred_train = lm.predict(X_train)
  y_pred_test = lm.predict(X_test)

  training_accuracy = 100*np.mean(y_train == y_pred_train)
  testing_accuracy = 100*np.mean(y_test == y_pred_test)
  print("Training accuracy:", str(training_accuracy)+"%")
  print("Testing accuracy:", str(testing_accuracy)+"%")


def train_cv(X_train, X_test, y_train, y_test):
  lm_cv = linear_model.LogisticRegressionCV(
      multi_class="multinomial",
      max_iter=1000,
      fit_intercept=False,
      tol=0.001,
      solver='saga',
      random_state=42,
      Cs=5,
      penalty='l1'
  )
  lm_cv.fit(X_train, y_train)

  print("Training accuracy:", str(
      100*np.mean(y_train == lm_cv.predict(X_train))+"%"))
  print("Testing accuracy:", str(100*np.mean(y_test == lm_cv.predict(X_test))+"%"))
  print("Number of non-zero coefficients in lasso model:",
        sum(lm_cv.coef_[0] != 0))
  print("Lambda decided on by cross validation:", 1/lm_cv.C_[0])
