from sklearn import model_selection, linear_model
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import sys
import tqdm


def extract_features(sequences):
  """
  Returns DataFrame where features are presence/absence of certain mutations
  """
  mutation_df = pd.DataFrame()
  n_bases_in_seq = len(sequences[0])

  for location in tqdm.tqdm(range(n_bases_in_seq)):
    bases_at_location = np.array([s[location] for s in sequences])
    if len(set(bases_at_location)) == 1:
      continue
    for base in ['A', 'T', 'G', 'C', '-']:
      feature_values = (bases_at_location == base)
      feature_values[bases_at_location == 'N'] = np.nan
      feature_values = feature_values*1
      column_name = str(location)+"_"+base
      mutation_df[column_name] = feature_values

  return mutation_df


def balance_data(mutation_df, sequences, countries_regions):
  """
  Balance mutation data such that the numbers of entries from each region are equal
  """
  balanced_df = mutation_df.copy()
  countries = [(s.description).split('|')[-1] for s in sequences]
  regions = [countries_regions[c] if c in countries_regions
              else 'N/A' for c in countries]
  balanced_df['region'] = regions

  df_north_america = balanced_df[balanced_df['region'] == "North America"]
  df_asia = balanced_df[balanced_df['region'] == "Asia"]
  df_oceania = balanced_df[balanced_df['region'] == "Oceania"]
  N = min(len(df_north_america), len(df_asia), len(df_oceania))

  balanced_df = pd.concat(
    [df_north_america, df_asia, df_oceania]).sample(frac=1)

  return balanced_df
    
def train(balanced_df):
  X = balanced_df.drop('region', axis=1)
  y = balanced_df['region']

  X_train, X_test, y_train, y_test = model_selection.train_test_split(
      X, y, train_size=.8, random_state=42)

  lm = linear_model.LogisticRegression(
    multi_class="multinomial",
    max_iter=100,
    fit_intercept=False,
    tol=0.001,
    solver='saga',
    random_state=42
  )

  lm.fit(X_train, y_train)

  return lm, X_test, y_test


def evaluate(lm, X_test, y_test):
  y_pred = lm.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", str(accuracy*100)+"%")

  confusion_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))
  confusion_mat.columns = [c + ' predicted' for c in lm.classes_]
  confusion_mat.index = [c + ' true' for c in lm.classes_]
  np.set_printoptions(threshold=sys.maxsize)

  print(confusion_mat.to_string())
