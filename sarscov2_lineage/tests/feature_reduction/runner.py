from Bio import SeqIO
from sarscov2_lineage.data.countries_regions import countries_regions
from sarscov2_lineage.src import logistic_regression, feature_reduction

cov2_sequences = 'sarscov2_lineage/data/SARS_CoV_2_sequences_global.fasta'
sequences = list(SeqIO.parse(cov2_sequences, 'fasta'))

print("Logistic regression")
mutation_df = logistic_regression.extract_features(sequences)
X, y = logistic_regression.balance_data(
    mutation_df, sequences, countries_regions)
lm, X_train, X_test, y_train, y_test = logistic_regression.train(X, y)

print("Feature reduction")
feature_reduction.features_used(lm)
feature_reduction.accuracy_compare(lm, X_train, X_test, y_train, y_test)
feature_reduction.train_cv(X_train, X_test, y_train, y_test)
