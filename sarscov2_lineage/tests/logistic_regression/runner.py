from Bio import SeqIO
from sarscov2_lineage.data.countries_regions import countries_regions
from sarscov2_lineage.src import logistic_regression

cov2_sequences = 'sarscov2_lineage/data/SARS_CoV_2_sequences_global.fasta'
sequences = list(SeqIO.parse(cov2_sequences, 'fasta'))

print("Extracting features")
mutation_df = logistic_regression.extract_features(sequences)

print("Balancing data")
X, y = logistic_regression.balance_data(
    mutation_df, sequences, countries_regions)

print("Training model")
lm, _, X_test, _, Y_test = logistic_regression.train(X, y)

print("Evaluating model")
logistic_regression.evaluate(lm, X_test, Y_test)
