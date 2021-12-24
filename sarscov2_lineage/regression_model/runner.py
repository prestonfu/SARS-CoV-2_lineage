from Bio import SeqIO
from sarscov2_lineage.data.countries_regions import countries_regions
from sarscov2_lineage.regression_model import main

cov2_sequences = 'sarscov2_lineage/data/SARS_CoV_2_sequences_global.fasta'
sequences = list(SeqIO.parse(cov2_sequences, 'fasta'))

mutation_df = main.extract_features(sequences)
balanced_df = main.balance_data(mutation_df, sequences, countries_regions)
lm, X_test, Y_test = main.train(balanced_df)
main.evaluate(lm, X_test, Y_test)
