import oraculo
import sample_selection
import utils
"""
O = oraculo.Oracle(multiple=True, data_path='../datasets/balancedAllMutants.csv',
                   data_sep=';', index_col=0, golden_label='original',
                   training_cols=['side1', 'side2', 'side3'], train_size=0.8,
                   train=False, models_path='../code/models/')


MT = sample_selection.MutantReport(O)

random_kills, segmented_kills, random_n_to_killed, segmented_n_to_killed, plot1, plot2 = \
MT.compare_random_vs_segmented(n_iters_random=200, per_cent=0.9, also_jumps=False)"""