import pandas as pd
import seaborn as sns
import utils
import matplotlib.pyplot as plt
import time
from sklearn.metrics import auc

class MutantReport():
    """
    Class that makes reports from the mutant analysis of an Oracle object. Should be run after an oracle is loaded.
    """

    def __init__(self, Oracle):
        """
        Params:
            Oracle (Oracle): an Oracle object from the oraculo module.
        """

        self.Oracle = Oracle
        self.Y = pd.Series(utils.undo_dummy(self.Oracle.y_test, self.Oracle.label_encoder))
        self.Y.index = self.Oracle.X_test.index 
        self.data = pd.DataFrame([self.Oracle.MN_predictions_output, self.Oracle.MN_predictions_prob], columns=self.Oracle.X_test.index, index=['Y', 'prob']).T
        self.compute_mutants_killed()
        self.compute_killable_mutants()
    
    def _compare_columns(self, mut_col, original):
        """
        internal function used to compare whether two columns are different. It's used to compute whether a mutant is killed.
            Params:
                mut_col (pd.Series): target mutant column.
                original (pd.Series): column containing the true label.
            Returns:
                pd.Series: a series of booleans resulting from the comparison.
        """
        return mut_col!=original 
    
    def compute_mutants_killed(self):
        """
        Computes a dataframe containing whether each mutant has been killed and the number of killed mutants.
        """
        self.mutants_killed_df = self.Oracle.mutants_test.apply(self._compare_columns, axis=0, original=self.Y)
        self.mutants_killed = self.mutants_killed_df.sum(1)
        self.data['mutants_killed'] = self.mutants_killed
        

    def AccumuledMutantsKilled(self, sort_column=False, random=False, segmented=False, jumps=False,per_cent=1):
        """
        Computes the accumulated mutants killed.
            Params:
            sort_column (str): the column to use to sort the rows. If False the rows are not sorted.
            random (bool): whether to randomly sort the rows.
            segmented (bool): whether to use a segmented model. It picks 10 samples, the first from each 10th 
                              percentile of the probability assigned by the networks to each prediction.
            jumps (bool): whether to use the "jumps" model. It tries to pick the most different samples by means
                          of picking two sample for each class, one correctly predicted and one that has been not.
                          Both of them are picked with the highest network probability and lowest respectively.
            per_cent (float): percentage to consider all samples killed.
            Returns:
                killed (list): number of mutants killed at each iteration.
                samples_until_all_killed (int): number of samples until all mutants are killed.
        """
        tmp = self.mutants_killed_df

        if sort_column and sort_column in self.data.columns and not random:
            print('sorting by', sort_column)
            tmp = tmp.loc[self.data.sort_values(by=sort_column).index]

        elif random:
            tmp = tmp.sample(frac=1)

        elif segmented:
            tmp = tmp.loc[self.segment_probs(self.data['prob'])]
        
        elif jumps:
            tmp = tmp.loc[self.segments_by_jumping()]

        killed = self.calculate_acc_mutants_killed(tmp)
        samples_until_all_killed = self.compute_samples_until_all_killed(killed, self.killable_mutants, per_cent=per_cent)
        #print(f'Time to kill {per_cent*100}% of mutants: {samples_until_all_killed}. Random: {random}')
        return killed, samples_until_all_killed

    
    def calculate_acc_mutants_killed(self, df):
        """
        Internal function used to calculate the number of mutants killed at each iteration.
        Params:
            df (pd.DataFrame): the dataframe containing the necessary information.
        Returns:
            unique_kills_count(list): the number of mutants killed at each iteration.
        """
        unique_kills = set([])
        unique_kills_count = []
        
        for row in df.iterrows():
            unique_kills = unique_kills.union(list(row[1][row[1] == True].index))
            unique_kills_count.append(len(unique_kills))
        
        return unique_kills_count

    
    def segment_probs(self, series):
        """
        Function used to segment a series by each 10th percentile.
        Params:
            series (pd.Series): a series with numerical values.
        Returns:
            indexes (list): a list containing the indexes of the segmented data.
        """

        indexes = []
        tmp = series.sort_values(ascending=False)
        for q in range(0, 101, 10):
            idx = (tmp<=series.quantile(q/100)).idxmax()
            indexes.append(idx)
            #print(q/100, idx, tmp.loc[idx])
        return indexes

    def segments_by_jumping(self):
        """
        Internal Function use to segment "by jumps". 
        It tries to pick the most different samples by means of picking two sample for each class, one correctly 
        predicted and one that has been not.Both of them are picked with the highest network probability and 
        lowest respectively.
        
        Returns:
            seeds1.tolist() (list): the indexes necessary the retrieve half the data of this method.
            seeds2.tolist() (list): the indexes necessary the retrieve half the data of this method.
        """
        a= self.data.copy()
        a = a.join(self.Oracle.data[['original']])
        a['correct'] = a['original'] == a['Y']
        seeds1 = a.reset_index().sort_values(by='prob').groupby(['correct', 'original']).first()['seed'].values
        seeds2 = a.reset_index().sort_values(by='prob').groupby(['correct', 'original']).last()['seed'].values

        return seeds1.tolist() + seeds2.tolist() 

    def compute_samples_until_all_killed(self, iterable, n_mutants, per_cent=1):
        """
        Computes te number of samples necessary to check until all mutants are killed.
        Params:
            iterable (list) an interable containign the number of mutants killed at each step.
            n_mutants (n_mutants): total number of mutants.
            per_cent (float): percentage to consider all samples killed.
        
        Returns:
            compare.idxmax()+1 : the number of samples that have been necessary to kill all the 
            mutants. It reutns 0 if it is not possible. 
        """
        tmp = pd.Series(iterable)
        thrs = round(n_mutants*per_cent)
        compare = tmp>=thrs
        if (compare==False).all():
            return 0
        else:
            return compare.idxmax() + 1

    def compute_killable_mutants(self):
        """
        Computes the number of mutants that are possbile to kill.
        """
        self.killable_mutants = (~(self.mutants_killed_df==False).all()).sum()

    def compare_random_vs_segmented(self, n_iters_random, per_cent=0.9, also_jumps=False):
        """
        Performs a comparision of a random and a semgented model.

        Params:
            n_iters_random: how many random experiments to generate. The greater the more accurate but slower.
            per_cent (float): number between 0 and 1 to represent the percentage of mutants killed as a measure
                              of performance. Default is 0.9.
            also_jumps (bool): if True it performs a special segmented method in which it doesn't use percentiles
                               but tries to take the most extreme samples by means of how they are predicted and 
                               the true labels. 
        Returns:
            random_kills (dict): a dictionary that contains the number of mutants that each iteartion kills by
                                 random.
            segmented_kills (dict): a dictionary that contains the number of mutants that each iteartion kills by
                                 the segmented method used.
            random_n_to_killed (int): number of samples until all mutants are killed with the random method.
            segmented_n_to_killed (int): number of samples until all mutants are killed with the segmented method.
            plot1 (plot): plot path to the line plot comparision between random and segmented.
            plot2 (plot): plot path comparing the random distribution vs the segmented.

        """
        if also_jumps:
            segmented_kills, segmented_n_to_killed = self.AccumuledMutantsKilled(jumps=True, per_cent=per_cent)
        else:
            segmented_kills, segmented_n_to_killed = self.AccumuledMutantsKilled(segmented=True, per_cent=per_cent)
        
        random_kills = {}
        random_n_to_killed = {}

        for i in range(n_iters_random):
            random_kills[i+1], random_n_to_killed[i+1] = self.AccumuledMutantsKilled(random=True, per_cent=per_cent)

        segmented_kills = pd.DataFrame(pd.Series(segmented_kills)).reset_index()
        segmented_kills['index'] += 1
        segmented_kills['label'] = 'segmented'
        segmented_kills = segmented_kills.rename(columns={'index':'variable', 0:'value'})


        random_kills=pd.DataFrame(random_kills).iloc[0:len(segmented_kills)].T.melt()
        random_kills['label'] = 'random'
        random_kills['variable'] += 1

        #data = pd.concat([segmented_kills, random_kills],0)

        plot1 = self.generate_comparision_line_plot_with_steps(random_kills, segmented_kills)
        plot2 = self.generate_comparision_histogram(list(random_n_to_killed.values()), segmented_n_to_killed)
        
        return random_kills, segmented_kills, random_n_to_killed, segmented_n_to_killed, plot1, plot2

    def generate_comparision_line_plot(self, data):
        """
        (DEPRECATED, ONLY FOR COMPATIBILITY) 
        Generates a line plot to compare the random and the segmented model.
        Params:
            data (pd.DataFrame): dataframe containing the data to make the plot.
        Returns:
            plot_path (str): path where the plot is stored.
        """
        sns.lineplot(data=data, x='variable', y='value', hue='label')
        plt.title('Random vs Segmented sample selection')
        plt.ylabel('Mutants killed')
        plt.xlabel('Samples checked')
        plot_path = f'./plots/comparision_pipeline__{time.strftime("%D__%H_%M_%S").replace("/","_")}.png'
        plt.savefig(plot_path)
        plt.plot()
        plt.close('all')
        return plot_path

    def generate_comparision_line_plot_with_steps(self, data_random, data_model):
        """
        Generates a line plot to compare the random and the segmented model.
        Params:
            data_random (pd.DataFrame): dataframe containing the random data to make the plot.
            data_model (pd.DataFrame): dataframe containing the segmented data to make the plot.
        Returns:
            plot_path (str): path where the plot is stored.
        """
        data_random['value'] = data_random['value']/self.killable_mutants*100
        data_model['value'] = data_model['value']/self.killable_mutants*100
        sns.lineplot(data=data_random, x='variable', y='value', label='random')
        sns.lineplot(data=data_model, x='variable', y='value', drawstyle='steps-post', label='segmented')
        auc_model = auc(data_model['variable'], data_model['value'])
        tmp_data_random = data_random.groupby('variable')['value'].mean().reset_index()
        auc_random = auc(tmp_data_random['variable'], tmp_data_random['value'])
        auc_ratio = (auc_model/auc_random).round(3)
        plt.title(f'Random vs Segmented mutant score. AUC ratio: {auc_ratio}')
        plt.ylabel('Mutant Score')
        plt.xlabel('Samples checked')
        plot_path = f'./plots/comparision_pipeline__{time.strftime("%D__%H_%M_%S").replace("/","_")}.png'
        plt.savefig(plot_path)
        plt.plot()
        plt.close('all')

        return plot_path

    def generate_comparision_histogram(self, list_of_random_n_to_killed, model_n_to_killed):
        """
        Generates a line plot to compare the random and the segmented model.
        Params:
            list_of_random_n_to_killed (list): list containing the different lists of mutants killed 
                                               at each iteration of the random process.
            model_n_to_killed (int): number of necessary samples to kill all the mutants for the 
                                     segmented model.
        Returns:
            plot_path (str): path where the plot is stored.
        """
        sns.distplot(list_of_random_n_to_killed, hist=False)                                                                                                                                                                                                                           
        plt.axvline(model_n_to_killed)                                                                                                                                                                                                                                          
        plt.ylim((0,0.25))    
        plt.title('Necessary samples to kill 90% of mutants. Rand dist vs segmented')
        plot_path=f'./plots/comparision_pipeline_hist__{time.strftime("%D__%H_%M_%S").replace("/","_")}.png'                                                                                                                                                                                                                                        
        plt.savefig(plot_path)                                                                                                                                                                                                                                         
        plt.close('all')
        return plot_path                          