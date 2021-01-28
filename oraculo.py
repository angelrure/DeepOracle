import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import utils
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

logger = logging.getLogger()

class Oracle():
    """
    General Oracle class. Use it to load or generate oracle models. See the __init__method for the parameters.
    """

    def __init__(self, multiple, data_path, data_sep, index_col, golden_label, training_cols, train_size=0.8, train=True, models_path='./models',
                random_seed=1, output_prob_name='triangle_probs.csv', load_best_model=False, train_verbose=False, to_dummies=False):
        """
        Params:
            multiple (bool): whether to use a multiple network (MN) oracle. If Fals it uses a single network (SN) oracle.
            data_path (str): path to the input data file.
            data_sep (str): separator in the input data file to propertly separate the columns.
            index_col (int): index of the column to use as index (unique identifiers) of the samples.
            golden_label (str): name of the column containing the real labels.
            training_cols (list): names of the columns to use as explicative variables.
            train_size (float): proportion of the samples to use as training set.
            train (bool): whether to train the oracle. If False, the best model is searched in the models_path path.
            models_path (str): path to the folder where the models are (to be) stored.
            random_seed (int): random_seed to use for the training set generation. If 0 or False the seed is omited.
            output_prob_name (str): name to be given to the output predictions.
            load_best_model (bool): Whether to check in the models folder for a better model. If False, the trained model
                                    is used always.
            train_verbose (bool): if True, the logs of the training process are printed. They take a lot of space.
            to_dummies (list): list of variables from the training_cols to convert tu dummy. Use it when the data is non-numerical.
        
        Returns:
            Oracle: an oracle object. Usally to be fed to a MutantReport object.
        """

        self.multiple = multiple
        self.models_path = models_path
        self.random_seed = random_seed
        self.out_name = output_prob_name
        self.use_best_model = load_best_model
        self.train_verbose = train_verbose

        self.data = pd.read_table(data_path, sep=data_sep, index_col=index_col)
        if to_dummies:
            self.data = pd.get_dummies(self.data, columns=to_dummies)
            training_cols = [col for col in self.data.columns if col.startswith(tuple(training_cols))]

        Y = self.encode_label(golden_label)

        self.data_preprocess(training_cols, golden_label, Y, train_size)

        if train:
            if self.multiple:
                self.train_multiple_N_oracle()
                self.predict_using_MN_model()
                self.MN_oracle_output()

            else:
                self.train_single_N_oracle()
                self.SN_oracle_output()
        
        if self.use_best_model:
            self.load_best_model()
        else:
            if not train:
                logger.warning('Parameter train set to False and load best model set to False. Change one to True')
                return 

        if self.multiple:
            self.predict_using_MN_model()
            self.MN_oracle_output()
        else:
            self.SN_oracle_output()
        
        self.report()

    def encode_label(self, golden_label):
        """
        Function used to encode the label into dummies to use for the models.
        It also creates a label encoder and a onehot (dummy encoder) for the oracle to use when necessary.

        Params:
            golden_label: the target column to encode. Usually the target variable.

        Returns:
            pd.DataFrame: a dataframe containing the encoded variable.∫
        """
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(self.data[golden_label])
        onehot_encoder = OneHotEncoder()
        Y = onehot_encoder.fit_transform(Y.reshape(Y.shape[0],1))
        
        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder

        return Y

    def data_preprocess(self, training_cols, golden_label ,Y, train_size=0.8):
        """
        Function that preprocesses the data by doing a train and test split.

        Params:
            training_cols (list): a list with the names of the columns to use to train the data.
            golden_label (str): the name of the target column.
            Y (pd.Series): a Series containing the golden_label data.
            train_size (float): the percentage of the data to be used during training.
        """
        if not self.random_seed:
            random_seed = None
        else:
            random_seed = self.random_seed
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[training_cols],\
                                                                                Y, train_size=train_size,
                                                                                random_state=random_seed)

        self.mutants_train = self.data.drop(training_cols+[golden_label],1).loc[self.X_train.index]     
        self.mutants_test = self.data.drop(training_cols+[golden_label],1).loc[self.X_test.index]

    def train_single_N_oracle(self):
        """
        Function that creates the oracle model for with a single neural network.
        """

        self.SN_model = Sequential()
        self.SN_model.add(Dense(units=32, activation='relu', input_dim=self.X_train.shape[1]))
        self.SN_model.add(Dense(units=16, activation='relu'))
        self.SN_model.add(Dense(units=8, activation='relu'))
        self.SN_model.add(Dense(units=6, activation='relu'))
        self.SN_model.add(Dense(units=4, activation='softmax'))

        self.SN_model.compile(loss='categorical_crossentropy',
                                   optimizer='adam',
                                   metrics=['accuracy'])
        
        model_path = f'{self.models_path}/triangle_SN_{time.strftime("%D__%H_%M_%S").replace("/","_")}'
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=self.train_verbose, save_best_only=True, mode='max')

        self.SN_model.fit(self.X_train, self.y_train, epochs=10000, batch_size=80, verbose=self.train_verbose, 
                          validation_split=0.1, use_multiprocessing=False, 
                          callbacks=[checkpoint])

        self.SN_model.model.load_weights(model_path)

        #print(self.SN_model.evaluate(self.X_train, self.y_train))
        #print(self.SN_model.evaluate(self.X_test, self.y_test))

    def train_multiple_N_oracle(self):
        """
        Function that manages the creation of several networks in order to build a multi network oracle.
        """
        logger.info('Training several networks. This make take a while.')
        train_y = pd.DataFrame(self.y_train.toarray())
        test_y = pd.DataFrame(self.y_test.toarray())
        
        self.MN_models = {}

        for column in train_y.columns:
            logger.info(f'Training network to predict label: {self.label_encoder.inverse_transform([column])[0]}')
            self.MN_models[column] = self.train_individual_NN_from_MN(train_y[column], test_y[column])

        logger.info('Finished training MN oracle')

    def train_individual_NN_from_MN(self, y_train, y_test):
        """
        Function that train a single neural network that will be part of a multi network oracle.
        Params:
            y_train (list) an iterable containing the labels for the train set.
            y_test (list)  an iterable containing the labels for the test set.

        Returns:
            keras.models.Sequential: a trained keras model.
        """
        model = Sequential()
        model.add(Dense(units=64, activation='sigmoid', input_dim=self.X_train.shape[1]))
        model.add(Dense(units=32, activation='sigmoid'))
        model.add(Dense(units=16, activation='sigmoid'))
        model.add(Dense(units=8, activation='sigmoid'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        
        model_path = f'{self.models_path}/triangle_MN_{y_train.name}_{time.strftime("%D__%H_%M_%S").replace("/","_")}'
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=self.train_verbose, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5000, verbose=self.train_verbose)

        model.fit(self.X_train, y_train, epochs=100000, batch_size=80, verbose=self.train_verbose, 
                        validation_split=0.25, use_multiprocessing=False, 
                        callbacks=[checkpoint, early_stopping])

        model.model.load_weights(model_path)
        #print(model.evaluate(self.X_train, y_train))
        #print(model.evaluate(self.X_test, y_test))

        return model

    def predict_using_MN_model(self):
        """
        A funcition used to predict using a multi network oracle. 
        """
        self.MN_predictions = {}

        for col in self.MN_models.keys():
            self.MN_predictions[col] = self.MN_models[col].predict(self.X_test).flatten()

    def SN_oracle_output(self):
        """
        Function that generates the outputs of the predictions of a single network oracle. It
        stores them for later evaluation.
        """
        predictions = pd.DataFrame(self.SN_model.predict(self.X_test))
        self.SN_predictions_output = predictions.idxmax(1)
        self.SN_predictions_output = self.label_encoder.inverse_transform(self.SN_predictions_output)
        self.SN_predictions_prob = predictions.max()

    def MN_oracle_output(self):
        """
        Function that generates the outputs of the predictions of a multi network oracle. It
        stores them for later evaluation.
        """
        predictions = pd.DataFrame(self.MN_predictions)
        self.MN_predictions_output = predictions.idxmax(1)
        self.MN_predictions_output = self.label_encoder.inverse_transform(self.MN_predictions_output)
        self.MN_predictions_prob = utils.extract_max_certainity(self.MN_predictions)

        if self.out_name:
            logger.info(f'Saving model probabilities output in: {os.path.join(os.getcwd(), self.out_name)}')
            probs = pd.DataFrame(self.MN_predictions)
            probs.columns = self.label_encoder.inverse_transform(probs.columns)
            probs.index = self.X_test.index
            probs.to_csv(self.out_name)                

    def report(self):
        """
        Function that geneartes the metrics report about the model's performance.
        """
        real_test = utils.undo_dummy(self.y_test, self.label_encoder)
        if self.multiple:
            print(classification_report(real_test, self.MN_predictions_output))
        else:
            print(classification_report(real_test, self.SN_predictions_output))    


    def mutant_analysis(self):
        """
        Performs an analysis of the oracle based on whether the prediction was True/False and
        Negative/Positive. See utils.run_four_analysis for more details.
        """
        self.mutant_analysis_results = {}
        self.mutant_analysis_summary = {}
        sut_output = utils.undo_dummy(self.y_test, self.label_encoder)
        oracle_output = self.MN_predictions_output

        for mutant in self.mutants_test.columns:
            self.mutant_analysis_results[mutant], self.mutant_analysis_summary[mutant] = \
                utils.run_four_analysis(oracle_output, sut_output, self.mutants_test[mutant])
    
    def load_best_model(self):
        """
        Loads the best model from previously generated models.
        """
        if self.multiple:
            self._load_best_multiple_model()
        else:
            best_single_model = self._get_best_single_model()
            print(f'Found best model: {best_single_model}')
            self.SN_model = keras.models.load_model(best_single_model)


    def _load_best_multiple_model(self):
        """
        Internal function used to load the best model when using a multi network oracle, since there is 
        on best possible model for each class of the target variable.
        """
        self.MN_models = {}
        for i in range(self.y_train.toarray().shape[1]):
            self.MN_models[i] = keras.models.load_model(self._get_best_single_model(f'_MN_{i}', i))

    def _get_best_single_model(self, pattern='_SN_', i='all'):
        """
        Searches the best model which file follows a pattern. It is used to search for the best model for a specific 
        kind of network and target variable (if using a multi network oracle)
        Params:
            pattern (str): only uses models with the pattern in the name. Internally used to differentiate single 
                           networks (_SN_) than multin etowkrs (_MN_)
            i (str): used to differentiate between the different models of a same multi network oracle.
        
        Returns:
            path to the best model.
        """
        tested_models = {}
        for model_file in os.listdir(self.models_path):
            if pattern in model_file:
                model = keras.models.load_model(os.path.join(self.models_path, model_file))
                if i=='all':
                    tested_models[model_file] = model.evaluate(self.X_test, self.y_test)[1]
                else:
                    tested_models[model_file] = model.evaluate(self.X_test, self.y_test[:, i])[1]

        best = pd.Series(tested_models).idxmax()
        return os.path.join(self.models_path, best)

### TODO: try with individual networks to take the samples with most disagreements.