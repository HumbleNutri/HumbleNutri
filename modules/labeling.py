import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from modules.utils import *
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner as kt
from sklearn.preprocessing import OneHotEncoder
import argparse


class LabelingModule(keras.Model):
    def __init__(
            self, 
            input_shape, 
            output_shape,
            binary, 
            random_seed = 2025,
            batch_size = 64, 
            num_epochs = 30, 
            activation = 'relu',
            validation_split = 0.2
        ):
        super().__init__()
        self.random_seed = random_seed
        self.input_shape = input_shape
        self.binary = binary
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation
        self.validation_split = validation_split
        self.es = EarlyStopping(monitor='val_loss',patience=5)
    
    def model_builder(self, hp):
        # hyperparameters
        hp_units = hp.Int('units', min_value=128, max_value=512, step=128)
        hp_rate =hp.Float('do_rate',min_value=0.2,max_value=0.5,step=0.1)
        hp_init = hp.Choice('initializer', values=['glorot_uniform','glorot_normal']) #'he'-initializer excluded
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_wd = hp.Choice('weight_decay', values=[0.0, 1e-2, 1e-3, 1e-4])
        model = keras.Sequential()
        # Input
        model.add(tf.keras.Input(shape=(self.input_shape,)))
        # Get model architecture
        model.add(Dense(units=hp_units, activation=self.activation,kernel_initializer=hp_init))
        model.add(Dropout(hp_rate))
        model.add(Dense(units=hp_units, activation=self.activation,kernel_initializer=hp_init))
        model.add(Dropout(hp_rate))
        model.add(Dense(units=hp_units, activation=self.activation,kernel_initializer=hp_init))
        model.add(Dropout(hp_rate))
        model.add(Dense(self.output_shape, activation=classifier_type(self.binary)))
        #compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate, decay=hp_wd),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model
    
    def tune_classifiers(self, X, y, num_batch, cat):
        # Tune the model via Hyperband
        tuner = kt.Hyperband(self.model_builder,
                             objective='val_accuracy',
                             # max_epochs=10,
                             factor=3,
                             seed=self.random_seed,
                             directory=f'classifier_hpt_{cat}',
                             project_name=f'classifier_{num_batch}')
        tuner.search(X, y, epochs=self.num_epochs, validation_split=self.validation_split,
                     batch_size=self.batch_size, callbacks=[self.es], shuffle=True)
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        # Export best hp
        with open(f'classifier_hpt_{cat}/best_hp_batch{num_batch}.json', 'w') as f:
            f.write(json.dumps([{'units': best_hps.get('units'),
                                 'initializer': best_hps.get('initializer'),
                                 'do_rate': best_hps.get('do_rate'),
                                 'learning_rate': best_hps.get('learning_rate'),
                                 'weight_decay': best_hps.get('weight_decay')}]))
        
        
        return tuner, best_hps
    
    def fit_model(self, X, y, tuner, best_hps):
        # Train with optimal hyperparameters
        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        history = hypermodel.fit(X, y, epochs=self.num_epochs, validation_split=self.validation_split,
                                 batch_size=self.batch_size, callbacks=[self.es], shuffle=True)
        
        return hypermodel, history
    
def label_assigning(cls_type, pred_prob, keep_ind):
    # Binary classification label assigning
    if cls_type == 'binary':
        y_lp_label = np.concatenate([y_lp_label, np.where(pred_prob[keep_ind] > 0.5, 1,0).reshape(-1)])
    # Softmax classificaiton
    else:
        pred_labels = np.zeros_like(pred_prob[keep_ind])
        pred_labels[np.arange(len(pred_prob[keep_ind])), pred_prob[keep_ind].argmax(axis=1)] = 1
        y_lp_label = np.concatenate([y_lp_label, pred_labels])

    return y_lp_label

def add_pred(cls_type, model, X_inconf):
    # Add predicted labels for the last inconfident instances
    if cls_type == 'binary':
        pred_labels = np.where(model.predict(X_inconf) > 0.5, 1,0).reshape(-1)
    else:
        pred_prob = model.predict(X_inconf)
        pred_labels = np.zeros_like(pred_prob)
        pred_labels[np.arange(len(pred_prob)), pred_prob.argmax(axis=1)] = 1
    
    return pred_labels

def label_encoding(self, train_df):
    """
        One hot encoding for the categorical label (food category pseudo-label).

        Args:
            train_df: training dataset that needs to be one hot encoded

        Returns:
            ohe_train: one hot encoded labels
    """
    # set seed
    np.random.seed(self.random_seed)
    # Change category to categorical label: One hot encoding
    sc_tr = np.array(train_df).reshape(-1, 1)
    # sc_ts = np.array(test_df).reshape(len(test_df),1)
    # Integer label to one hot encoding dummy variable
    ohe = OneHotEncoder(sparse=False)
    ohe_train = ohe.fit_transform(sc_tr)
    # ohe_test = ohe.transform(sc_ts)

    return ohe_train


def SelfLabeling(X_unlabeled, y_unlabeled, X_lp_train, y_lp_train, category, cls_type):
    # first y-label # Latam
    if cls_type == 'binary':
        y_lp_label = np.array(y_lp_train[category])
    else:
        y_lp_label =  label_encoding(y_lp_train[category])
    # inconfident data
    X_inconf = np.empty((0, X_lp_train.shape[1]))
    y_inconf = pd.DataFrame()
    # call trainer
    trainer = LabelingModule(input_shape=X_lp_train.shape[1], output_shape=max(set(y_lp_train[category])), binary=cls_type)
    # num batches to divide the unlabeled data for incremental training
    num_incr = 10
    minidf_size = len(y_unlabeled) // num_incr
    
    # label propagation for 10 batches
    for i in tqdm(range(num_incr), desc="Processing incremental training"):
        # Tune only until first unbalanced batch # Stable validation accuracy without continuous tuning
        if i < 2:
            tuner, tuned_hp = trainer.tune_classifiers(X_lp_train, y_lp_label, num_batch = i+1, cat=category)
        # Fit model
        model, history = trainer.fit_model(X_lp_train, y_lp_label, tuner, tuned_hp)
        print(f'Batch_{i} max val_acc: ', round(max(history.history['val_accuracy']),4))
        
        # Get test data
        if i == (num_incr - 1):
            # get predicted class probability for last batch
            X_test = X_unlabeled[i*minidf_size:]
            y_test = y_unlabeled.iloc[i*minidf_size:].reset_index(drop=True)
            pred_prob = model.predict(X_test)      
        else:
            # get predicted class probability
            X_test = X_unlabeled[i*minidf_size :(i+1)*minidf_size]
            y_test = y_unlabeled.iloc[i*minidf_size :(i+1)*minidf_size].reset_index(drop=True)
            pred_prob = model.predict(X_test) 
        # Make tuple with its index location
        pred_tup = [(prob[np.argmax(prob)], ind) for prob, ind in zip(pred_prob, list(range(0, len(pred_prob))))]
        # Get confidence of the prediction, setting class_prob=0.5 being least confident prediction
        pred_conf=[(ind, abs(prob - 0.5)) for prob, ind in pred_tup]
        # Sort by confidence
        sorted_pred= sorted(pred_conf, key=lambda x: x[1], reverse=True)
        # keep 90%
        threshold_90conf = int(0.9 * len(sorted_pred))
        keep_ind = [keep_ind for keep_ind, conf in sorted_pred[:threshold_90conf]] 
        exc_ind = [exc_ind for exc_ind, conf in sorted_pred[threshold_90conf:]] 
        # Add additional data to training set
        X_lp_train = np.concatenate([X_lp_train, X_test[keep_ind]])
        y_lp_train = pd.concat([y_lp_train ,y_test.iloc[keep_ind]]).reset_index(drop=True)
        y_lp_label = label_assigning(cls_type=cls_type, pred_prob=pred_prob, keep_ind=keep_ind)
        # Set aside inconfidnet data
        X_inconf = np.concatenate([X_inconf, X_test[exc_ind]])
        y_inconf = pd.concat([y_inconf, y_test.iloc[exc_ind]]).reset_index(drop=True)
    
    # Final prediction on inconfidnet data
    model, history = trainer.fit_model(X_lp_train, y_lp_label, tuner, tuned_hp)
    print('Last Batch max val_acc: ', round(max(history.history['val_accuracy']),4))
    # Get final labels for inconfident instances
    pred_labels = add_pred(cls_type=cls_type, model=model, X_inconf=X_inconf)
    # Make full datset
    X_lp_train = np.concatenate([X_lp_train, X_inconf])
    y_lp_train = pd.concat([y_lp_train, y_inconf]).reset_index(drop=True)
    y_lp_label = np.concatenate([y_lp_label, pred_labels])
    # Assign label
    if cls_type == 'binary':
        y_lp_train[category.replace('tag', 'label')] = y_lp_label
    else:
        y_lp_train[category.replace('tag', 'label')] = np.argmax(y_lp_label, axis=1) + 1
    # return labeled dataset and sorted embeddings
    
    return X_lp_train, y_lp_train


if __name__ == "__main__":
    # # Get arguments for cls type to test on
    # parser = argparse.ArgumentParser(description="Run the script with the type of classifier you want to test on.")
    # parser.add_argument('labeltype', type=str, help="Type of label to create. ex) latam, bf, amdds, all")
    # args = parser.parse_args()
    # Read the HUMMUS reicpe dataset (post-processed)
    recipes = pd.read_csv('pp_recipes.csv',low_memory=False)
    # Read FastText embeddings
    hummus_recipeFT = np.load('hummus_RecipeFT_embedding.npy')
    # Get LatAm ground truth data
    X_unlabeled_latam, y_unlabeled_latam, X_lp_train_latam, y_lp_train_latam = set_gt(df=recipes, 
                                                                                      X=hummus_recipeFT, 
                                                                                      cls_target='latam')
    # Perform semi-supervised self-training for labeling LatAm cuisine
    X_labeled_latam, y_labeled_latam = SelfLabeling(X_unlabeled=X_unlabeled_latam, y_unlabeled=y_unlabeled_latam,
                                                    X_lp_train=X_lp_train_latam, y_lp_train=y_lp_train_latam,
                                                    category='latam_tag', cls_type='binary')
    # Save result
    # np.save('X_labeled_latam.npy', X_labeled_latam)
    y_labeled_latam.to_csv('y_labeled_latam.csv', index=False)
    
    # Get Breakfast ground truth data
    X_unlabeled_bf, y_unlabeled_bf, X_lp_train_bf, y_lp_train_bf = set_gt(df=recipes, 
                                                                          X=hummus_recipeFT, 
                                                                          cls_target='bf')
    # Perform semi-supervised self-training for labeling Breafast dishes
    X_labeled_bf, y_labeled_bf = SelfLabeling(X_unlabeled=X_unlabeled_bf, y_unlabeled=y_unlabeled_bf,
                                              X_lp_train=X_lp_train_bf, y_lp_train=y_lp_train_bf,
                                              category='bf_tag', cls_type='binary')
    # Save result
    # np.save('X_labeled_bf.npy', X_labeled_bf)
    y_labeled_bf.to_csv('y_labeled_bf.csv', index=False)

    # AMDDS (Appetizer, Main, Dessert, Drink, Sauce) classification
    app_ind = get_tag_ind(recipes, get_tags('app')) # 45222
    main_ind = get_tag_ind(recipes, get_tags('main')) # 71577
    dsrt_ind = get_tag_ind(recipes, get_tags('dsrt')) # 43000
    drnk_ind = get_tag_ind(recipes, get_tags('drnk')) # 11069
    sauce_ind = get_tag_ind(recipes, get_tags('sauce')) # 14531
    # Make AMDDS groud truth label
    recipes['amdds_tag'] = 0 #65.6% # 332635
    recipes.loc[app_ind, 'amdds_tag'] = 1 # 7.3% # 37174
    recipes.loc[main_ind, 'amdds_tag'] = 2 # 13.9% # 70526 # If the item have both app or main keep as main
    recipes.loc[dsrt_ind, 'amdds_tag'] = 3 # 8.3% # 42036 # If the item have either app, main, dsrt keep as dsrt
    recipes.loc[drnk_ind, 'amdds_tag'] = 4 # 2.2% # 11008 # If the item have either app, main, dsrt, drnk keep as drnk
    recipes.loc[sauce_ind, 'amdds_tag'] = 5 # 2.8% # 13956 # If the item have intersect with any amdds keep as sauce
    # Divide dataset using tags
    X_amdds = hummus_recipeFT[recipes['amdds_tag']!=0]
    y_amdds = recipes[recipes['amdds_tag']!=0].reset_index(drop=True) # 174700
    X_Noamdds = hummus_recipeFT[recipes['amdds_tag']==0]
    y_Noamdds = recipes[recipes['amdds_tag']==0].reset_index(drop=True) # 332635
    # Perform semi-supervised self-training for labeling AMDDS meal type
    X_labeled_amdds, y_labeled_amdds = SelfLabeling(X_unlabeled=X_Noamdds, y_unlabeled=y_Noamdds,
                                                    X_lp_train=X_amdds, y_lp_train=y_amdds,
                                                    category='amdds_tag', cls_type='multi')
    # Save result
    # np.save('X_labeled_amdds.npy', X_labeled_amdds)
    y_labeled_amdds.to_csv('y_labeled_amdds.csv', index=False)