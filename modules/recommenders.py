import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch
from modules.utils import *
import argparse
import cornac
# Recommenders
from recommenders.datasets.python_splitters import python_random_split #, python_stratified_split
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, recall_at_k #, get_top_k_items
from recommenders.utils.constants import SEED
# Light GCN
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.models.deeprec.deeprec_utils import prepare_hparams


class RecommenderModels(cornac.models):
    def __init__(
            self,
            interactions,
            random_seed = 2025, 
            top_k = 10,
            # BPR
            num_epochs_bpr = 1000,
            num_factors = 1000,
            lr_bpr = 0.01,
            lambda_bpr = 0.001,
            # bivae hyperparameters
            latent_dim = 50,
            encoder_dims = [100],
            activation = 'tanh',
            likelihood = 'pois',
            num_epochs = 100,
            batch_size_bivae = 64,
            lr = 0.001,
            # lightgcn
            batch_size_gcn = 256,
            n_layers = 10,
            eval_epochs = 5
        ):
        super().__init__()
        self.interactions = interactions
        self.random_seed = random_seed
        self.top_k = top_k
        self.num_epochs_bpr = num_epochs_bpr
        self.num_factors = num_factors
        self.lr_bpr = lr_bpr
        self.lambda_bpr = lambda_bpr
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.activation = activation
        self.likelihood = likelihood
        self.num_epochs = num_epochs
        self.batch_size_bivae = batch_size_bivae
        self.lr = lr
        self.batch_size_gcn = batch_size_gcn
        self.n_layers = n_layers
        self.eval_epochs = eval_epochs

        # Get data # Divide train-test
        self.train, self.test = python_random_split(self.interactions, 0.8)
        # Training set
        self.train_set = cornac.data.Dataset.from_uir(self.train.itertuples(index=False), seed=SEED)


    def bpr(self):
        """
            Train collaborative filtering recommender model: Bayesian Personalized Ranking (BPR)

            Returns:
                predictions_bpr
        """
        #BPR
        bpr = cornac.models.BPR(
            k=self.num_factors,
            max_iter=self.num_epochs_bpr,
            learning_rate=self.lr_bpr,
            lambda_reg=self.lambda_bpr,
            verbose=True,
            seed=SEED
        )
        # Train
        with Timer() as t:
            bpr.fit(self.train_set)
        print("Took {} seconds for training.".format(t))
        # Predict
        with Timer() as t:
            predictions_bpr = predict_ranking(bpr, self.train, usercol='userID', itemcol='itemID', remove_seen=True)
        print("Took {} seconds for prediction.".format(t))

        return predictions_bpr

    def bivae(self):
        """
            Train collaborative filtering recommender model: Bilateral Variational Autoencoder (BiVAE)

            Returns:
                predictions_bivae
        """
        # BiVAE
        bivae = cornac.models.BiVAECF(
            k=self.latent_dim,
            encoder_structure=self.encoder_dims,
            act_fn=self.activation,
            likelihood=self.activation,
            n_epochs=self.num_epochs,
            batch_size=self.batch_size_bivae,
            learning_rate=self.lr,
            seed=SEED,
            use_gpu=torch.cuda.is_available(),
            verbose=True
        )
        # Train
        with Timer() as t:
            bivae.fit(self.train_set)
        print("Took {} seconds for training.".format(t))
        # Predict
        with Timer() as t:
            predictions_bivae = predict_ranking(bivae, self.train, usercol='userID', itemcol='itemID', remove_seen=True)
        print("Took {} seconds for prediction.".format(t))

        return predictions_bivae
    
    def lightgcn(self):
        """
            Train collaborative filtering recommender model: LightGCN (Graph Convolutional Netoworks)

            Returns:
                predictions_lightgcn
        """
        # Set data
        data = ImplicitCF(train=self.train, test=self.test, seed=SEED)
        # Set hyperparameters
        hparams = prepare_hparams('lightgcn.yaml',
                                  n_layers=self.n_layers,
                                  batch_size=self.batch_size_gcn,
                                  epochs=self.num_epochs,
                                  learning_rate=self.lr,
                                  eval_epoch=self.eval_epochs,
                                  top_k=self.top_k,
                                  )
        # LightGCN
        model = LightGCN(hparams, data, seed=SEED)
        # Train
        with Timer() as t:
            model.fit()
        print("Took {} seconds for training.".format(t.interval))
        # Predict
        predictions_ligthgcn = model.recommend_k_items(self.test, top_k=self.top_k, remove_seen=True)

        return predictions_ligthgcn
    
    def eval(self, pred, model, gcn):
        # Set seed
        np.random.seed(self.random_seed)
        # Evaluate prediction result based on multiple offline metrics
        if gcn:
            eval_map = map_at_k(self.test, pred, k=self.top_k)
            eval_ndcg = ndcg_at_k(self.test, pred, k=self.top_k)
            eval_recall = recall_at_k(self.test, pred, k=self.top_k)
        else:
            eval_map = map_at_k(self.test, pred, col_prediction='prediction', k=self.top_k)
            eval_ndcg = ndcg_at_k(self.test, pred, col_prediction='prediction', k=self.top_k)
            eval_recall = recall_at_k(self.test, pred, col_prediction='prediction', k=self.top_k)
        # Print
        print("MAP:\t%f" % eval_map,
              "NDCG:\t%f" % eval_ndcg,
              "Recall@K:\t%f" % eval_recall, sep='\n')
        # Export to  text file
        with open(f"{model}_evaluation_results.txt", "w") as f:
            f.write("MAP:\t%f\n" % eval_map)
            f.write("NDCG:\t%f\n" % eval_ndcg)
            f.write("Recall@K:\t%f\n" % eval_recall)



if __name__ == "__main__":
    # # Get arguments for recommenders to test on
    # parser = argparse.ArgumentParser(description="Run the script with the type of recommenders you want to test on.")
    # parser.add_argument('rectype', type=str, help="Type of label to create. ex) bf, side, latam_main, all")
    # args = parser.parse_args()
    # Read the labeled HUMMUS reicpe dataset
    y_labeled = pd.read_csv('recipes_labeled.csv',low_memory=False)
    # Read user-item interactions
    reviews = pd.read_csv('pp_reviews.csv',low_memory=False) # 1.9M
    # Remove authorship review
    reviews=reviews[reviews['rating']!=6].reset_index(drop=True) # 1.4M
    # Get labeled data 
    # Breakfast
    bf = y_labeled[y_labeled['bf_label']==1].reset_index(drop=True)
    nbf = y_labeled[y_labeled['bf_label']==0].reset_index(drop=True)
    # Remove duplicates
    bf = duplicate_recipes(bf)
    nbf = duplicate_recipes(nbf)
    # Side dish (App labels)
    nbf_app = nbf[nbf['amdds_label']==1].reset_index(drop=True)
    # LatAm Main dish
    latam_nbf = nbf[nbf['latam_label']==1].reset_index(drop=True)
    latam_nbf_main = latam_nbf[latam_nbf['amdds_label']==2].reset_index(drop=True)

    # Initial filtering via nutri score
    bf = initial_filter(bf)
    nbf_app = initial_filter(nbf_app)
    latam_nbf_main = initial_filter(latam_nbf_main)
    # Get interactions
    bf_interactions = get_reviews(bf, reviews)
    side_interactions = get_reviews(nbf_app, reviews)
    latam_main_interactions = get_reviews(latam_nbf_main, reviews)

    # Recommenders
    for name, meals in zip(['bf', 'side', 'latam_main'], [bf_interactions, side_interactions, latam_main_interactions]):
        rec = RecommenderModels(meals)
        # BPR
        bpr_pred = rec.bpr()
        rec.eval(bpr_pred, f'BPR_{name}', gcn=False)
        # BiVAE
        bivae_pred = rec.bivae()
        rec.eval(bivae_pred, f'BiVAE_{name}', gcn=False)
        # LightGCN
        gcn_pred = rec.lightgcn()
        rec.eval(gcn_pred, f'LightGCN_{name}', gcn=True)
        # Save BiVAE result based on the experiment result
        bivae_pred.to_pickle(f'{name}_predictions.pkl') 