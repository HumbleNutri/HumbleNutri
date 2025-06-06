import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def get_tag_ind(df, tags_lst):
    tag_ind=[]
    for i in tqdm(range(len(df))):
        if pd.isna(df['tags'][i]):
            continue
        else:
            for tags in tags_lst:
                if tags in df['tags'][i]:
                    tag_ind.append(i)
                    continue
    tag_ind=list(set(tag_ind))
    
    return tag_ind

def classifier_type(binary):
    activation = {
        'multi': 'softmax'
    }.get(binary, 'sigmoid')
    
    return activation

def get_tags(tag_type):
    # Tags related to Latin American cuisine that can be found in the source HUMMUS data
    latam_tags = [
        'argentine', 'chilean',  'ecuadorean', 'puerto-rican', 'peruvian', 'honduran', 'caribbean', 'brazilian',
        'central-american', 'south-american', 'mexican', 'guatemalan', 'colombian', 'oaxacan', 'cuban', 'venezuelan',
        'baja', 'costa-rican', 'tex-mex', 'cinco-de-mayo', 'salsas'
    ] # default
    # Breakfast tag
    breakfast_tags = ['breakfast']
    # AMDDS (Appetizer, Main-dish, Dessert, Drink, Sauce(condiments)) label tags
    appetizer_tags = ['appetizers','side-dishes']
    main_tags = ['main-dish']
    dessert_tags = ['desserts']
    drink_tags = ['beverages']
    sauce_tags = ['condiments-etc','salad-dressings']
    tags = {
        'latam': latam_tags,
        'bf': breakfast_tags,
        'app': appetizer_tags,
        'main': main_tags,
        'dsrt': dessert_tags,
        'drnk': drink_tags,
        'sauce': sauce_tags
    }.get(tag_type, latam_tags)

    return tags

def set_gt(df, X, cls_target):
    random.seed(2025)
    # Get target indices
    target_ind = get_tag_ind(df, get_tags(cls_target))
    # Make ground truth label
    df[f'{cls_target}_tag'] = 0
    df.loc[target_ind, f'{cls_target}_tag'] = 1 
    # Divide dataset using tags
    X_target = X[df[f'{cls_target}_tag']==1]
    y_target = df[df[f'{cls_target}_tag']==1].reset_index(drop=True)
    X_Notarget = X[df[f'{cls_target}_tag']==0]
    y_Notarget = df[df[f'{cls_target}_tag']==0].reset_index(drop=True)
    # Sample non target dataset, majority class
    # For first set of training data, take the recipes that have tags: more confident not latin american
    notarget_ind_wtags = list(y_Notarget.dropna(subset=['tags']).index)
    random_indices = random.sample(notarget_ind_wtags, len(X_target))
    # Sampled data that will be used as 1st traing dataset
    X_Notarget_matched = X_Notarget[random_indices]
    y_Notarget_matched = y_Notarget.iloc[random_indices].reset_index(drop=True)
    # Unlabeled dataset # will be used for label propagtion
    X_unlabeled = X_Notarget[list(y_Notarget.drop(random_indices).index)]
    y_unlabeled = y_Notarget.drop(random_indices).reset_index(drop=True)
    # 1st training dataset
    X_train = np.concatenate([X_target, X_Notarget_matched])
    y_train = pd.concat([y_target ,y_Notarget_matched]).reset_index(drop=True)

    return X_unlabeled, y_unlabeled, X_train, y_train