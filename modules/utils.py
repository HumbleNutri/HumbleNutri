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

def duplicate_recipes(df):
    # Remove duplicate recipes
    df = df.drop_duplicates(subset=['new_recipe_id']).reset_index(drop=True)

    return df

def initial_filter(df):
    # Filter out recipes with lowest nutri score
    df = df[df['nutri_score']!=0].reset_index(drop=True) 

    return df

def get_reviews(df, reviews):
    df_reviews = reviews[reviews['new_recipe_id'].isin(list(set(df['new_recipe_id'])))].reset_index(drop=True)
     # Build feature matrix for whole latam bf
    pp_interactions = df_reviews[['new_member_id', 'new_recipe_id', 'rating']]
    pp_interactions = pp_interactions.rename(columns={'new_member_id': 'userID', 'new_recipe_id': 'itemID'})

    return pp_interactions

def hummus_remove_outliers(df, column_names):
    # Post-processing method from HUMMUS paper
    data = pd.DataFrame(df)

    for column_name in column_names:
        q1 = np.percentile(data[column_name], 25)
        q3 = np.percentile(data[column_name], 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        length_before = len(data)
        data = data.drop((data[data[column_name] >= upper].index | data[data[column_name] <= lower].index), axis=0)

        print('Removed ' + str(length_before - len(data)) + ' outliers of ' + column_name + '.')

    return data

def get_keywords(meal_type):
    # RDN-defined keywords
    bf_categories = {
    "healthy_ingredients": ["oatmeal", "wholegrain", "whole wheat", "berries", "nuts", "seeds", "avocado", "egg",
        "yogurt", "quinoa", "chia", "flaxseed", "spinach", "kale", "broccoli", "sweet potato", "oat", "grain", "buckwheat",
        "salmon", "tuna", "chicken breast", "turkey", "tofu", "tempeh", "lentils", "beans",
       "chickpeas", "olive oil", "almond milk", "soy milk", 'breakfast bars',
        "cucumber", "tomato", "bell pepper", "carrot", "zucchini", "asparagus", "brussels sprouts",
        "ginger", "turmeric", "almonds", "walnuts", "hazlenuts", "pecans", "sesame seeds", "peanuts",
        "pumpkin seeds", "sunflower seeds", "Greek yogurt", "cottage cheese",
        "mushrooms", "apples", "bananas", "oranges", "strawberries", "blueberries", "raspberries",
        "blackberries", "plums", "prunes", "apricots", "peach", "nectarines", "banana"],
    "healthy_preparation": ["steamed", "boiled", "baked", "grilled", "roasted", "raw", "toast", "sauteed", "puree","mixed", "blended","healthy","fermented", "simmered", "slow cooked"],
    # no meat for bf
    "unhealthy_ingredients": ["sugar", "syrup", "chocolate", "cream", "creamy", "bacon", "margarine", "shortening", "lard", "mayonnaise", "ranch", "thousand island", "sour cream",
                             "beef", "pork", "chicken", "turkey", "lamb", "duck", "ground", "steak", "leg", "shoulder", "belly", "breast", "skirt", "shank", "loin", "bone", "ham", "veal", "cutlet", "brisket", "meat", "ribs", "flank", "burger", "slider", "hot dog", "sausage", "hamburger", "BBQ", "barbecue"],
    "unhealthy_preparation": ["fried", "deep fried", "candied", "battered", "sweetened","fries", "fry", "frying", "deep fry", "deep frying", "candy", "candying", "batter", "battering", "dredge", "dredging", "dredged"]
    }
    # Define categories and their associated keywords
    wg_categories = {
        "healthy_ingredients": ["wholegrain", "whole wheat", "nuts", "seeds", "quinoa",
                                "potato", "sweet potato", "grain","tofu", "tempeh", "lentils", "beans","chickpeas",
                                "brown rice", "farro", "couscous", "bulgar", "barley", "wheat berries", "spelt", "buckwheat"],
        "healthy_preparation": ["steamed", "boiled", "baked", "grilled", "roasted", "raw", "toast", "sauteed", "puree","mixed", "blended","healthy","fermented", "simmered", "slow cooked"],
        "unhealthy_ingredients": ["sugar", "syrup", "chocolate", "cream", "creamy", "bacon", "margarine", "shortening", "lard", "mayonnaise", "ranch", "thousand island", "sour cream"],
        "unhealthy_preparation": ["fried", "deep fried", "candied", "battered", "sweetened","fries", "fry", "frying", "deep fry", "deep frying", "candy", "candying", "batter", "battering", "dredge", "dredging", "dredged"]
    }
    vg_categories = {
        "healthy_ingredients": ["spinach", "kale", "broccoli",
            "beans","snap peas","green","vegetable", "beet",
            "cauliflower",  "arugula", "mixed vegetables", "mixed green","ginger",
            "cucumber", "tomato", "bell pepper", "carrot", "zucchini", "asparagus", "brussels sprouts","mushrooms"],
        "healthy_preparation": ["steamed", "boiled", "baked", "grilled", "raw", "toast", "sauteed", "puree","mixed"],
        "unhealthy_ingredients": ["sugar", "syrup", "chocolate", "cream", "creamy", "bacon", "margarine", "shortening", "lard", "mayonnaise", "ranch", "thousand island", "sour cream"],
        "unhealthy_preparation": ["fried", "deep fried", "candied", "battered", "sweetened","fries", "fry", "frying", "deep fry", "deep frying", "candy", "candying", "batter", "battering", "dredge", "dredging", "dredged"]
    } 
    main_categories = {
        "healthy_ingredients": [],
        "healthy_preparation": ["steamed", "boiled", "baked", "grilled", "roasted", "raw", "toast", "sauteed", "puree","mixed", "blended","healthy","fermented", "simmered", "slow cooked"],
        "unhealthy_ingredients": ["sugar", "syrup", "chocolate", "cream", "creamy", "bacon", "margarine", "shortening", "lard", "mayonnaise", "ranch", "thousand island", "sour cream"],
        "unhealthy_preparation": ["fried", "deep fried", "candied", "battered", "sweetened","fries", "fry", "frying", "deep fry", "deep frying", "candy", "candying", "batter", "battering", "dredge", "dredging", "dredged"]
    } 
    # Get keywords dict
    keywords = {
        'bf': bf_categories,
        'wg_side': wg_categories,
        'vg_side': vg_categories,
        'main': main_categories
    }.get(meal_type, bf_categories)

    return keywords