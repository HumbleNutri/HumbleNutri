import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pickle
from modules.utils import *
import argparse

nltk.download('stopwords')

def preprocess_item(item):
    """Preprocess the item by converting to lowercase and removing punctuation, numbers, and stopwords."""
    # Convert to lowercase
    item = item.lower()
    
    # Remove numbers
    item = ''.join(char for char in item if not char.isdigit())
    
    # Remove punctuation
    item = ''.join(char for char in item if char.isalnum() or char.isspace())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = item.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def extract_menu(tuples_list):
    return [item[0] for item in tuples_list]

def create_vector_representation(items):
    """Create TF-IDF vector representations of items"""
    processed_items = [preprocess_item(item_name + ', ' + ingr + ', '+ recipe) for item_name, ingr, recipe in items]
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(processed_items)

def calculate_health_score(item_vector, category_vectors):
    """Calculate a health score based on cosine similarity with healthy and unhealthy categories"""
    penalty = 3 # Penalty on unhealthy items
    healthy_sim = cosine_similarity(item_vector, category_vectors['healthy']).flatten()[0]
    unhealthy_sim = cosine_similarity(item_vector, category_vectors['unhealthy']).flatten()[0]
    return healthy_sim - unhealthy_sim*penalty

def rank_items(all_items, categories):
    # Create vector representations
    vectorizer, all_item_vectors = create_vector_representation(all_items)
    # Create category vectors
    healthy_vector =vectorizer.transform([' '.join(categories['healthy_ingredients'] + categories['healthy_preparation'])])
    unhealthy_vector = vectorizer.transform([' '.join(categories['unhealthy_ingredients'] + categories['unhealthy_preparation'])])
    category_vectors = {'healthy': healthy_vector, 'unhealthy': unhealthy_vector}
    
    # Calculate health scores
    health_scores = [calculate_health_score(item_vector, category_vectors) for item_vector in all_item_vectors]
    
    # Rank items by health score
    sorted_items = sorted(zip(extract_menu(all_items), health_scores), key=lambda x: x[1], reverse=True)
    
    candidate_items = [item for item in sorted_items if item[1] > 0]
    
    return candidate_items

def filtering(all_candidates, user, keywords):
    try:
        all_candidate_items = []
        for item in all_candidates:
            name = item[0]
            data_dict = ast.literal_eval(item[4])
            ingr =' '.join([ingr for key in data_dict for ingr, _ in data_dict[key]])
            recipe = item[3]
            all_candidate_items.append((name,ingr,recipe))
        recommended_items = rank_items(all_candidate_items, keywords)

        print(f"\nRecommended healthy items for user {user}:")
        for item, score in recommended_items:
            print(f"- {item} (Health Score: {score:.3f})")

        print(f"\nRecommended {len(recommended_items)} items out of {len(all_candidate_items)} total items.")

    except Exception as e:
        print(f"An error occurred for user {user}: {str(e)}")
        
    return recommended_items

def align(meal_type):
    user = 3958 # Use one user for showcase
    # Set meals str
    if meal_type in ['wg_side, vg_side']:
        meal = 'app'
    else:
        meal = meal_type
    # Over the initial candidates
    for t in range(5,26,5):
        # read all breakfast
        with open(f'./final_target_items/{meal}_items_{user}_t{t}.pkl', 'rb') as f:
            all_candidate = pickle.load(f)
        # define list
        candidate_lst=[]
        # get candidate lst
        candid = filtering(all_candidate, user, get_keywords(meal_type=meal_type))
        candidates =[item[0] for item in candid]
        # write lst
        for i in range(len(all_candidate)):
            if all_candidate[i][0] in candidates:
                if all_candidate[i][17] + all_candidate[i][18] + all_candidate[i][19] != 0:
                    candidate_lst.append(all_candidate[i])
        # No duplicate side for different side types
        if meal_type == 'vg_side':
            # read wholegrain side
            with open(f'./final_target_items/wg_side_items_{user}_filtered_t{t}.pkl', 'rb') as f:
                wg_side = pickle.load(f)
            # Additional filtering to exclude items in wholegrain side
            wg_items = {item[0] for item in wg_side}
            candidate_lst = [item for item in candidate_lst if item[0] not in wg_items]
        # Export
        with open(f'./final_target_items/{meal_type}_items_{user}_filtered_t{t}.pkl', 'wb') as f: pickle.dump(candidate_lst, f)


if __name__ == "__main__":
    # # Get arguments for alignment to test on
    # parser = argparse.ArgumentParser(description="Run the script with the meal type you want to test alignment on.")
    # parser.add_argument('mealtype', type=str, help="Type of label to create. ex) bf, wg_side, vg_side, main, all")
    # args = parser.parse_args()
    # RDN-guided alignment
    align(meal_type='bf')
    align(meal_type='wg_side')
    align(meal_type='vg_side')
    align(meal_type='main')