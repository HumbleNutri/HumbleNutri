import streamlit as st
import pandas as pd
import pickle
from utils import get_user
from optimize_LP import LP_MealBundle

feature_lst = ['title', 'description','duration', 'directions','ingredients',
               'direction_size','ingredients_sizes',
               'average_rating', 'number_of_ratings',
                'servingsPerRecipe', 'servingSize [g]',
               'calories [cal]','saturatedFat [g]','sugars [g]','cholesterol [mg]', 'sodium [mg]','totalCarbohydrate [g]',
                'protein [g]','dietaryFiber [g]','totalFat [g]','RRR_macro','nutri_score','rec_score']
feature_lst_all=['bundle_num','meal_type']+feature_lst

def main():
    st.title('Nutrition Optimization App')

    # Input from the user
    with st.form("input_form"):
        gender_choice = st.selectbox('Gender', ['Male', 'Female'])
        height_choice = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, step=0.1)
        weight_choice = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, step=0.1)
        age_choice = st.number_input('Age', min_value=18, max_value=100, step=1)
        after_surgery_choice = st.radio('After Surgery', [True, False])
        activity_level_choice = st.selectbox('Activity Level', ['Sedentary', 'Lightly active', 'Moderately active', 'Active', 'Very active'])
        pre_diabetes_choice = st.radio('Pre-Diabetes', [True, False])
        high_cholesterol_choice = st.radio('High Cholesterol', [True, False])
        hypertension_choice = st.radio('Hypertension', [True, False])
        user = st.radio('User', ['User-1', 'User-2', 'User-3', 'User-4', 'User-5',])

        submitted = st.form_submit_button("Submit")
        if submitted:
            all_bundles=list()
            for t in range(5,26,5):
                n = get_user(user)
                # Assume they are our target patient
                with open(f'./final_target_items/bf_items_{n}_filtered_t{t}.pkl', 'rb') as x: bf = pickle.load(x)
                with open(f'./final_target_items/app_items_{n}_wg_t{t}.pkl', 'rb') as x: wg = pickle.load(x)
                with open(f'./final_target_items/app_items_{n}_vegie_t{t}.pkl', 'rb') as x: vg = pickle.load(x)
                with open(f'./final_target_items/main_items_{n}_filtered_t{t}.pkl', 'rb') as x: main = pickle.load(x)
                # Call the optimization function
                bundles, _ = LP_MealBundle(bf_items=bf, wg_items=wg, vg_items=vg, main_items=main,
                                           gender=gender_choice, height = height_choice, weight = weight_choice,
                                           age = age_choice, after_surgery = after_surgery_choice, activity_level = activity_level_choice,
                                           pre_diabetes = pre_diabetes_choice, high_cholesterol = high_cholesterol_choice,
                                           hypertension = hypertension_choice, user=n)
                all_bundles += bundles
            lp_lst = []
            for i, d in enumerate(all_bundles):
                for key, value in d.items():
                    if key != 'objective_value':
                        lp_lst.append(('Bundle-'+str(i+1), key, *value))
            # to df
            lp_tmp = pd.DataFrame(lp_lst,columns=feature_lst_all)
            lp_tmp['patient_num'] = user
            lp_df = pd.concat([lp_df,lp_tmp])
            st.write(lp_df)

if __name__ == "__main__":
    main()