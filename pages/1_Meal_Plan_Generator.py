import streamlit as st
import pandas as pd
import pickle
import io
from utils import get_obesity_status, change_order # get_user
from optimize_LP import LP_MealBundle
from constraints import calc_bmi, get_ibw, IBW_constraints

st.set_page_config(
    page_title="HumbleNutri App",
    page_icon=":green_salad:",
    layout="wide"
)

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)
st.sidebar.text("")
st.sidebar.success("👆️ Select an option above")
st.sidebar.text("")
st.sidebar.text("©️Information Sciences Institute 2024")

feature_lst = ['bundle_num','meal_type','title', 'description','duration', 'directions','ingredients',
               'direction_size','ingredients_sizes',
               'average_rating', 'number_of_ratings',
                'servingsPerRecipe', 'servingSize [g]',
               'calories [cal]','saturatedFat [g]','sugars [g]','cholesterol [mg]', 'sodium [mg]','totalCarbohydrate [g]',
                'protein [g]','dietaryFiber [g]','totalFat [g]','RRR_macro','nutri_score','rec_score']

@st.cache_data
def to_excel(df1, df2):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        
        # Access the workbook and worksheets for formatting if needed
        workbook  = writer.book
        worksheet1 = writer.sheets['Sheet1']
        worksheet2 = writer.sheets['Sheet2']
    return output.getvalue()

# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv(index=False).encode("utf-8")

def main():
    st.title('Bundle Generator') # HumbleNutri App prototype
    # # Initialize session state
    # if 'submitted' in st.session_state.keys():
    #     del st.session_state['submitted']
    # Input from the user
    with st.form("input_form"):
        gender_choice = st.selectbox('Gender', ['Male', 'Female'])
        height_choice = st.number_input('Height (cm)', min_value=150.0, max_value=250.0, step=0.1)
        weight_choice = st.number_input('Weight (kg)', min_value=35.0, max_value=300.0, step=0.1)
        age_choice = st.number_input('Age', min_value=18, max_value=100, step=1)
        after_surgery_choice = st.radio('Post-Surgery Recovery Phase', [True, False])
        activity_level_choice = st.selectbox('Activity Level', ['Sedentary', 'Lightly active', 'Moderately active', 'Active', 'Very active'])
        pre_diabetes_choice = st.radio('Pre-Diabetes', [True, False])
        high_cholesterol_choice = st.radio('High Cholesterol', [True, False])
        hypertension_choice = st.radio('Hypertension', [True, False])
        # user = st.radio('User', ['User-1', 'User-2', 'User-3', 'User-4', 'User-5',])
        ### Submit
        submitted = st.form_submit_button("Submit")
    
    # # Stay showing the health result and weekly agenda # Bug: reruns automatically when switching pages
    # if submitted:
    #     st.session_state.submitted = True

    # # Run LP and show result
    # if "submitted" in st.session_state:
    if submitted:
        # Calculate health info based on submitted information
        bmi = calc_bmi(height_choice, weight_choice)
        ibw_in_kg = get_ibw(gender = gender_choice, height = height_choice, weight = weight_choice, bmi=bmi)
        calories_const, protein_const_lower, protein_const_upper, sugar_const, carb_const, satfat_const, sodium_const,\
              fiber_const = IBW_constraints(gender = gender_choice, height = height_choice, weight = weight_choice,
                                            age = age_choice, after_surgery = after_surgery_choice,
                                            activity_level = activity_level_choice, pre_diabetes = pre_diabetes_choice,
                                            high_cholesterol = high_cholesterol_choice, hypertension = hypertension_choice)
        constraints = {'Nutrients': ['Calories', 'Carbohydrate', 'Total Fat', 'Saturated Fat', 'Total Sugar',
                                     'Sodium', 'Protein', '','Fiber'],
                                ' ': ['less than'] * 6 + ['more than', 'less than', 'more than'],
                                'Amount': [f"{round(calories_const)} (cal)",
                                           f"{round(carb_const)} (g)",
                                           f"{round(calories_const*0.3/9)} (g)",
                                           f"{round(satfat_const)} (g)",
                                           f"{round(sugar_const)} (g)",
                                           f"{sodium_const} (mg)", 
                                           f"{round(protein_const_lower)} (g)",
                                           f"{round(protein_const_upper)} (g)",
                                           f"{round(fiber_const)} (g)"]}
        constraints_df=pd.DataFrame(constraints)
        st.markdown(f"> ##### **BMI:** <u>{round(bmi,1)}</u> (<u>{get_obesity_status(bmi)[0]}</u>)", unsafe_allow_html=True)
        st.markdown(f"> ##### **Weight Goals:** <u>{get_obesity_status(bmi)[1]}</u> &ensp; :arrow_right: &ensp; **Ideal Body Weight:** <u>{round(ibw_in_kg,1)} (kg)</u>", unsafe_allow_html=True)
        st.markdown(f"> ##### Nutrient Constraints", unsafe_allow_html=True)
        #st.table(constraints_df)
        st.dataframe(constraints_df, hide_index = True, use_container_width = True)
        # Generate bundles
        all_bundles=list()
        lp_df = pd.DataFrame()
        # st.write(f"Finding optimal bundles...")
        for t in range(5,26,5):
            n = 3958 # Choose just 1 user to minimize confusion # get_user(user)
            # Assume they are our target patient
            with open(f'./final_target_items/bf_items_{n}_filtered_t{t}.pkl', 'rb') as x: bf = pickle.load(x)
            with open(f'./final_target_items/app_items_{n}_wg_t{t}.pkl', 'rb') as x: wg = pickle.load(x)
            with open(f'./final_target_items/app_items_{n}_vegie_t{t}.pkl', 'rb') as x: vg = pickle.load(x)
            with open(f'./final_target_items/main_items_{n}_filtered_t{t}.pkl', 'rb') as x: main = pickle.load(x)
            # Call the optimization function
            bundles, _ = LP_MealBundle(bf_items = bf, wg_items = wg, vg_items = vg, main_items = main,
                                        gender=gender_choice, height = height_choice, weight = weight_choice,
                                        age = age_choice, after_surgery = after_surgery_choice, activity_level = activity_level_choice,
                                        pre_diabetes = pre_diabetes_choice, high_cholesterol = high_cholesterol_choice,
                                        hypertension = hypertension_choice)
            all_bundles += bundles
        # Format it in to lst then data frame
        lp_lst = []
        for i, d in enumerate(all_bundles):
            for key, value in d.items():
                if key != 'objective_value':
                    lp_lst.append(('Bundle-'+str(i+1), key, *value))
        # to df
        lp_tmp = pd.DataFrame(lp_lst,columns=feature_lst)
        # lp_tmp['patient_num'] = user
        lp_df = pd.concat([lp_df,lp_tmp])
        # st.write(lp_df)
        # Write weekly plan 
        st.header("Weekly Plan A", divider="blue")
        # Bundles are already sorted by rec_score
        # first_week = lp_df[lp_df['bundle_num'].isin(['Bundle-1','Bundle-2','Bundle-3'])]
        weekly_plan = lp_df[lp_df['bundle_num'].isin(pd.Series(lp_df['bundle_num'].unique()).sample(n=6))].reset_index(drop=True)
        weekly_plan['bundle_num'] = [f'Bundle-{i}' for i in range(1, 7) for _ in range(6)]
        schedule = {'Meal': ["Breakfast", "Lunch", "Lunch-Side", "Dinner-Main","Dinner-Side (whole-grains)","Dinner-Side (vegetables)"],
                    'Monday': ['♻️ (Leftovers)'] * 6 ,
                    'Tuesday': ['♻️ (Leftovers)'] * 6,
                    'Wednesday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-1']['title'])),
                    'Thursday': ['♻️ (Leftovers)'] * 6,
                    'Friday': ['♻️ (Leftovers)'] * 6 ,
                    'Saturday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-2']['title'])),
                    'Sunday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-3']['title']))}
        first_week_df = pd.DataFrame(schedule)
        # st.dataframe(first_week_df, hide_index = True)
        st.table(first_week_df)
        # st.write("* Bundles are sorted based on recommendation score")
        # st.write("* Weekly plan A: First 3 bundles (Bundle-1, 2, 3) from the entire bundle set")
        # Write weekly plan
        st.header("Weekly Plan B", divider="green")
        # Bundles are already sorted by rec_score
        # second_week = lp_df[lp_df['bundle_num'].isin(['Bundle-4','Bundle-5','Bundle-6'])].reset_index(drop=True)
        schedule = {'Meal': ["Breakfast", "Lunch", "Lunch-Side", "Dinner-Main","Dinner-Side (whole-grains)","Dinner-Side (vegetables)"],
                    'Monday': ['♻️ (Leftovers)'] * 6 ,
                    'Tuesday': ['♻️ (Leftovers)'] * 6,
                    'Wednesday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-4']['title'])),
                    'Thursday': ['♻️ (Leftovers)'] * 6,
                    'Friday': ['♻️ (Leftovers)'] * 6 ,
                    'Saturday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-5']['title'])),
                    'Sunday': change_order(list(weekly_plan[weekly_plan.bundle_num=='Bundle-6']['title']))}
        second_week_df=pd.DataFrame(schedule)
        # st.dataframe(first_week_df, hide_index = True)
        st.table(second_week_df)
        st.write("* Weekly plans were randomly chose from the recommended candidate bundles. Re-submit to explore different weekly plans, or download all candidate bundles below.")
        st.write("* Nutrient constraints based on provided patient information is included in Sheet-2 of Excel files.")
        st.download_button(
        label="Download these weekly meal plans in Excel",
        data=to_excel(weekly_plan, constraints_df),
        file_name="HumbleNutri_Bundles.xlsx",
        mime="application/vnd.ms-excel",
        )

        st.download_button(
        label="Download all bundles and patient constraints in Excel",
        data=to_excel(lp_df, constraints_df),
        file_name="HumbleNutri_Bundles.xlsx",
        mime="application/vnd.ms-excel",
        )


    # # Download button for csv
    # try:
    #     # st.download_button(
    #     # label="Download all bundles as CSV",
    #     # data=convert_df(lp_df),
    #     # file_name="HumbleNutri_Bundles.csv",
    #     # mime="text/csv",
    #     # )
    #     st.download_button(
    #     label="Download bundles and patient constraints in Excel",
    #     data=to_excel(lp_df, constraints_df),
    #     file_name="HumbleNutri_Bundles.xlsx",
    #     mime="application/vnd.ms-excel",
    #     )
    # except:
    #     pass
    


if __name__ == "__main__":
    main()