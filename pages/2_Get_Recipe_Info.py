import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import round_up_numbers_in_string, get_
# pip install openpyxl # dependency for pd.read_excel()


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

st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

# Initialize session state
if 'submitted' in st.session_state.keys():
    del st.session_state['submitted']
# main meals # side dishes can be made in parallel to the main dishes
main_meals = ['breakfast','lunch','dinner-main']
nutrient_info = ['calories [cal]','totalCarbohydrate [g]','totalFat [g]','saturatedFat [g]', 'sugars [g]',  'sodium [mg]',
            'protein [g]', 'dietaryFiber [g]']

def main():
    uploaded_file = st.file_uploader("Choose an Excel file", type = "xlsx", accept_multiple_files=False)
    while uploaded_file is not None:
        try:
            lp_df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
            constraints_df  = pd.read_excel(uploaded_file, sheet_name='Sheet2')
            break
        except:
            pass
    # Input from the user
    try:
        with st.form("input_form"):
            meal_number = st.selectbox('Choose a Meal Bundle', sorted(set(lp_df['meal_num']), key=lambda x: int(x.split('-')[-1])))
            meal_choice = st.selectbox('Choose a Meal', ["Breakfast", "Lunch", "Lunch-Side", "Dinner-Main","Dinner-Side (whole-grains)","Dinner-Side (vegetables)"])
            ### Submit
            submitted = st.form_submit_button("Submit")
        # # Stay showing the health result and weekly agenda # Bug: reruns automatically when switching pages
        # if submitted:
        #     st.session_state.submitted = True

        # # Run LP and show result
        # if "submitted" in st.session_state:
        if submitted:
            # Chosen bundle, meal
            bundle_df = lp_df[lp_df['meal_num']==meal_number].reset_index(drop=True)
            meal_df = bundle_df[bundle_df['meal_type']==get_(meal_choice)].reset_index(drop=True)
            # List meal name, ingredients, and recipe
            st.header(f"{meal_df['title'][0]}", divider="red")
            st.markdown(f"*** Servings per Recipe: {meal_df['servingsPerRecipe'][0]}", unsafe_allow_html=True)
            st.markdown(f"*** Servings Size: {meal_df['servingSize [g]'][0]} (g)", unsafe_allow_html=True)
            st.markdown(f"*** Duration: {meal_df['duration'][0]} (min)", unsafe_allow_html=True)

            col1, col2= st.columns(2)

            with col1:
                st.markdown("> #### Ingredients")
                for k in eval(meal_df['ingredients'][0]).keys():
                    if k != '':
                        st.markdown(f"> {k}", unsafe_allow_html=True)
                    for item, amount in eval(meal_df['ingredients'][0])[k]:
                        amount = round_up_numbers_in_string(amount)
                        st.markdown(f"- {amount.replace(' time(s)', '').replace('  ', ' ').replace('$template2$','').replace('/ ','1 ').strip()} {item.replace('$template2$','').strip()}", unsafe_allow_html=True)

            with col2:
                st.markdown("> #### Directions", unsafe_allow_html=True)
                for i, step in enumerate(eval(meal_df['directions'][0]), start=1):
                    st.markdown(f"Step {i}: {step}", unsafe_allow_html=True)

            # Show bar plots
            daily_values = constraints_df['Amount'].str.extract('(\d+)')[0].astype(int).tolist()
            # remove protien constraint upper
            daily_values.pop(-2)
            # nutirent names
            amounts = meal_df[nutrient_info].values.flatten().tolist()
            daily_value_percentages = [round(amount / daily_value * 100,1) for amount, daily_value in zip(amounts, daily_values)]
            # plot with plotly
            df = pd.DataFrame({'Nutrient': nutrient_info, 'Amount': amounts, 'Daily Value %': daily_value_percentages})
            df['Category'] = 'Restricted'
            df['Category'] = np.where(df['Nutrient'].isin(['protein [g]', 'dietaryFiber [g]']), 'Recommended', df['Category'])
            fig = px.bar(df, x='Nutrient', y='Daily Value %', hover_data={'Amount': True},  # Show exact amount on hover
                        labels={'Daily Value %': '% Daily Value'}, color = 'Category',
                        color_discrete_map={'Restricted': 'red',
                                            'Recommended': 'green'},
                        title='Nutritional Contents of the Recipe')
            st.plotly_chart(fig, use_container_width=True)

    except:
        pass
    


if __name__ == "__main__":
    main()