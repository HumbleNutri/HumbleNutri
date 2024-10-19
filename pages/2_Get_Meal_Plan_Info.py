import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# Initialize session state
if 'submitted' in st.session_state.keys():
    del st.session_state['submitted']
# main meals # side dishes can be made in parallel to the main dishes
main_meals = ['breakfast','lunch','dinner-main']
key_info = ['meal_type', 'servingSize [g]', 'duration', 'average_rating','calories [cal]','totalCarbohydrate [g]',
            'totalFat [g]','saturatedFat [g]', 'sugars [g]',  'sodium [mg]','protein [g]', 'dietaryFiber [g]']

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
            ### Submit
            submitted = st.form_submit_button("Submit")
        # # Stay showing the health result and weekly agenda # Bug: reruns automatically when switching pages
        # if submitted:
        #     st.session_state.submitted = True

        # # Run LP and show result
        # if "submitted" in st.session_state:
        if submitted:
            # Chosen bundle
            viz_df = lp_df[lp_df['meal_num']==meal_number].reset_index(drop=True)
            # Show main metric
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total meal bundle calories", f"{round(viz_df['calories [cal]'].sum())} (cal)")
            col2.metric("Total meal prep time", f"{round(viz_df[viz_df['meal_type'].isin(main_meals)]['duration'].sum())} (min)")
            col3.metric("Total servings made", f"{viz_df['servingsPerRecipe'].sum()}")
            col4.metric("Average ratings", f"{round(viz_df['average_rating'].mean(),2)}")
            # List bundles
            st.markdown(f"> ##### Breakfast: <u>{viz_df['title'][0]}</u>", unsafe_allow_html=True)
            st.markdown(f"> ##### Lunch: <u>{viz_df['title'][2]}</u> with <u>{viz_df['title'][1]}</u>", unsafe_allow_html=True)
            st.markdown(f"> ##### Dinner: <u>{viz_df['title'][5]}</u> with <u>{viz_df['title'][3]}</u> and <u>{viz_df['title'][4]}</u>", unsafe_allow_html=True)
            # Show nutrients
            viz_df = viz_df.replace(['breakfast','lunch','lunch-side','dinner-main','dinner-side-wg','dinner-side-vg'],
                                    ["Breakfast", "Lunch", "Lunch-Side", "Dinner-Main","Dinner-Side (whole-grains)","Dinner-Side (vegetables)"])
            viz_df = viz_df.reindex([0,2,1,5,3,4]).reset_index(drop=True)
            st.dataframe(viz_df[key_info].rename(columns={"meal_type":"Meal Type",
                                                          "servingSize [g]":"Serving Size [g]",
                                                          "duration":"Duration",
                                                          "average_rating":"Average Rating",
                                                          "totalCarbohydrate [g]":"total carbohydrate [g]",
                                                          "totalFat [g]":"total fat [g]",
                                                          "saturatedFat [g]":"saturated fat [g]",
                                                          "dietaryFiber [g]":"dietary fiber [g]"}), hide_index = True, use_container_width = True)
            # Show bar plots
            daily_values = constraints_df['Amount'].str.extract('(\d+)')[0].astype(int).tolist()
            # remove protien constraint upper
            daily_values.pop(-2)
            # nutirent names
            show_nutrients = key_info[4:]
            amounts_raw = list(viz_df[show_nutrients].iloc[0].values + viz_df[show_nutrients].iloc[1].values +\
                            viz_df[show_nutrients].iloc[2].values + viz_df[show_nutrients].iloc[3].values +\
                            viz_df[show_nutrients].iloc[4].values + viz_df[show_nutrients].iloc[5].values)
            amounts = [round(am, 1) for am in amounts_raw]
            daily_value_percentages = [round(amount / daily_value * 100,1) for amount, daily_value in zip(amounts, daily_values)]
            # plot with plotly
            df = pd.DataFrame({'Nutrient': show_nutrients, 'Amount': amounts, 'Daily Value %': daily_value_percentages})
            df['Category'] = 'Restricted'
            df['Category'] = np.where(df['Nutrient'].isin(['protein [g]', 'dietaryFiber [g]']), 'Recommended', df['Category'])
            fig = px.bar(df, x='Nutrient', y='Daily Value %', hover_data={'Amount': True},  # Show exact amount on hover
                        labels={'Daily Value %': '% Daily Value'}, color = 'Category',
                        color_discrete_map={'Restricted': 'red',
                                            'Recommended': 'green'},
                        title='Total Nutritional Contents of the Meal Bundle')
            st.plotly_chart(fig, use_container_width=True)

    except:
        pass
    


if __name__ == "__main__":
    main()
