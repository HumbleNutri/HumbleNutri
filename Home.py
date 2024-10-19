import streamlit as st

st.set_page_config(
    page_title="HumbleNutri App",
    page_icon=":green_salad:",
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

st.write("healthy and culture-aware meal plan recommender system")
st.write("")

st.markdown(
    """ 
        > Welcome to HumbleNutri App! 👋 We are research scientists and engineers from USC's Information Sciences Institute, \
            and we created this meal prescription recommendation app for patients with dietary requirements \
                through the guidance from a Registered Dietitian Nutritionist.
    """
)

st.write("")

st.markdown(
        """
            To date, clinicians have implemented interventions in patient populations by either manually \
                designing meal plans (a burdensome and unscalable approach), or employing off-the-shelf meal planning applications (apps).\
                Existing apps on the market suffer a major limitation: while they account for dietary constraints, their meal recommendations\
                focus on Western-normative diets, and their approaches do not enable customization to other sociocultural dietary patterns (e.g., Hispanic foods).\
                This represents a major gap given that:
            - Populations typically targeted in meal prescription interventions are of racial/ethnic minorities and may not have Western normative diets
            - Extensive evidence from dietary interventions has established that dietary changes are sustainably maintained and lead to nutritional benefits only if the intervention works around individuals\' existing diets.
    	"""
    )

st.write("")

st.markdown("Towards this aim, we present")
st.markdown("<u>\"*HUMBLE-NUTRI: **H**ealthy and c**U**lture-aware **M**eal **B**und**LE** recommendation with **NUTRI**tionist feedback*\"</u>.", unsafe_allow_html=True)
st.markdown(
    """      
        To build HumbleNutri, we employ the HUMMUS dataset consisting of 2M reviews from 300K users over 500k recipes. For more details\
        check their [GitLab repository](https://gitlab.com/felix134/connected-recipe-data-set). HUMMUS is an aggregated dataset from various\
        source including Recip1M, Food.com, FoodKG, and etc. In addition to the offering of rich details of each recipe and user-recipe interaction,\
        it provides a healthiness score along with its nutrients. 45% of the recipes in the dataset have tags that indicate what type of food it is, \
        for example, breakfast or Italian. We use these tags to label this entire unlabeled dataset by performing a semi-supervised self-training approach. \
        Using this approach, we identify if the recipe is a breakfast item or not, if the recipe is an appetizer, main dish, dessert, drink, or sauce, and \
        if the recipe is Latin American food or not. Details of this approach to fully utilize the HUMMUS dataset to build HumbleNutri is included in our\
        paper. HumbleNutri consists of three independent recommender systems for breakfast, appetizers, and main dishes. Using the \
        recommended items from these recommender systems and feedback from nutritionists, we generate the meal bundle. We note that we also apply the initial healthiness filter \
        using the 'nutri score' suggested by authors of HUMMUS before putting into the recommender system.
    """
    )

st.text("")
# for users that we picked from the HUMMUS dataset
st.markdown(
    """
       In this prototype web app, we generate a weekly meal plan consists of 3 bundles. We selected an example user if: \
        i) Contributed a minimum of 30 reviews ii) Reviewed at least one item each from breakfast, appetizers, and main dishes \
        iii) a significant portion of their reviews were dedicated to Latin American cuisine. We consider the top 25% recommended items for each user as a candidate recipes. \
        Example user that was used in this prototype could be substitute with your patient, upon providing us with your patient's recipe preference (e.g. ratings on recipes). \
        We then apply linear programming with objective function to maximize recommendation score from the recommender system to tailor the bundles to each user \
        along with multiple constraints based on patient information (age, weight, health condition, etc.) that we assigned to each user to generate bundles from candidate items. \
        Details of this approach is be also included in our paper.
    """
)
st.text("")

st.text("")

st.markdown(
    """
        ##### 👉️ Check out [the GitHub repo](https://github.com/HumbleNutri/HumbleNutri)
    """
)

# Initialize session state
if 'submitted' in st.session_state.keys():
    del st.session_state['submitted']