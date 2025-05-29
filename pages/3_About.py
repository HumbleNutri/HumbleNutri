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
#

st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

st.markdown("## Healthy and Culturally-tailored Meal Plan Recommender System")
st.text("")
st.markdown(
    """ 
        > Welcome to HumbleNutri App! 👋 This app contains a showcase of meal prescription recommender systems designed for nutritionists helping patients with dietary requirements. This synergistic systems with integrated modules are designed through the guidance \
                of a Registered Dietitian Nutritionist (RDN) specializing in organ donors.
    """
)

st.text("")

st.markdown(
    """ 
        > **How to use:**

        > Start at `Home`. Input the clinical information needed to generate a personalized meal prescription plan. No information will be saved on this app.
       
        > Download an Excel file with the personalized meal plan, and upload the Excel file at the `Get Meal Plan Info` and `Get Recipe Info` pages to get detailed statistical visualization as well as descriptions about the meals and recipes on the dashboard.
    """
)

st.text("")

st.markdown(
        """
            To date, clinicians have implemented interventions in patient populations by either manually \
                designing meal plans (a burdensome and unscalable approach), or employing off-the-shelf meal planning applications (apps).\
                Existing apps on the market suffer a major limitation: while they account for dietary constraints, their meal recommendations\
                focus on Western-normative diets, and their approaches do not offer customization to other sociocultural dietary patterns (e.g., Hispanic foods).\
                This represents a major gap given that:
            - Populations commonly targeted in meal prescription interventions are of racial/ethnic minorities and may not have Western normative diets.
            - Extensive evidence from dietary interventions has established that dietary changes are sustainably maintained and lead to nutritional benefits only if the intervention works around individuals\' existing diets.
    	"""
    )


st.markdown("Towards this aim, we present")
st.markdown("<u>\"*HUMBLE-NUTRI: **H**ealthy and c**U**lturally-tailored **M**eal **B**und**LE** recommendation with **NUTRI**tionist guidance*\"</u>.", unsafe_allow_html=True)
st.text("")
st.text("")
st.markdown("##### **Meal Structure**", unsafe_allow_html=True)
st.markdown(
    """      
        Following the guidance from our Dietitian Nutritionist, the HumbleNutri web app enforces a daily meal structure of \
            a “bundle” of 3 meals a day, combined into a weekly meal plan consisting of three daily meal bundles. The implication is that a user of the app \
                will cook larger amounts during these three days a week to enable leftovers on the other days. The daily meal bundles include breakfast; lunch, consisting of a main dish with \
                    a vegetable side; and dinner, consisting of a main dish with a vegetable side and whole grain or legume side. 
    """
    )
st.markdown("##### **Inputs**", unsafe_allow_html=True)
st.markdown(
        """
            - The clinical factors for the user/patient (who the recommended meal plans are generated for). These include sex, age, weight, height, and whether they have certain health conditions or are recovering from surgery). These should be input by the patient’s nutritionist/medical provider.
            - Information on the types of cuisines and foods a user likes, and user's rating on a set of 10-15 example meals (matching their preferred cuisine) for a warm-start recommendation.<a href="#note1" id="note1ref"><sup>1</sup></a>
    	""", unsafe_allow_html=True
    )
st.markdown("##### **Data**", unsafe_allow_html=True)
st.markdown(
    """      
        To build HumbleNutri, we employ the HUMMUS<a href="#note1" id="note1ref"><sup>2</sup></a> dataset consisting of ~2M reviews from ~300K users over ~500k recipes. For more details\
        check their [GitLab repository](https://gitlab.com/felix134/connected-recipe-data-set). HUMMUS is an aggregated dataset from various\
        sources. Recipes include ingredients, nutrients, instructions, and tags (labels of meal types, such as breakfast, or cuisines, such as Italian), and several \
            calculated nutrition scores (WHO, FSA, NutriScore). The dataset also includes reviews of these recipes from users, who may have multiple reviews, which can be used for a recommendation task. \
        We utilized ~45% of the recipes in the dataset that have tags, and employ a self-training approach in semi-supervised fashion to label the entire recipe dataset by meal type (breakfast; appetizer, main dish, dessert, drink, or sauce). \
            We also create a label indicating whether the recipe is of a Latin American heritage style cuisine or not (This could be any other cuisine based on the patient population). We aggregated existing tags from all countries in Central and South America to create this category, which we then used to predict it across the dataset.
        Additional filters are applied before collaborative filtering step to ensure all meals included fit within a general healthy dietary pattern, removing the recipes that have a low ‘NutriScore’ (as calculated in the HUMMUS dataset). \
            Details of our approach to fully utilize the HUMMUS dataset to build HumbleNutri are stated in our paper.
    """, unsafe_allow_html=True
    )
st.markdown("##### **System**", unsafe_allow_html=True)
st.markdown(
    """
        HumbleNutri is a multi-module system that generates culturally-tailored and clinically personalized meal recommendations for patients, guided by clinical nutrition principles and cultural relevance. The system works through integration of four modules: (1) a semi-supervised learning method to infer cuisine and meal type labels at scale to identify culturally relevant recipes; (2) collaborative filtering for recipe recommendation, (3) a "recipe alignment" step to reflect nutritionist-specified constraints on ingredients and preparation methods, and (4) a structured optimization framework to generate personalized weekly meal bundles. Following the guidance from the Registered Dietitian Nutritionist (RDN), these modules are integrated to produce daily meal bundles structured into breakfast, lunch consisting of main-dish and 1 side-dish (vegetables), and dinner consisting of main-dish and 2 side-dishes (vegetables and whole grains).
    """
)

st.markdown("> **RDN Guidance**", unsafe_allow_html=True)
st.markdown(
    """
        After collaborative filtering module for initial recommendation, we align the recipes based on RDN guidance by retrieving recipes appropriate to the patient populations. The full set of nutritional equations and the constraints defined by our collaborating RDN are summarized in this [linked document](https://docs.google.com/document/d/12w9B8cionp8PxD6zfMDzp7uv377Dk4BwlfZGY-NVYEg/edit).
    """
)
st.markdown("> **Nutritional Constraint Model**", unsafe_allow_html=True)
st.markdown(
    """
        We then apply a integer linear programming model that is designed to meet a patient’s clinical nutritional needs while identifying recipes that are most closely aligned with their cuisine preferences. \
            This is achieved using an objective function that maximizes the recommendation score from the recommender system model while adhering to nutritional constraints defined by our RDN. \
                The set of nutritional constraints for patient is computed based on the information input by the patient’s nutritionist/medical provider on each patient’s clinical factors (sex, age, weight, height, and whether they have certain health conditions or are recovering from surgery)(See link above). \
                    These constraints define an individual’s daily macro and micronutrient needs. The linear programming model aims to identify combinations of recipes that taken together meet these daily nutrient needs, while also meeting a high score for meals that align with their dietary preferences. \
    """, unsafe_allow_html=True
)

st.text("")

st.markdown(
    """
        <a id="note1" href="#note1ref"><sup>1</sup></a> In this showcase app, example perference profile will be used instead of the warm-start to generate the meal plans.

       <a id="note1" href="#note1ref"><sup>2</sup></a>HUMMUS Dataset: Bölz, Felix, et al. "HUMMUS: A Linked, Healthiness-Aware, User-centered and Argument-Enabling Recipe Data Set for Recommendation." Proceedings of the 17th ACM Conference on Recommender Systems, 2023.
    """, unsafe_allow_html=True
)

st.text("")
st.text("")

st.markdown(
    """
        ##### 👉️ Check out [the GitHub repo](https://github.com/HumbleNutri/HumbleNutri)
    """
)

st.text("")
st.text("")


# Initialize session state
if 'submitted' in st.session_state.keys():
    del st.session_state['submitted']