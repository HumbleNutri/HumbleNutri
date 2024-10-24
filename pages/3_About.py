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

st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

st.markdown("## Healthy and Culture-aware Meal Plan Recommender System")
st.write("")

st.markdown(
    """ 
        > Welcome to HumbleNutri App! 👋 We are research scientists and engineers from USC's Information Sciences Institute, \
            and we created this meal prescription recommender app for patients with dietary requirements through the guidance \
                of a Registered Dietitian Nutritionist specializing in organ donors.
    """
)

st.write("")

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

st.text("")

st.markdown("Towards this aim, we present")
st.markdown("<u>\"*HUMBLE-NUTRI: **H**ealthy and c**U**lture-aware **M**eal **B**und**LE** recommendation with **NUTRI**tionist feedback*\"</u>.", unsafe_allow_html=True)
st.text("")
st.markdown("##### **Meal Structure**", unsafe_allow_html=True)
st.markdown(
    """      
        Following the guidance from our Dietitian Nutritionist, the prototype HumbleNutri web app enforces a daily meal structure of \
            a “bundle” of 3 meals a day, combined into a weekly meal plan consisting of three daily meal bundles. The implication is that a user of the app \
                will cook larger amounts during these three days a week to enable leftovers on the other days. The daily meal bundles include breakfast; lunch, consisting of a main dish with \
                    a vegetable side; and dinner, consisting of a main dish with a vegetable side and whole grain or legume side. 
    """
    )
st.markdown("##### **Inputs**", unsafe_allow_html=True)
st.markdown(
        """
            - The clinical factors for the user/patient (who the recommended meal plans are generated for). These include sex, age, weight, height, and whether they have certain health conditions or are recovering from surgery). These should be input by the patient’s medical provider.
            - [Coming soon] Information on the types of cuisines and foods a user likes, based on their scoring of a set of 10-15 example meals<a href="#note1" id="note1ref"><sup>1</sup></a>.
    	""", unsafe_allow_html=True
    )
st.markdown("##### **Data**", unsafe_allow_html=True)
st.markdown(
    """      
        To build HumbleNutri, we employ the HUMMUS<a href="#note1" id="note1ref"><sup>2</sup></a> dataset consisting of 2M reviews from 300K users over 500k recipes. For more details\
        check their [GitLab repository](https://gitlab.com/felix134/connected-recipe-data-set). HUMMUS is an aggregated dataset from various\
        sources. All recipes come from Food.com and include ingredients, nutrients, instructions, and tags (labels of meal types, such as breakfast, or cuisines, such as Italian), and several \
            calculated nutrition scores (WHO, FSA, NutriScore). Important for the recommendation model task, the dataset also includes reviews of these recipes from users, who may have multiple reviews. \
                Additional graphical structure behind these recipes is provided by using the knowledge graph FoodKG to map ingredients in the recipes as well as the recipe instructions to the FoodOn food ontology. \
        Since only 45% of the recipes in the dataset have tags, we use a self-training approach in semi-supervised fashion to label the entire recipe dataset by meal type (breakfast, appetizer, main dish, dessert, drink, or sauce). \
            We also create a label indicating whether the recipe is of a Latin American heritage style cuisine or not. We aggregated existing tags from all countries in Central and South America to create this category, which we then used to predict it across the dataset.
        Additional filters are applied to ensure all meals included fit within a general healthy dietary pattern, removing from the dataset recipes that have a low ‘NutriScore’ (as calculated in the HUMMUS dataset), or include specific restricted ingredients (e.g., ‘margarine’) or preparation step (e.g., ‘deep-fry’). \
            Details of our approach to fully utilize the HUMMUS dataset to build HumbleNutri are forthcoming in our paper.
    """, unsafe_allow_html=True
    )
st.markdown("##### **Model**", unsafe_allow_html=True)
st.markdown(
    """
       HumbleNutri recommends recipes to achieve the daily “meal bundle” structure using a hybrid model with a first stage involving recommendation models that choose meals fitting a user’s dietary preferences, followed by a second stage using optimization models to find meals that meet their clinical nutritional requirements.
    """
)

st.markdown("> **Dietary Recommender Model**", unsafe_allow_html=True)
st.markdown(
    """
        > The recommender model involves three independent recommender systems for HumbleNutri consists of three independent recommender systems for breakfast, appetizers, and main dishes. The side dishes are recommended from a subset of all appetizers to those having keywords representing vegetables (including salad) or grain and legume dishes.
    """
)
st.markdown("> **Nutritional Constraint Model**", unsafe_allow_html=True)
st.markdown(
    """
        > We then apply a linear programming model that is designed to meet a patient’s clinical nutritional needs while identifying recipes that are most closely aligned with their cuisine preferences. \
            This is achieved using an objective function that maximizes the recommendation score from the recommender system model while adhering to nutritional constraints defined by our Dietitian Nutritionist. \
                The set of nutritional constraints for any patient is automatically computed by the app based on the information input by the medical provider on each patient’s clinical factors (sex, age, weight, height, and whether they have certain health conditions or are recovering from surgery). \
                    These constraints define an individual’s daily macro and micronutrient needs. The linear programming model aims to identify combinations of recipes that taken together meet these daily nutrient needs, while also meeting a high score for meals that follow their dietary preferences. \
                    The full set of nutritional equations the constraints are based on are summarized in this [linked document](https://docs.google.com/document/d/1MBgeoLw5g78iKkGNSd6zqvn005TrKiXiz_iq-hjZK3E/edit). Details of this approach are also forthcoming in our paper.
    """, unsafe_allow_html=True
)

st.text("")

st.markdown(
    """
        <a id="note1" href="#note1ref"><sup>1</sup></a>The current installation of the app does not yet enable users to define their dietary preferences by reviewing example meals. In its place, the recommendation component of the model is based on example dietary preference profiles that we created from reviewers of the Food.com dataset. We selected an example user if they: i) Contributed a minimum of 30 reviews; ii) Reviewed at least one item each from breakfast, appetizers, and main dishes; iii) a significant portion of their reviews were dedicated to Latin American cuisine. We consider the top 25% recommended items for each user as candidate recipes.
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

st.markdown(
    """
        ### :star: Meet the team :star:
    """
)

col1, _ = st.columns(2)

col1.image("./images/AH.jpg", caption="PI: Abigail Horn, PhD                                                        ")
col1.markdown("""
              Research Assistant Professor, <br />
              Information Sciences Institute (ISI), <br />
              Viterbi School of Engineering <br />
              Co-Director, ISI AI4Health Center <br />
              - Overseeing project, ensuring algorithm meets clinical and cultural needs
              """, unsafe_allow_html=True)
col1.image("./images/SK.jpg", caption="Co-I: Susan Kim, MS, RDN, CCTD")
col1.markdown("""
              Dietician, Program Manager, <br />
              Abdominal Organ Transplant Program, <br />
              USC Transplant Institute, <br />
              Keck Medicine of USC <br />
              - Overseeing incorporation of clinical nutrition equations, patient needs
              """, unsafe_allow_html=True)
col1.image("./images/KB.jpg", caption="Co-I: Keith Burghardt, PhD")
col1.markdown("""
              Research Computer Scientist, <br />
              Information Sciences Institute (ISI), <br />
              Viterbi School of Engineering <br />
              - Advising development of algorithm, data science
              """, unsafe_allow_html=True)
col1.image("./images/AS.jpg", caption="Co-I: Alex DongHyeon Seo, MS")
col1.markdown("""
              Machine Learning Engineer, <br />
              Information Sciences Institute (ISI), <br />
              Viterbi School of Engineering <br />
              - Developed algorithm, leading data science, app development
              """, unsafe_allow_html=True)


# Initialize session state
if 'submitted' in st.session_state.keys():
    del st.session_state['submitted']