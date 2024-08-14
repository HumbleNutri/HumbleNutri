import random
import pulp
from constraints import IBW_constraints
from utils import torf
import streamlit as st
import re
import subprocess

def sanitize_variable_name(name):
    # Replace or remove illegal characters
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

class CustomCBCSolver(pulp.LpSolver):
    def __init__(self, solver_path="/usr/bin/cbc"):
        super().__init__()
        self.solver_path = solver_path

    def actualSolve(self, lp):
        # Write the problem to an LP file
        lp.writeLP("temp_problem.lp")

        # Run the CBC solver manually, without text=True to get bytes
        result = subprocess.run([self.solver_path, 'temp_problem.lp', 'solve'], capture_output=True)
        
        if result.returncode != 0:
            raise pulp.PulpSolverError("Error running CBC solver")
        
        # Ensure the output is correctly processed
        objective_value = None

        for line in result.stdout.splitlines():
            # Debugging: Ensure that each line is a string
            if isinstance(line, str):
                if "Objective value:" in line:
                    # Extract the objective value from the line
                    try:
                        objective_value = float(line.split(":")[1].strip())
                    except ValueError as e:
                        st.write(f"Error parsing objective value from line: {line}")
                        raise e
            else:
                st.write(f"Unexpected line type: {type(line)} - Content: {line}")

        # Set the objective value manually in the PuLP model
        if objective_value is not None:
            lp.objective.setInitialValue(objective_value)
        else:
            return pulp.constants.LpStatusNotSolved

        # Manually assign variable values if possible
        for v in lp.variables():
            v.varValue = 1.0  # Replace with actual logic to assign correct values
        # # Decode the output safely
        # try:
        #     stdout_decoded = result.stdout.decode('utf-8', errors='replace')
        #     stderr_decoded = result.stderr.decode('utf-8', errors='replace')
        # except Exception as e:
        #     raise RuntimeError(f"Error decoding CBC output: {e}")

        #st.write(result.stdout)

        # # Print output for debugging
        # st.write("CBC Solver Output:", stdout_decoded)
        # st.write("CBC Solver Error Output:", stderr_decoded)

        # # Optional: Clean up temporary file
        # os.remove("temp_problem.lp")

        # Assuming further processing or manual parsing here
        return pulp.constants.LpStatusOptimal


def LP_MealBundle(bf_items, wg_items, vg_items, main_items, gender, height, weight, age, after_surgery, activity_level, pre_diabetes, high_cholesterol, hypertension):
    # # Find the solver path using a shell command
    # solver_path = subprocess.run(['which', 'cbc'], capture_output=True, text=True).stdout.strip()
    # if not solver_path:
    #     raise FileNotFoundError("Solver not found. Please ensure it's installed and available in the PATH.")
    # set seed
    random.seed(2024)
    # Get constraints
    calorie_needs, protein_needs_lower, protein_needs_upper,sugar_needs, carb_needs, satfat_needs, sodium_needs, fiber_needs = IBW_constraints(gender, height, weight, age,
                                                                                                                                               torf(after_surgery), activity_level, torf(pre_diabetes),
                                                                                                                                               torf(high_cholesterol), torf(hypertension))
    # Define the problem
    prob = pulp.LpProblem("Bundle_Generation", pulp.LpMaximize)

    # Define decision variables for item choices
    item_choices = pulp.LpVariable.dicts("ItemChoice", [(meal_type, sanitize_variable_name(item[0])) for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                                                         for item in (bf_items if meal_type == 'breakfast' else
                                                                      vg_items if meal_type == 'lunch-side' else
                                                                      main_items if meal_type == 'lunch' else
                                                                      wg_items if meal_type == 'dinner-side-wg' else
                                                                      vg_items if meal_type == 'dinner-side-vg' else
                                                                      main_items)], cat='Binary')

    # Define the objective function
    # Maximize the recommendation score
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[22] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)])
                           
    #Define constraints
    # Exactly one item per meal type for each bundle
    for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']:
        prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] for item in (bf_items if meal_type == 'breakfast' else
                                                                   vg_items if meal_type == 'lunch-side' else
                                                                   main_items if meal_type == 'lunch' else
                                                                   wg_items if meal_type == 'dinner-side-wg' else
                                                                   vg_items if meal_type == 'dinner-side-vg' else
                                                                   main_items)]) == 1

    # Calorie constraint (maximum 2250 calories for all meals)
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[11] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= calorie_needs

    # Carbohydrate constraint 
    # Get amount per patient as % of overall calories: should be 50-60% # 4 cal / 1 g carbohydrate
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[16] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= carb_needs

    # Protein constraint IBW
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[17] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) >= protein_needs_lower
    # Protein constraint IBW
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[17] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= protein_needs_upper
    
    # Fiber constraint (minimum 25 for entire bundle) 
    # Amount per patient: should be 25-35 g
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[18] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) >= fiber_needs

    # Total Fat constraint
    #  <30% of kcals # 9 cal / 1 g fat 
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[19] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= calorie_needs*0.3/9
    
    # Saturated Fat constraint
    #  <10% of kcals # 9 cal / 1 g fat 
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[12] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= satfat_needs
    
    # Sugar constraints
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[13] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= sugar_needs
    
    # Sodium constraint
    #   <2300 mg
    prob += sum([item_choices[meal_type, sanitize_variable_name(item[0])] * item[15] for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']
                 for item in (bf_items if meal_type == 'breakfast' else
                              vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                              main_items)]) <= sodium_needs


    # Duration constraint (maximum 35 minutes for bf)(maximum 45 minutes for lunch) 
    # (maximum 30 minutes for side dish)(maximum 120 minutes for dinner main)
    prob += sum([item_choices['breakfast',sanitize_variable_name(item[0])] * item[2] for item in bf_items]) <= 20
    prob += sum([item_choices['lunch-side', sanitize_variable_name(item[0])] * item[2] for item in vg_items]) <= 15
    prob += sum([item_choices['lunch', sanitize_variable_name(item[0])] * item[2] for item in main_items]) <= 45
    prob += sum([item_choices['dinner-side-wg', sanitize_variable_name(item[0])] * item[2] for item in wg_items]) <= 15
    prob += sum([item_choices['dinner-side-vg', sanitize_variable_name(item[0])] * item[2] for item in vg_items]) <= 15
    prob += sum([item_choices['dinner-main', sanitize_variable_name(item[0])] * item[2] for item in main_items]) <= 100

    # Constraint to prevent selecting the same item for lunch and dinner
    for lunch_item in main_items:
        for dinner_item in main_items:
            if sanitize_variable_name(lunch_item[0]) == sanitize_variable_name(dinner_item[0]):
                prob += item_choices['lunch', sanitize_variable_name(lunch_item[0])] + item_choices['dinner-main', sanitize_variable_name(dinner_item[0])] <= 1
                
    # Constraint to prevent selecting the same item for lunch-side and dinner-side
    for lunch_side_item in vg_items:
        for dinner_side_item in vg_items:
            if sanitize_variable_name(lunch_side_item[0]) == sanitize_variable_name(dinner_side_item[0]):
                prob += item_choices['lunch-side', sanitize_variable_name(lunch_side_item[0])] + item_choices['dinner-side-vg', sanitize_variable_name(dinner_side_item[0])] <= 1

    # List to store bundles
    bundles = []
    # List to store previously selected items
    selected_items =  {
    'breakfast': set(),
    'lunch-side': set(),
    'lunch': set(),
    'dinner-side-wg': set(),
    'dinner-side-vg': set(),
    'dinner-main': set()
    }

    # Solve the problem iteratively # Generate 10 bundles
    while len(bundles)< 50:
        solver = CustomCBCSolver() #pulp.PULP_CBC_CMD(msg=True)
        status=prob.solve(solver)
        if status == pulp.LpStatusOptimal:
            st.write("Objective Value:", pulp.value(prob.objective))
            for v in prob.variables():
                st.write(f"{v.name} = {v.varValue}")
        if prob.status == pulp.LpStatusOptimal:
            # Create a bundle from the optimal solution
            bundle = {
                'breakfast': None,
                'lunch-side': None,
                'lunch': None,
                'dinner-side-wg': None,
                'dinner-side-vg': None,
                'dinner-main': None,
                'objective_value': pulp.value(prob.objective)
            }
            for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']:
                for item in (bf_items if meal_type == 'breakfast' else
                             vg_items if meal_type == 'lunch-side' else
                              main_items if meal_type == 'lunch' else
                              wg_items if meal_type == 'dinner-side-wg' else
                              vg_items if meal_type == 'dinner-side-vg' else
                             main_items):
                    if pulp.value(item_choices[meal_type, sanitize_variable_name(item[0])]) == 1:
                        bundle[meal_type] = item
            bundles.append(bundle)

            # Add constraints to exclude the selected items from future bundles
            for meal_type in ['breakfast', 'lunch-side','lunch', 'dinner-side-wg', 'dinner-side-vg', 'dinner-main']:
                if bundle[meal_type] is not None:
                    selected_items[meal_type].add(bundle[meal_type][0])
                    prob += item_choices[meal_type, bundle[meal_type][0]] == 0
    
            # Add constraints to exclude selected main-dish items from being used as lunch or dinner-main in future bundles
            for item_id in selected_items['lunch'] | selected_items['dinner-main']:
                prob += item_choices['lunch', item_id] + item_choices['dinner-main', item_id] == 0
                
            # Add constraints to exclude selected vegie-dish items from being used as lunch-side or dinner-side-vg in future bundles
            for item_id in selected_items['lunch-side'] | selected_items['dinner-side-vg']:
                prob += item_choices['lunch-side', item_id] + item_choices['dinner-side-vg', item_id] == 0
        else:
            break

    return bundles, selected_items