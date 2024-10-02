import re

def round_up_numbers_in_string(input_str):
    # Define a regular expression pattern to match numbers with more than 2 decimal places
    pattern = r'\d+\.\d{3,}'

    # Use re.sub() with a lambda function to round matched numbers
    rounded_str = re.sub(pattern, lambda match: '{:.2f}'.format(float(match.group(0))), input_str)

    return rounded_str

def torf(input_str):
    if input_str == True:
        output_str = 'T'
    else:
        output_str = 'F'
    
    return output_str

def get_(x):
    _mapping = {
        "Breakfast": "breakfast",
        "Lunch": "lunch",
        "Lunch-Side": "lunch-side",
        "Dinner-Main": "dinner-main",
        "Dinner-Side (whole-grains)": "dinner-side-wg",
        "Dinner-Side (vegetables)":"dinner-side-vg"
    }
    return _mapping.get(x)

def change_order(lst):
    # Move Lunch after breakfast
    lunch = lst.pop(2)
    lst.insert(1, lunch)

    # Move Dinner after lunch-side
    dinner = lst.pop(5)
    lst.insert(3, dinner)

    return lst

def get_user(user):
    user_mapping = {
        'User-1': 159,
        'User-2': 1738,
        'User-3': 3958,
        'User-4': 7933,
        'User-5': 26472
    }
    return user_mapping.get(user)

def round_up_numbers_in_string(input_str):
    # Define a regular expression pattern to match numbers with more than 2 decimal places
    pattern = r'\d+\.\d{3,}'

    # Use re.sub() with a lambda function to round matched numbers
    rounded_str = re.sub(pattern, lambda match: '{:.2f}'.format(float(match.group(0))), input_str)

    return rounded_str

def get_obesity_status(bmi):
    if bmi < 18.5:
        obesity_status = "Underweight"
        weight_goals = "Gain"
    elif bmi >= 18.5 and bmi < 25:
        obesity_status = "Normal weight"
        weight_goals = "Maintain"
    elif bmi >= 25 and bmi < 30:
        obesity_status = "Overweight"
        weight_goals = "Lose"
    elif bmi >= 30 and bmi < 35:
        obesity_status = "Obesity Class I"
        weight_goals = "Lose"
    elif bmi >= 35 and bmi < 40:
        obesity_status = "Obesity Class II"
        weight_goals = "Lose"
    elif bmi >= 40:
        obesity_status = "Obesity Class III"
        weight_goals = "Lose"
    
    return obesity_status, weight_goals