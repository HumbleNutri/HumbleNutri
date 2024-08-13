import re

def round_up_numbers_in_string(input_str):
    # Define a regular expression pattern to match numbers with more than 2 decimal places
    pattern = r'\d+\.\d{3,}'

    # Use re.sub() with a lambda function to round matched numbers
    rounded_str = re.sub(pattern, lambda match: '{:.2f}'.format(float(match.group(0))), input_str)

    return rounded_str

def torf(input_str):
    if input_str == 'True':
        output_str = 'T'
    else:
        output_str = 'F'
    
    return output_str

def get_user(user):
    user_mapping = {
        'User-1': 159,
        'User-2': 1738,
        'User-3': 3958,
        'User-4': 7933,
        'User-5': 25472
    }
    return user_mapping.get(user)