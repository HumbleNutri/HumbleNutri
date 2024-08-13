def calc_bmi(height, weight):
    bmi = weight / (height/100)**2
    
    return bmi

def get_ibw(gender, height, weight, bmi):
    # Get input as (kg) convert to lbs
    lbs = weight * 2.20462
    # Get input as (cm) covert to inches
    inch = height * 0.393701
    # 5 feet = 60inch
    if gender == 'Male':
        ibw = 106 + (6 *(inch-60))
    else:
        ibw = 100 + (5 *(inch-60))
        
    # If obese, get adjusted ibw
    if bmi >= 30:
        ibw_adj = ibw + 0.25*(lbs - ibw)
    else:
        ibw_adj = ibw # Not obese, use ibw without adjusting
        
    ibw_in_kg = ibw_adj * 0.453592
    
    return ibw_in_kg

def IBW_constraints(gender, height, weight, age, after_surgery, activity_level, pre_diabetes, high_cholesterol, hypertension):
    # Get IBW (Ideal Body Weight)
    bmi = calc_bmi(height, weight)
    ibw_in_kg = get_ibw(gender, height, weight, bmi)
    # Get RMR (in kg and centimeters)
    if gender == 'Male':
        RMR = (9.99*ibw_in_kg) + (6.25*height) - (4.92*age) + 5
        if age > 50:
            fiber_needs = 28
        else:
            fiber_needs = 31
    else:
        RMR = (9.99*ibw_in_kg) + (6.25*height) - (4.92*age) - 161
        if age > 50:
            fiber_needs = 22
        else:
            fiber_needs = 25
    # Acrtivitiy level
    if activity_level == 'Sedentary':
        eps = 1.2
        protein_needs_lower = 0.6
        protein_needs_upper = ibw_in_kg * 0.8
    elif activity_level == 'Lightly active':
        eps = 1.375
        protein_needs_lower = ibw_in_kg * 0.8
        protein_needs_upper = ibw_in_kg * 1.0
    elif activity_level == 'Moderately active':
        eps = 1.55
        protein_needs_lower = ibw_in_kg * 1.0
        protein_needs_upper = ibw_in_kg * 1.2
    elif activity_level == 'Active':
        eps = 1.725
        protein_needs_lower = ibw_in_kg * 1.2
        protein_needs_upper = ibw_in_kg * 1.6
    elif activity_level == 'Very active':
        eps = 1.9
        protein_needs_lower = ibw_in_kg * 1.6
        protein_needs_upper = ibw_in_kg * 1.8
    else:
        print('Choose between Sedentary, Lightly active, Moderately active, Active, Very active')
        return None
    # Get final calorie needs
    calorie_needs = RMR * eps
    
    # Protien needs
    if after_surgery == 'T' or age >= 65:
        protein_needs_lower = ibw_in_kg * 1.2
        protein_needs_upper = ibw_in_kg * 1.5
    
    # Pre-diabetes
    if pre_diabetes == 'T':
        sugar_needs = calorie_needs*0.05/4 # 5% of total calories
        carb_needs = calorie_needs*0.45/4 # 45% of total calories
    else:
        sugar_needs = calorie_needs*0.1/4 # 10% of total calories
        carb_needs = calorie_needs*0.6/4 # 60% of total calories
    
    # High cholesterol
    if high_cholesterol == 'T':
        satfat_needs = calorie_needs*0.06/9 # 6% of total calories
    else:
        satfat_needs = calorie_needs*0.1/9 # 10% of total calories
    
    # Hypertension
    if hypertension == 'T':
        sodium_needs = 1500
    else:
        sodium_needs = 2300
    
    return calorie_needs, protein_needs_lower, protein_needs_upper, sugar_needs, carb_needs, satfat_needs, sodium_needs, fiber_needs