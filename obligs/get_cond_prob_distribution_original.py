import matplotlib.pyplot as plt
import pandas as pd
"""
    I first calculated the conditional probability distribution from the bottom without pandas or numpy.
    Then when I was able to produce the wanted results i used this as input to generate a improved version using pandas.
    see ./get_cond_prob_distribution.py

    NB!! The printing format of this file is also generated using genai
"""

def solve_for_numbers(joint_probs, search_string):
    number = int(search_string.split("#")[1])
    total = 0
    result = []
    if search_string.startswith("column"):
        for row in joint_probs:
            if isinstance(row[number], float):
                total += row[number]
        for row in joint_probs:
                if isinstance(row[number], float):    
                    result.append(round(row[number]/total,2))
    if search_string.startswith("row"):
        total += sum(joint_probs[number][1:])
        for col in joint_probs[number]:
            if isinstance(col, float):
                result.append(round(col/total,2))
    return result 

def get_value_name_index(joint_probs, search_string):
    value = search_string.split(' ')[1]
    if search_string.startswith("column"):
        for i in range(0,len(joint_probs[0])):
            if joint_probs[0][i] == value:
                # adding a one here to return the proper col number
                return i+1
    if search_string.startswith("row"):
        for i in range(0,len(joint_probs)):
           if joint_probs[i][0] == value:
               # no need to add one here
               return i

def get_cond_prob_distribution(joint_probs, search_string):
    if '#' in search_string:
        print("Using function")
        distribution = solve_for_numbers(joint_probs, search_string)
        return distribution
    else:
        print('gettin index')
        index = get_value_name_index(joint_probs, search_string)
        new_search_string = " #".join([search_string.split(' ')[0], str(index)])
        print(new_search_string)
        distribution = solve_for_numbers(joint_probs, new_search_string)
        return distribution
        
if __name__ == "__main__":
    joint_probs = [['ambulance', 'helicopter', 'other'],['urgent',0.11,0.04,0.03],['non-urgent',0.17,0.01,0.64]]

    # Test the original implementation
    print("=== Original Implementation ===")
    print("Joint probability data structure:")
    for row in joint_probs:
        print(row)
        print()

    try:
        income_data = pd.read_csv('../extra_datasets/income_data.csv')
        print("Income data loaded successfully:")
        print(income_data.head())
        print()
    except FileNotFoundError:
        print("Income data file not found - skipping that part.")
        print()

    print("Testing column #2:")
    r = get_cond_prob_distribution(joint_probs, "column #2")
    print("Result:", r)
    print()
    print("Testing column helicopter")
    rHeli = get_cond_prob_distribution(joint_probs, "column helicopter")
    print("Result", rHeli)
    print()

    print("Testing row urgent:")
    r2 = get_cond_prob_distribution(joint_probs, "row urgent")
    print("Result:", r2)
    print()

    # Additional test cases to demonstrate functionality
    print("Testing column #0 (ambulance):")
    r3 = get_cond_prob_distribution(joint_probs, "column #1")
    print("Result:", r3)
    print()

    print("Testing row non-urgent:")
    r4 = get_cond_prob_distribution(joint_probs, "row non-urgent")
    print("Result:", r4)
