import matplotlib.pyplot as plt
import pandas as pd

def create_joint_prob_dataframe(joint_probs_list):
    """Convert nested list to proper DataFrame with labeled rows and columns."""
    # Extract headers and data
    columns = joint_probs_list[0]
    rows_data = joint_probs_list[1:]
    
    # Create DataFrame
    df = pd.DataFrame(
        data=[row[1:] for row in rows_data],  # numeric data only
        index=[row[0] for row in rows_data],  # row labels
        columns=columns                       # column labels
    )
    return df

def get_cond_prob_distribution(joint_probs, search_string):
    """
    Calculate conditional probability distribution using pandas. Generated using Claude Sonnet 4.0 from my original implemntation in ./get_cond_prob_distribution_original.py
    
    Args:
        joint_probs: Either nested list or pandas DataFrame
        search_string: Format examples:
            - "column #2" (by index)
            - "column helicopter" (by name)  
            - "row #1" (by index)
            - "row urgent" (by name)
    
    Returns:
        pandas Series with conditional probabilities
    """
    # Convert to DataFrame if needed
    if isinstance(joint_probs, list):
        df = create_joint_prob_dataframe(joint_probs)
    else:
        df = joint_probs.copy()
    
    # Parse search string
    parts = search_string.split()
    direction = parts[0]  # 'column' or 'row'
    identifier = parts[1] # either '#2' or 'helicopter'
    
    if direction == "column":
        # Get column conditional probabilities
        if identifier.startswith("#"):
            col_idx = int(identifier[1:])
            col_data = df.iloc[:, col_idx]
            col_name = df.columns[col_idx]
        else:
            col_data = df[identifier]
            col_name = identifier
        
        # Calculate P(row | column) = P(row, column) / P(column)
        total = col_data.sum()
        result = (col_data / total).round(2)
        result.name = f"P(row | {col_name})"
        
    elif direction == "row":
        # Get row conditional probabilities  
        if identifier.startswith("#"):
            row_idx = int(identifier[1:])
            row_data = df.iloc[row_idx, :]
            row_name = df.index[row_idx]
        else:
            row_data = df.loc[identifier]
            row_name = identifier
            
        # Calculate P(column | row) = P(row, column) / P(row)
        total = row_data.sum()
        result = (row_data / total).round(2)
        result.name = f"P(column | {row_name})"
    
    else:
        raise ValueError("Search string must start with 'column' or 'row'")
    
    return result
        

# Example usage and testing
if __name__ == "__main__":
    # Original data structure
    joint_probs = [
        ['ambulance', 'helicopter', 'other'],
        ['urgent', 0.11, 0.04, 0.03],
        ['non-urgent', 0.17, 0.01, 0.64]
    ]
    
    # Convert to DataFrame for better visualization
    df = create_joint_prob_dataframe(joint_probs)
    print("Joint Probability Table:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Test examples
    print("1. Column conditional probability by index:")
    result1 = get_cond_prob_distribution(joint_probs, "column #2")
    print(result1)
    print()
    
    print("2. Column conditional probability by name:")
    result2 = get_cond_prob_distribution(joint_probs, "column helicopter")
    print(result2)
    print()
    
    print("3. Row conditional probability by name:")
    result3 = get_cond_prob_distribution(joint_probs, "row urgent")
    print(result3)
    print()
    
    print("4. Row conditional probability by index:")
    result4 = get_cond_prob_distribution(joint_probs, "row #0")
    print(result4)
    
    # Optional: Load and display income data if file exists
    try:
        income_data = pd.read_csv('../extra_datasets/income_data.csv')
        print("\n" + "="*50)
        print("Income Data Preview:")
        print(income_data.head())
    except FileNotFoundError:
        print("\nIncome data file not found - skipping that part.")

