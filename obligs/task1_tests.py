"""Simple test comparing both implementations"""

from get_cond_prob_distribution import get_cond_prob_distribution as pandas_version
from get_cond_prob_distribution_original import get_cond_prob_distribution as original_version

# Test data
joint_probs = [
    ['ambulance', 'helicopter', 'other'],
    ['urgent', 0.11, 0.04, 0.03],
    ['non-urgent', 0.17, 0.01, 0.64]
]

def test_case(input_str, expected):
    """Test a single case and show results"""
    print(f"Input: '{input_str}'")
    print(f"Expected: {expected}")
    
    # Test original
    orig_result = original_version(joint_probs, input_str)
    print(f"Original: {orig_result}")
    
    # Test pandas  
    pandas_result = pandas_version(joint_probs, input_str)
    if hasattr(pandas_result, 'tolist'):
        pandas_result = pandas_result.tolist()
    print(f"Pandas:   {pandas_result}")
    
    # Check if they match
    match_orig = orig_result == expected
    match_pandas = pandas_result == expected
    both_match = orig_result == pandas_result
    
    if match_orig and match_pandas and both_match:
        print("✅ Pass")
    else:
        print("❌ Fail")
        if not match_orig:
            print(f"   Original failed: got {orig_result}, expected {expected}")
        if not match_pandas:
            print(f"   Pandas failed: got {pandas_result}, expected {expected}")
        if not both_match:
            print(f"   Implementations don't match: Original={orig_result}, Pandas={pandas_result}")
    print("-" * 40)

if __name__ == "__main__":
    print("Testing Conditional Probability Distributions\n")
    
    # Key test cases with expected results
    test_case("column #2", [0.8, 0.2])  # P(urgent|other), P(non-urgent|other)
    test_case("row urgent", [0.61, 0.22, 0.17])  # P(ambulance|urgent), P(helicopter|urgent), P(other|urgent)
    test_case("column helicopter", [0.8, 0.2])  # P(urgent|helicopter), P(non-urgent|helicopter)
    test_case("row non-urgent", [0.21, 0.01, 0.78])  # P(ambulance|non-urgent), P(helicopter|non-urgent), P(other|non-urgent)
