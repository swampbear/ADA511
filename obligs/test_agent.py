
"""Simple test for prob_tenth_patient_urgent"""

from optimal_ai_agent import prob_tenth_patient_urgent

def test_case(joint_probs, patient_sequence, expected):
    print(f"p-list: {joint_probs}")
    print(f"patient sequence: {patient_sequence}")
    print(f"Expected: {expected}%")

    result = prob_tenth_patient_urgent(joint_probs.copy(), patient_sequence)

    # Extract number from string
    prob_str = result.split("=")[-1].strip().replace("%","")
    try:
        prob_val = float(prob_str)
    except ValueError:
        prob_val = None

    print(f"Got: {result}")
    if prob_val is not None:
        match = round(prob_val, 1) == expected
    else:
        match = False

    if match:
        print("✅ Pass")
    else:
        print("❌ Fail")
        if prob_val is not None:
            print(f"   Expected: {expected}%, Got: {prob_val}%")
        else:
            print(f"   Could not parse result: {result}")
    print("-" * 40)


if __name__ == "__main__":
    print("Testing prob_tenth_patient_urgent\n")

    # Test cases from your screenshot
    test_case(
        [0.204, 0.013, 0.012, 0.112, 0.127, 0.095, 0.073, 0.209, 0.011, 0.064, 0.08],
        [1, 1, 0, 1, 1, 1, 0, 0, 1],
        83.4
    )

    test_case(
        [0.001, 0.01, 0.044, 0.117, 0.205, 0.246, 0.205, 0.117, 0.044, 0.01, 0.001],
        [1, 0, 0, 1, 0, 1, 1, 1, 0],
        50.0
    )

    test_case(
        [0.024, 0.218, 0.017, 0.151, 0.104, 0.208, 0.07, 0.003, 0.085, 0.101, 0.019],
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        62.5
    )

    test_case(
        [0.001, 0.004, 0.301, 0.28, 0.027, 0.026, 0.014, 0.074, 0.006, 0.246, 0.021],
        [0, 0, 0, 0, 0, 0,1, 0, 0],
        94.4
    )
