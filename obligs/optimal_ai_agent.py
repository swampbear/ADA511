import math

from get_cond_prob_distribution_original import get_cond_prob_distribution



def prob_tenth_patient_urgent(p_list, patient_sequence):
    n = sum(patient_sequence)
    print(sum(p_list))
    new_n = n+1
    seq_prob = f(p_list, n)
    seqand10_prob = f(p_list, new_n)
    prob_10th_urgent = seqand10_prob/(seq_prob+seqand10_prob)

    return f'P("10th patient is urgent" | "seen data") ={round(prob_10th_urgent*100, 1)}'
    

def bayes_theroem():
    pass

def f(joint_probs,i):
    probability_i = joint_probs[i] * (math.factorial(i)*(math.factorial(10-i)))/math.factorial(10)
    return probability_i
