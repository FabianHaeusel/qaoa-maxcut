import numpy as np
from util import bcolors
from numpy import random

class Knapsack:
    items_values = {
    0: 21,
    1: 35,
    2: 24,
    3: 18,
    4: 39,
    5: 13,
    6: 30,
    7: 10
    }
    items_weights = {
    0: 15,
    1: 24,
    2: 39,
    3: 29,
    4: 50,
    5: 4,
    6: 45,
    7: 36
    }
    max_cap = 100

    def abs(self):
        pass


# brute force solution
def optimize_knapsack_bruteforce():
    n = len(Knapsack.items_values)
    print("\n==== knapsack bruteforce ====")
    print("items: ", n)
    print("iterations: ", 2**n)

    max_value = -1
    optimal_config = None

    for i in range(2**n):
        bitstring = np.binary_repr(i, n)

        items = list(Knapsack.items_values.keys())
        chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
        chosen_items = [items[i] for i in chosen_indices]
        total_value = sum([Knapsack.items_values[i] for i in chosen_items])
        total_weight = sum([Knapsack.items_weights[i] for i in chosen_items])

        # print("bitstring ", binstring)
        # print("chosen_indices: ", chosen_indices)
        # print("items: ", chosen_items)
        # print("val: ", total_value)
        # print("weight: " , total_weight)
        # print("")

        if(total_weight > Knapsack.max_cap):
            continue

        if(total_value > max_value):
            optimal_config = {
                "indices": chosen_indices,
                "items": chosen_items,
                "value": total_value,
                "weight": total_weight,
                "bitstring": bitstring
            }
            max_value = total_value

    print("The optimal configuration is: ", optimal_config)
    print("Bitstring: ", optimal_config["bitstring"])
    print("\n")
    return optimal_config


def is_bitstring_valid(bitstring):
    if len(bitstring) != len(Knapsack.items_values):
        raise ValueError("Invalid bitstring of length ", len(bitstring))
    items = list(Knapsack.items_values.keys())
    chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
    chosen_items = [items[i] for i in chosen_indices]
    total_weight = sum([Knapsack.items_weights[i] for i in chosen_items])

    return total_weight <= Knapsack.max_cap


def get_bitstring_knapsack_score(bitstring, optimal_config, log=True):

    if len(bitstring) != len(Knapsack.items_values):
        raise ValueError("Invalid bitstring of length ", len(bitstring))

    bitstring = bitstring[::-1]

    items = list(Knapsack.items_values.keys())
    chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
    chosen_items = [items[i] for i in chosen_indices]
    total_value = sum([Knapsack.items_values[i] for i in chosen_items])
    total_weight = sum([Knapsack.items_weights[i] for i in chosen_items])

    if log:
        print("max cap (max weight): " , Knapsack.max_cap)
        print("Value for bitstring ", bitstring, " is: " , total_value, " (weight: ", total_weight, ")")
        print("[Bruteforce] Optimal bitstring: ", optimal_config["bitstring"], " with value: " , optimal_config["value"], " (weight: ", optimal_config["weight"], ")")

    if total_weight > Knapsack.max_cap:
        if log: print(bcolors.ERROR, "The bitstring resulting from QAOA is not a valid solution!", bcolors.ENDC)
        return 0

    return total_value / optimal_config["value"]

def generate_random_problem(item_count):
    items_values = {}
    items_weights = {}
    max = 50
    for i in range(item_count):
        items_values[i] = random.randint(1, 50)
        items_weights[i] = random.randint(1, 50)

    weights_sum = sum(items_weights[i] for i in range(item_count))
    max_cap = min(250, weights_sum / 2)

    Knapsack.items_values = items_values
    Knapsack.items_weights = items_weights
    Knapsack.max_cap = max_cap
    return items_values, items_weights, max_cap

def use_tutorial_problem():
    Knapsack.items_values = {
        "football": 8,
        "laptop": 47,
        "camera": 10,
        "books": 5,
        "guitar": 16
    }
    Knapsack.items_weights = {
        "football": 3,
        "laptop": 11,
        "camera": 14,
        "books": 19,
        "guitar": 5
    }
    Knapsack.max_cap = 26