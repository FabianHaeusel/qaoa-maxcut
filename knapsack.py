import numpy as np
from util import bcolors

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


# brute force solution
def optimize_knapsack_bruteforce():
    n = len(items_values)
    print("\n==== knapsack bruteforce ====")
    print("items: ", n)
    print("iterations: ", 2**n)

    max_value = -1
    optimal_config = None

    for i in range(2**n):
        bitstring = np.binary_repr(i, n)

        items = list(items_values.keys())
        chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
        chosen_items = [items[i] for i in chosen_indices]
        total_value = sum([items_values[i] for i in chosen_items])
        total_weight = sum([items_weights[i] for i in chosen_items])

        # print("bitstring ", binstring)
        # print("chosen_indices: ", chosen_indices)
        # print("items: ", chosen_items)
        # print("val: ", total_value)
        # print("weight: " , total_weight)
        # print("")

        if(total_weight > max_cap):
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
    if len(bitstring) != len(items_values):
        raise ValueError("Invalid bitstring of length ", len(bitstring))
    items = list(items_values.keys())
    chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
    chosen_items = [items[i] for i in chosen_indices]
    total_weight = sum([items_weights[i] for i in chosen_items])

    return total_weight <= max_cap


def get_bitstring_knapsack_score(bitstring, optimal_config):

    if len(bitstring) != len(items_values):
        raise ValueError("Invalid bitstring of length ", len(bitstring))

    items = list(items_values.keys())
    chosen_indices = [j for j, i in enumerate(bitstring) if i == "1"]
    chosen_items = [items[i] for i in chosen_indices]
    total_value = sum([items_values[i] for i in chosen_items])
    total_weight = sum([items_weights[i] for i in chosen_items])

    print("max cap (max weight): " , max_cap)
    print("Value for bitstring ", bitstring, " is: " , total_value, " (weight: ", total_weight, ")")
    print("[Bruteforce] Optimal value for bitstring ", optimal_config["bitstring"], " is: " , optimal_config["value"], " (weight: ", optimal_config["weight"], ")")

    if total_weight > max_cap:
        print(bcolors.ERROR, "The bitstring resulting from QAOA is not a valid solution!", bcolors.ENDC)
        return 0

    return total_value / optimal_config["value"]
