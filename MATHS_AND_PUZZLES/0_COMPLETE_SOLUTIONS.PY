import random

# Q7: Policy of stopping as soon as 1 girl is born and each family needs to have at least 1 girl. Simulation to get the gender ratio.
def get_gender_ratio_n_family(n):
    girls = 0
    boys = 0
    for _ in range(1, n + 1):
        l = get_gender_ratio_1_family()
        girls += l[0]
        boys += l[1]
    return girls / (girls + boys)
def get_gender_ratio_1_family():
    girls = 0
    boys = 0
    result = []
    while girls == 0:
        rand = random.randint(1,10)
        if rand % 2 == 0:
            girls += 1
        else:
            boys += 1
    result.append(girls)
    result.append(boys)
    return result
# print(get_gender_ratio_n_family(4500))

# Q8: Dropping Egg Problem Simulation:
# Bldg with 100 Floors. If an egg drops from a particular floor, it breaks. Below this floor, it doesn't.
# You have been given 2 eggs, find a value of N which will minimize the number of drops to find the breaking point in worst case.
# Simulation will show the number of drop for each case. Problem solving will give solution for N as N(N + 1) / 2 = 100 => N = 13.65
# Round up higher [explain this as well] and N = 14
def get_number_of_drops(break_floor, total_floors):
    result = [0]
    interval = 14
    prev_floor = 0
    egg1 = interval
    while not drop(egg1, break_floor, result) and egg1 <= total_floors:
        interval -= 1
        prev_floor = egg1
        egg1 += interval
    egg2 = prev_floor + 1
    while not drop(egg2, break_floor, result) and egg2 < egg1 and egg2 <= total_floors:
        egg2 += 1
    return result[0]
def drop(floor_num, break_floor, result):
    result[0] += 1
    return floor_num >= break_floor 
# for r in [1,3,5,10, 14, 20, 30, 35, 50, 70, 90, 95, 99, 100, 101]:
#     print('{}:{}'.format(r, get_number_of_drops(r, 100)))


