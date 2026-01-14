def calculate_cost(schedule):
    capacity_violation = 0
    wasted_capacity = 0

    for s in schedule:
        if s["num_students"] > s["capacity"]:
            capacity_violation += (s["num_students"] - s["capacity"])
        else:
            wasted_capacity += (s["capacity"] - s["num_students"])

    # Objectives
    alpha = 50
    beta = 5
    total_cost = alpha * capacity_violation + beta * wasted_capacity
    return total_cost


def calculate_fitness(schedule):
    cost = calculate_cost(schedule)
    fitness = 1 / (1 + cost)
    return fitness, cost
