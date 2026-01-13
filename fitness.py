def calculate_fitness(schedule):
    """
    Higher fitness value indicates a better solution.
    Penalties are applied for room capacity violations and conflicts.
    """
    conflicts = 0
    capacity_penalty = 0

    for s in schedule:
        if s["num_students"] > s["capacity"]:
            capacity_penalty += (s["num_students"] - s["capacity"])

    fitness = 1 / (1 + conflicts + capacity_penalty)
    return fitness
