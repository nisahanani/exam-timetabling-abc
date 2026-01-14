def calculate_fitness(schedule, alpha=50, beta=5):
    """
    Fitness function aligned with project objectives:
    - Hard constraint: capacity violation
    - Soft constraint: wasted classroom capacity
    """

    capacity_violation = 0
    wasted_capacity = 0

    for s in schedule:
        students = s["num_students"]
        capacity = s["capacity"]

        if capacity < students:
            capacity_violation += (students - capacity)
        else:
            wasted_capacity += (capacity - students)

    total_cost = alpha * capacity_violation + beta * wasted_capacity

    # ABC maximizes fitness, so convert cost to fitness
    fitness = 1 / (1 + total_cost)

    return fitness
