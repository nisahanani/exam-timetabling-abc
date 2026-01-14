import random
from fitness import calculate_fitness

def generate_solution(exams, rooms):
    solution = []
    for exam in exams:
        room = random.choice(rooms)
        solution.append({
            "course_code": exam["course_code"],
            "num_students": exam["num_students"],
            "exam_day": exam["exam_day"],
            "exam_time": exam["exam_time"],
            "building_name": room["building_name"],
            "room_number": room["room_number"],
            "capacity": room["capacity"]
        })
    return solution


def artificial_bee_colony(exams, rooms, num_bees, max_iter, scout_limit):
    population = [generate_solution(exams, rooms) for _ in range(num_bees)]
   fitness_values = []
cost_values = []

for sol in population:
    fitness, cost = calculate_fitness(sol)
    fitness_values.append(fitness)
    cost_values.append(cost)

    best_index = cost_values.index(min(cost_values))
    best_solution = population[best_index]
    best_cost = cost_values[best_index]

    convergence_curve = []
    trial_counter = [0] * num_bees

    for iteration in range(max_iter):
        for i in range(num_bees):
            candidate = generate_solution(exams, rooms)
            candidate_fitness, candidate_cost = calculate_fitness(candidate)

            if candidate_fitness > fitness_values[i]:
                population[i] = candidate
                fitness_values[i] = candidate_fitness
                cost value[i] = candidate_cost
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Scout bees
        for i in range(num_bees):
            if trial_counter[i] > scout_limit:
                population[i] = generate_solution(exams, rooms)
                fitness_values[i] = calculate_fitness(population[i])
                fitness_values[i]= fitness
                cost_values[i] = cost
                trial_counter[i] = 0

        current_best = max(fitness_values)
        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[fitness_values.index(current_best)]

        
        convergence_curve.append(best_cost)

    return best_solution, best_cost, convergence_curve

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
