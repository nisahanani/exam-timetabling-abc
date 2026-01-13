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
    fitness_values = [calculate_fitness(sol) for sol in population]

    best_solution = population[fitness_values.index(max(fitness_values))]
    best_fitness = max(fitness_values)

    convergence_curve = []
    trial_counter = [0] * num_bees

    for iteration in range(max_iter):
        for i in range(num_bees):
            candidate = generate_solution(exams, rooms)
            candidate_fitness = calculate_fitness(candidate)

            if candidate_fitness > fitness_values[i]:
                population[i] = candidate
                fitness_values[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Scout bees
        for i in range(num_bees):
            if trial_counter[i] > scout_limit:
                population[i] = generate_solution(exams, rooms)
                fitness_values[i] = calculate_fitness(population[i])
                trial_counter[i] = 0

        current_best = max(fitness_values)
        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[fitness_values.index(current_best)]

        # ⬅️ INI buat graf BERPERINGKAT
        convergence_curve.append(best_fitness)

    return best_solution, best_fitness, convergence_curve
