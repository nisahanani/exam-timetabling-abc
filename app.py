# University Exam Scheduling using ABC (Artificial Bee Colony) and Streamlit

import streamlit as st
import pandas as pd
import random
import time

st.set_page_config(page_title="Exam Scheduling with ABC", layout="wide")
st.title("University Exam Scheduling using ABC (Artificial Bee Colony)")

# -------------------- Load Datasets (Direct Paths) --------------------
st.subheader("Exam and Classroom Dataset")

# Set your full file paths here
exam_file = "exam_timeslot.csv"
room_file = "classrooms.csv"

# Load exam dataset
try:
    exams_df = pd.read_csv(exam_file)
    st.success("Exam dataset loaded successfully!")
    st.dataframe(exams_df)
except Exception as e:
    st.error(f"Failed to load exam dataset: {e}")
    st.stop()

# Load classroom dataset
try:
    rooms_df = pd.read_csv(room_file)
    st.success("Classroom dataset loaded successfully!")
    st.dataframe(rooms_df)
except Exception as e:
    st.error(f"Failed to load classroom dataset: {e}")
    st.stop()

# -------------------- Convert Datasets --------------------
exam_list = exams_df.to_dict('records')
room_list = rooms_df.to_dict('records')

# -------------------- Cost and Fitness Functions --------------------
def calculate_cost(schedule, alpha=50, beta=5):
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
    return total_cost

def calculate_fitness(schedule, alpha=50, beta=5):
    cost = calculate_cost(schedule, alpha, beta)
    fitness = 1 / (1 + cost)
    return fitness, cost

# -------------------- ABC Helper Functions --------------------
def generate_solution(exams, rooms):
    solution = []
    for exam in exams:
        room = random.choice(rooms)
        solution.append({
            "course_code": exam["exam_id"],
            "num_students": exam["num_students"],
            "exam_day": exam.get("exam_day", 1),
            "exam_time": exam.get("exam_time", "09:00"),
            "building_name": room.get("building_name", "BuildingA"),
            "room_number": room["classroom_id"],
            "capacity": room["capacity"]
        })
    return solution

# -------------------- ABC Algorithm --------------------
def artificial_bee_colony(exams, rooms, num_bees, max_iter, scout_limit, alpha=50, beta=5):
    population = [generate_solution(exams, rooms) for _ in range(num_bees)]
    fitness_values = []
    cost_values = []

    for sol in population:
        fitness, cost = calculate_fitness(sol, alpha, beta)
        fitness_values.append(fitness)
        cost_values.append(cost)

    best_index = cost_values.index(min(cost_values))
    best_solution = population[best_index]
    best_cost = cost_values[best_index]

    convergence_curve = []
    trial_counter = [0] * num_bees
    start_time = time.time()

    for iteration in range(max_iter):
        for i in range(num_bees):
            candidate = generate_solution(exams, rooms)
            candidate_fitness, candidate_cost = calculate_fitness(candidate, alpha, beta)
            if candidate_cost < cost_values[i]:
                population[i] = candidate
                fitness_values[i] = candidate_fitness
                cost_values[i] = candidate_cost
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        for i in range(num_bees):
            if trial_counter[i] > scout_limit:
                population[i] = generate_solution(exams, rooms)
                fitness, cost = calculate_fitness(population[i], alpha, beta)
                fitness_values[i] = fitness
                cost_values[i] = cost
                trial_counter[i] = 0
    current_best_index = cost_values.index(min(cost_values))
    if cost_values[current_best_index] < best_cost:
            best_cost = cost_values[current_best_index]
            best_solution = population[current_best_index]

        convergence_curve.append(best_cost)

        if (iteration + 1) % max(1, (max_iter // 10)) == 0:
            st.write(f"Iteration {iteration+1}/{max_iter} | Current Best Cost: {best_cost}")

    elapsed = time.time() - start_time
    st.write(f"ABC completed in {elapsed:.2f} seconds")
    st.write(f"Final Best Cost: {best_cost}")

    return best_solution, best_cost, convergence_curve

# -------------------- Streamlit Interface --------------------
st.subheader("ABC Parameters")
col1, col2 = st.columns(2)

with col1:
    num_bees = st.number_input("Number of Bees", 5, 50, 10)
    max_iter = st.number_input("Maximum Iterations", 10, 500, 100)
    scout_limit = st.number_input("Scout Limit", 5, 50, 10)

with col2:
    alpha = st.slider("Weight Capacity Violation (α)", 10, 100, 50)
    beta = st.slider("Weight Wasted Capacity (β)", 1, 20, 5)

# -------------------- Run ABC --------------------
if st.button("Run ABC"):
    best_schedule, best_cost, convergence = artificial_bee_colony(
        exam_list, room_list, num_bees, max_iter, scout_limit, alpha, beta
    )

    # Display final schedule
    result_df = pd.DataFrame([
        {
            "Exam ID": s["course_code"],
            "Students": s["num_students"],
            "Classroom": s["room_number"],
            "Room Capacity": s["capacity"]
        } for s in best_schedule
    ])
    st.subheader("Final Exam Schedule")
    st.dataframe(result_df)

    # Save final schedule
    output_csv = "C:/Users/ASUS/Documents/ce project/abc_exam_schedule.csv"
    result_df.to_csv(output_csv, index=False)
    st.success(f"Final schedule saved to {output_csv}")

    # Plot convergence (final cost)
    st.subheader("Convergence Curve (Best Cost)")
    st.line_chart(convergence)

st.info(
    "This project demonstrates multi-objective optimization in exam scheduling "
    "by balancing capacity violations (hard constraint) and wasted classroom capacity (soft constraint) "
    "using ABC (Artificial Bee Colony)."
)
    "This project demonstrates multi-objective optimization in exam scheduling "
    "by balancing capacity violations (hard constraint) and wasted classroom capacity (soft constraint) "
    "using ABC (Artificial Bee Colony)."
)
