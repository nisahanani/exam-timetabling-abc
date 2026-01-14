# University Exam Scheduling using Artificial Bee Colony (ABC) and Streamlit
import streamlit as st
import pandas as pd
import random
import math
import os
import time

st.set_page_config(page_title="Exam Scheduling with ABC", layout="wide")
st.title("University Exam Scheduling using Artificial Bee Colony (ABC)")

# -------------------- Load Datasets --------------------
st.subheader("Exam and Classroom Dataset")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Exam dataset
exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
if os.path.exists(exam_file):
    exams_df = pd.read_csv(exam_file)
    st.success("Exam dataset loaded successfully!")
    st.dataframe(exams_df)
else:
    st.error(f"Exam dataset not found at: {exam_file}")
    st.stop()

# Classroom dataset
room_file = os.path.join(BASE_DIR, "classrooms.csv")
if os.path.exists(room_file):
    rooms_df = pd.read_csv(room_file)
    st.success("Classroom dataset loaded successfully!")
    st.dataframe(rooms_df)
else:
    st.error(f"Classroom dataset not found at: {room_file}")
    st.stop()

# -------------------- Convert Datasets --------------------
exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()
exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# -------------------- Cost Function --------------------
def calculate_cost(schedule, alpha, beta):
    capacity_violation = 0
    wasted_capacity = 0
    for exam_id, room_id in schedule.items():
        students = exam_students[exam_id]
        capacity = room_capacity[room_id]
        if capacity < students:
            capacity_violation += (students - capacity)
        else:
            wasted_capacity += (capacity - students)
    total_cost = alpha * capacity_violation + beta * wasted_capacity
    return total_cost

# -------------------- ABC Helper Functions --------------------
def generate_solution(exams, rooms):
    solution = {}
    for exam_id in exams:
        solution[exam_id] = random.choice(rooms)
    return solution

def fitness_function(schedule, alpha, beta):
    """Higher fitness = better solution"""
    cost = calculate_cost(schedule, alpha, beta)
    return 10000 / (1 + cost)  # scale fitness

# -------------------- Artificial Bee Colony Algorithm --------------------
def artificial_bee_colony(exams, rooms, alpha, beta,
                          num_bees=40, num_food_sources=30,
                          scout_limit=10, max_iter=150):
    
    start_time = time.time()
    
    # Initial population
    population = [generate_solution(exams, rooms) for _ in range(num_bees)]
    fitness_values = [fitness_function(sol, alpha, beta) for sol in population]
    
    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    best_cost = calculate_cost(best_solution, alpha, beta)
    
    convergence_curve = []
    trial_counter = [0] * num_bees
    
    for iteration in range(max_iter):
        # Employed Bees Phase
        for i in range(num_bees):
            candidate = generate_solution(exams, rooms)
            candidate_fitness = fitness_function(candidate, alpha, beta)
            if candidate_fitness > fitness_values[i]:
                population[i] = candidate
                fitness_values[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1
        
        # Scout Bees Phase
        for i in range(num_bees):
            if trial_counter[i] > scout_limit:
                population[i] = generate_solution(exams, rooms)
                fitness_values[i] = fitness_function(population[i], alpha, beta)
                trial_counter[i] = 0
        
        # Update best solution
        current_best_index = fitness_values.index(max(fitness_values))
        current_best_cost = calculate_cost(population[current_best_index], alpha, beta)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[current_best_index]
        
        convergence_curve.append(best_cost)
    
    elapsed = time.time() - start_time
    return best_solution, best_cost, convergence_curve, elapsed

# -------------------- Streamlit Interface --------------------
st.subheader("ABC Parameters")
col1, col2 = st.columns(2)

with col1:
    num_bees = st.number_input("Number of Bees", 10, 100, 40, step=5)
    num_food_sources = st.number_input("Number of Food Sources", 5, 50, 30, step=1)
    max_iter = st.number_input("Max Iterations (Cycles)", 10, 200, 150, step=5)

with col2:
    scout_limit = st.number_input("Scout Limit", 1, 50, 10)
    alpha = st.slider("Weight Capacity Violation (α)", 10, 100, 50)
    beta = st.slider("Weight Wasted Capacity (β)", 1, 20, 5)

# -------------------- Run ABC --------------------
if st.button("Run ABC"):
    best_schedule, best_cost, history, elapsed = artificial_bee_colony(
        exam_ids, room_ids, alpha, beta,
        num_bees, num_food_sources, scout_limit, max_iter
    )
    
    st.write(f"ABC completed in {elapsed:.2f} seconds")
    st.write(f"Final Best Cost: {best_cost}")
    
    # Display final schedule
    result_df = pd.DataFrame([
        {"Exam ID": e, "Students": exam_students[e],
         "Classroom": r, "Room Capacity": room_capacity[r]}
        for e, r in best_schedule.items()
    ])
    st.subheader("Final Exam Schedule")
    st.dataframe(result_df)
    
    # Save CSV
    output_csv = os.path.join(BASE_DIR, "abc_exam_schedule.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    st.success(f"Final schedule saved to {output_csv}")
    
    # Plot convergence
    st.subheader("Convergence Curve")
    st.line_chart(history)

st.info(
    "This project demonstrates multi-objective optimization in exam scheduling "
    "by balancing capacity violations and wasted classroom capacity using ABC."
)
