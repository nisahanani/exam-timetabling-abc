import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import os

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Exam Scheduling using ABC", layout="wide")
st.title("🐝 University Exam Scheduling using Artificial Bee Colony (ABC)")

st.write(
    "This application optimizes university exam timetables using the "
    "Artificial Bee Colony algorithm, considering student numbers and classroom capacity constraints."
)

# ==============================
# Load Data
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
room_file = os.path.join(BASE_DIR, "classrooms.csv")

if os.path.exists(exam_file) and os.path.exists(room_file):
    exams = pd.read_csv(exam_file)
    rooms = pd.read_csv(room_file)
    st.success("Datasets loaded successfully!")
else:
    st.error("Exam or classroom dataset not found.")
    st.stop()

# Normalize columns
exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ==============================
# Prepare Data
# ==============================
exam_ids = exams["exam_id"].tolist()
room_ids = rooms["classroom_id"].tolist()
num_students = dict(zip(exams["exam_id"], exams["num_students"]))
room_capacity = dict(zip(rooms["classroom_id"], rooms["capacity"]))
course_code = dict(zip(exams["exam_id"], exams["course_code"]))

# ==============================
# Cost & Metrics
# ==============================
def calculate_cost(schedule, alpha, beta):
    capacity_violations = 0
    wasted_capacity = 0

    for exam, room in schedule.items():
        students = num_students[exam]
        capacity = room_capacity[room]

        if students > capacity:
            capacity_violations += 1
        else:
            wasted_capacity += (capacity - students)

    total_cost = alpha * capacity_violations + beta * wasted_capacity
    return total_cost, capacity_violations, wasted_capacity

def fitness(schedule, alpha, beta):
    cost, _, _ = calculate_cost(schedule, alpha, beta)
    return 1 / (1 + cost)

# ==============================
# ABC Helper Functions
# ==============================
def generate_solution():
    return {exam: random.choice(room_ids) for exam in exam_ids}

def neighbor_solution(solution):
    new_solution = solution.copy()
    exam = random.choice(exam_ids)

    # 30% chance to intentionally pick a too-small room to create a violation
    if random.random() < 0.3:
        smaller_rooms = [r for r in room_ids if room_capacity[r] < num_students[exam]]
        if smaller_rooms:
            new_solution[exam] = random.choice(smaller_rooms)
        else:
            new_solution[exam] = random.choice(room_ids)
    else:
        # Pick the closest-fit room (safe)
        room_options = sorted(room_ids, key=lambda r: abs(room_capacity[r] - num_students[exam]))
        new_solution[exam] = room_options[0]

    return new_solution

# ==============================
# Artificial Bee Colony Algorithm
# ==============================
def artificial_bee_colony(colony_size, max_cycles, scout_limit, alpha, beta):
    start_time = time.time()
    food_sources = [generate_solution() for _ in range(colony_size)]
    trials = [0] * colony_size

    best_solution = None
    best_cost = float("inf")
    convergence = []

    for cycle in range(max_cycles):
        # Employed Bees
        for i in range(colony_size):
            candidate = neighbor_solution(food_sources[i])
            if fitness(candidate, alpha, beta) > fitness(food_sources[i], alpha, beta):
                food_sources[i] = candidate
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bees
        probabilities = [fitness(sol, alpha, beta) for sol in food_sources]
        total_prob = sum(probabilities)
        for _ in range(colony_size):
            r = random.uniform(0, total_prob)
            acc = 0
            for i, prob in enumerate(probabilities):
                acc += prob
                if acc >= r:
                    candidate = neighbor_solution(food_sources[i])
                    if fitness(candidate, alpha, beta) > fitness(food_sources[i], alpha, beta):
                        food_sources[i] = candidate
                        trials[i] = 0
                    else:
                        trials[i] += 1
                    break

        # Scout Bees
        for i in range(colony_size):
            if trials[i] > scout_limit:
                food_sources[i] = generate_solution()
                trials[i] = 0

        # Update best solution
        for sol in food_sources:
            cost, _, _ = calculate_cost(sol, alpha, beta)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol

        convergence.append(best_cost)

    elapsed = time.time() - start_time
    return best_solution, best_cost, convergence, elapsed

# ==============================
# Sidebar Parameters
# ==============================
st.sidebar.header("ABC Parameters")

colony_size = st.sidebar.slider("Number of Bees (Colony Size)", 10, 100, 50, 5)
max_cycles = st.sidebar.slider("Max Cycles", 50, 300, 150, 25)
scout_limit = st.sidebar.slider("Scout Limit", 5, 50, 20, 5)

st.sidebar.markdown("### Objective Weights")
alpha = st.sidebar.slider("Capacity Violation Weight (α)", 10, 100, 50)
beta = st.sidebar.slider("Wasted Capacity Weight (β)", 1, 20, 5)

# ==============================
# Run ABC
# ==============================
if st.button("🚀 Run ABC Optimization"):
    with st.spinner("Running Artificial Bee Colony..."):
        best_solution, best_cost, history, elapsed = artificial_bee_colony(
            colony_size, max_cycles, scout_limit, alpha, beta
        )

    cost, cap_violations, wasted = calculate_cost(best_solution, alpha, beta)

    # ==============================
    # Metrics
    # ==============================
    st.subheader("📌 Final Optimization Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Cost", round(cost, 2))
    col2.metric("Capacity Violations", cap_violations)
    col3.metric("Wasted Capacity", wasted)

    # ==============================
    # Convergence Curve
    # ==============================
    st.subheader("📈 ABC Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cost")
    ax.set_title("ABC Convergence Curve")
    st.pyplot(fig)

    # ==============================
    # Final Schedule
    # ==============================
    st.subheader("🗓️ Optimized Exam Schedule")
    result_df = pd.DataFrame([
        {
            "Exam ID": e,
            "Course Code": course_code[e],
            "Students": num_students[e],
            "Classroom": r,
            "Room Capacity": room_capacity[r]
        } for e, r in best_solution.items()
    ])
    st.dataframe(result_df, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Algorithm:** Artificial Bee Colony (ABC)  \n"
    "**Case Study:** University Exam Scheduling"
)
