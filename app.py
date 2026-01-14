import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Exam Scheduling using ABC", layout="wide")

st.title("🐝 University Exam Scheduling using Artificial Bee Colony")
st.write(
    "This application optimizes university exam timetables using the "
    "Artificial Bee Colony (ABC) algorithm while considering student numbers "
    "and classroom capacity constraints."
)

# ==============================
# Set Random Seed for Reproducibility
# ==============================
SEED = 42
random.seed(SEED)

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()

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
    for _ in range(2):  # Modify 2 exams randomly
        exam = random.choice(exam_ids)
        new_solution[exam] = random.choice(room_ids)
    return new_solution

# ==============================
# Artificial Bee Colony Algorithm
# ==============================
def artificial_bee_colony(
    num_bees, num_food_sources, max_cycles, scout_limit, alpha, beta
):
    start_time = time.time()

    # Initialize food sources
    food_sources = [generate_solution() for _ in range(num_food_sources)]
    trials = [0] * num_food_sources

    best_solution = None
    best_cost = float("inf")
    convergence = []

    for cycle in range(max_cycles):

        # Employed Bees Phase
        for i in range(num_food_sources):
            candidate = neighbor_solution(food_sources[i])
            if fitness(candidate, alpha, beta) > fitness(food_sources[i], alpha, beta):
                food_sources[i] = candidate
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bees Phase
        probabilities = [fitness(sol, alpha, beta) for sol in food_sources]
        total_prob = sum(probabilities)

        for _ in range(num_food_sources):
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

        # Scout Bees Phase
        for i in range(num_food_sources):
            if trials[i] > scout_limit:
                food_sources[i] = generate_solution()
                trials[i] = 0

        # Best Solution Update
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

num_bees = st.sidebar.slider("Number of Bees", 10, 100, 30, 5)
num_food_sources = st.sidebar.slider("Number of Food Sources", 5, 50, 15, 1)
max_cycles = st.sidebar.slider("Max Cycles", 50, 300, 150, 25)
scout_limit = st.sidebar.slider("Scout Limit", 5, 50, 20, 5)

st.sidebar.markdown("### Objective Weights")
alpha = st.sidebar.slider("Capacity Violation Weight (α)", 10, 100, 50)
beta = st.sidebar.slider("Wasted Capacity Weight (β)", 1, 20, 5)

# ==============================
# Run Button
# ==============================
if st.button("🚀 Run ABC Optimization"):

    with st.spinner("Running Artificial Bee Colony Optimization..."):
        best_solution, best_cost, history, elapsed = artificial_bee_colony(
            num_bees, num_food_sources, max_cycles, scout_limit, alpha, beta
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
        }
        for e, r in best_solution.items()
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
