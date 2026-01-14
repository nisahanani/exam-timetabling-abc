import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import os

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(page_title="Exam Scheduling using ABC", layout="wide")
st.title("🐝 University Exam Scheduling using Artificial Bee Colony (ABC)")

st.write(
    "This system optimizes university exam scheduling using the Artificial Bee Colony (ABC) algorithm. "
    "Hard constraints include room capacity, room–timeslot conflicts, room-type compatibility, and a "
    "minimum seating safety margin. Classroom utilization efficiency is optimized as a soft objective."
)

# ======================================================
# Load Data
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
room_file = os.path.join(BASE_DIR, "classrooms.csv")

if not os.path.exists(exam_file) or not os.path.exists(room_file):
    st.error("Dataset files not found.")
    st.stop()

exams = pd.read_csv(exam_file)
rooms = pd.read_csv(room_file)

exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ======================================================
# Prepare Data
# ======================================================
exam_ids = exams["exam_id"].tolist()
room_ids = rooms["classroom_id"].tolist()

num_students = dict(zip(exams["exam_id"], exams["num_students"]))
exam_day = dict(zip(exams["exam_id"], exams["exam_day"]))
exam_time = dict(zip(exams["exam_id"], exams["exam_time"]))
exam_type = dict(zip(exams["exam_id"], exams["exam_type"]))   # theory / practical
course_code = dict(zip(exams["exam_id"], exams["course_code"]))

room_capacity = dict(zip(rooms["classroom_id"], rooms["capacity"]))
room_type = dict(zip(rooms["classroom_id"], rooms["room_type"]))  # lecture / lab

NUM_EXAMS = len(exam_ids)

# ======================================================
# Cost Function (Hard + Soft Constraints)
# ======================================================
def calculate_cost(schedule, alpha, beta, gamma, delta, safety_margin):
    capacity_violations = 0
    timeslot_conflicts = 0
    type_violations = 0
    wasted_capacity = 0.0

    used_slots = set()

    for exam, room in schedule.items():
        students = num_students[exam]
        capacity = room_capacity[room]
        empty_seats = capacity - students

        # ---- Capacity + Safety Margin (Hard) ----
        if empty_seats < 0:
            capacity_violations += 1
        elif empty_seats < safety_margin:
            capacity_violations += 1
            wasted_capacity += empty_seats / capacity
        else:
            wasted_capacity += empty_seats / capacity

        # ---- Room-Type Compatibility (Hard) ----
        if exam_type[exam] == "practical" and room_type[room] != "lab":
            type_violations += 1
        if exam_type[exam] == "theory" and room_type[room] == "lab":
            type_violations += 1

        # ---- Room–Timeslot Conflict (Hard) ----
        slot_key = (room, exam_day[exam], exam_time[exam])
        if slot_key in used_slots:
            timeslot_conflicts += 1
        else:
            used_slots.add(slot_key)

    total_cost = (
        alpha * capacity_violations +
        gamma * timeslot_conflicts +
        delta * type_violations +
        beta * wasted_capacity
    )

    return (
        total_cost,
        capacity_violations,
        timeslot_conflicts,
        type_violations,
        wasted_capacity
    )


def fitness(schedule, alpha, beta, gamma, delta, safety_margin):
    cost, *_ = calculate_cost(schedule, alpha, beta, gamma, delta, safety_margin)
    return 1 / (1 + cost)

# ======================================================
# ABC Helper Functions
# ======================================================
def generate_solution():
    return {exam: random.choice(room_ids) for exam in exam_ids}


def generate_neighbor(solution):
    neighbor = solution.copy()
    exam = random.choice(exam_ids)

    compatible_rooms = [
        r for r in room_ids
        if (exam_type[exam] == "practical" and room_type[r] == "lab") or
           (exam_type[exam] == "theory" and room_type[r] != "lab")
    ]

    if compatible_rooms and random.random() < 0.7:
        neighbor[exam] = random.choice(compatible_rooms)
    else:
        neighbor[exam] = random.choice(room_ids)

    return neighbor

# ======================================================
# Artificial Bee Colony Algorithm
# ======================================================
def artificial_bee_colony(
    colony_size, max_cycles, scout_limit,
    alpha, beta, gamma, delta, safety_margin
):
    start_time = time.time()

    food_sources = [generate_solution() for _ in range(colony_size)]
    trials = [0] * colony_size

    best_solution = None
    best_cost = float("inf")
    convergence = []

    for cycle in range(max_cycles):

        # ---------- Employed Bees ----------
        for i in range(colony_size):
            candidate = generate_neighbor(food_sources[i])
            if fitness(candidate, alpha, beta, gamma, delta, safety_margin) > \
               fitness(food_sources[i], alpha, beta, gamma, delta, safety_margin):
                food_sources[i] = candidate
                trials[i] = 0
            else:
                trials[i] += 1

        # ---------- Onlooker Bees ----------
        fitness_values = [
            fitness(sol, alpha, beta, gamma, delta, safety_margin)
            for sol in food_sources
        ]
        total_fitness = sum(fitness_values)

        for _ in range(colony_size):
            r = random.uniform(0, total_fitness)
            acc = 0
            for i, fit in enumerate(fitness_values):
                acc += fit
                if acc >= r:
                    candidate = generate_neighbor(food_sources[i])
                    if fitness(candidate, alpha, beta, gamma, delta, safety_margin) > \
                       fitness(food_sources[i], alpha, beta, gamma, delta, safety_margin):
                        food_sources[i] = candidate
                        trials[i] = 0
                    else:
                        trials[i] += 1
                    break

        # ---------- Scout Bees ----------
        for i in range(colony_size):
            if trials[i] >= scout_limit:
                food_sources[i] = generate_solution()
                trials[i] = 0

        # ---------- Best Solution ----------
        for sol in food_sources:
            cost, *_ = calculate_cost(
                sol, alpha, beta, gamma, delta, safety_margin
            )
            if cost < best_cost:
                best_cost = cost
                best_solution = sol

        # 🔽 NORMALIZED COST FOR PLOTTING
        convergence.append(best_cost / NUM_EXAMS)

    elapsed = time.time() - start_time
    return best_solution, best_cost, convergence, elapsed

# ======================================================
# Sidebar Parameters
# ======================================================
st.sidebar.header("ABC Parameters")

colony_size = st.sidebar.slider("Colony Size", 10, 100, 40, 5)
max_cycles = st.sidebar.slider("Max Cycles", 50, 300, 150, 25)
scout_limit = st.sidebar.slider("Scout Limit", 5, 50, 15, 5)

st.sidebar.markdown("### Constraint Weights")
alpha = st.sidebar.slider("Capacity & Safety Violation (α)", 50, 200, 100)
gamma = st.sidebar.slider("Timeslot Conflict (γ)", 50, 200, 100)
delta = st.sidebar.slider("Room-Type Violation (δ)", 50, 200, 100)
beta = st.sidebar.slider("Wasted Capacity (β)", 1, 20, 5)

safety_margin = st.sidebar.slider(
    "Minimum Empty Seats (Safety Margin)", 0, 20, 5
)

# ======================================================
# Run ABC
# ======================================================
if st.button("🚀 Run ABC Optimization"):

    with st.spinner("Running Artificial Bee Colony..."):
        best_solution, best_cost, history, elapsed = artificial_bee_colony(
            colony_size, max_cycles, scout_limit,
            alpha, beta, gamma, delta, safety_margin
        )

    (
        cost,
        cap_v,
        time_v,
        type_v,
        wasted
    ) = calculate_cost(
        best_solution, alpha, beta, gamma, delta, safety_margin
    )

    # ---------- Results ----------
    st.subheader("📌 Final Optimization Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Cost", round(cost, 3))
    c2.metric("Capacity Violations", cap_v)
    c3.metric("Timeslot Conflicts", time_v)
    c4.metric("Room-Type Violations", type_v)
    c5.metric("Wasted Capacity", round(wasted, 3))

    # ---------- Convergence ----------
    st.subheader("📈 Convergence Curve (Normalized Cost per Exam)")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Average Cost per Exam")
    ax.set_title("ABC Convergence Curve")
    st.pyplot(fig)

    # ---------- Final Schedule ----------
    st.subheader("🗓️ Optimized Exam Schedule")
    result_df = pd.DataFrame([
        {
            "Exam ID": e,
            "Course Code": course_code[e],
            "Students": num_students[e],
            "Exam Day": exam_day[e],
            "Exam Time": exam_time[e],
            "Classroom": r,
            "Room Type": room_type[r],
            "Room Capacity": room_capacity[r]
        }
        for e, r in best_solution.items()
    ])
    st.dataframe(result_df, use_container_width=True)

# ======================================================
# Footer
# ======================================================
st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Algorithm:** Artificial Bee Colony (ABC)  \n"
    "**Case Study:** University Exam Scheduling"
)
