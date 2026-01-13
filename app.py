import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from abc_algorithm import artificial_bee_colony

st.set_page_config(page_title="ABC Exam Timetabling", layout="wide")

st.title("Exam Room Optimization using ABC Algorithm")

# Load data
classrooms = pd.read_csv("classrooms.csv")
exams = pd.read_csv("exam_timeslot.csv")

exams = exams.to_dict(orient="records")
classrooms = classrooms.to_dict(orient="records")

# Sidebar
st.sidebar.header("Algorithm Parameters")
num_bees = st.sidebar.slider("Number of Bees", 10, 100, 30)
max_iter = st.sidebar.slider("Max Iterations", 50, 300, 100)
scout_limit = st.sidebar.slider("Scout Limit", 5, 50, 20)

if st.sidebar.button("Run Optimization"):
    with st.spinner("Running Artificial Bee Colony..."):
        best_solution, best_fitness, curve = artificial_bee_colony(
            exams, classrooms, num_bees, max_iter, scout_limit
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(curve)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness Value")
        ax.set_title("ABC Convergence Curve")
        st.pyplot(fig)

    with col2:
        st.subheader("Final Fitness Score")
        st.success(f"Best Fitness Achieved: {best_fitness:.6f}")

    st.subheader("Optimized Schedule")
    st.dataframe(pd.DataFrame(best_solution))
