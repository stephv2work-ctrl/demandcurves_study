"""Streamlit interface for exploring and comparing six demand models."""

import matplotlib.pyplot as plt
import streamlit as st

from models import SOLVERS
from plotting import make_plot


st.set_page_config(page_title="Demand Model Explorer", page_icon="📈", layout="wide")
st.title("Consumer Demand Model Explorer")
st.caption("Maximize utility subject to pₓx + pᵧy ≤ m for two goods.")

study_mode = st.radio(
    "Study mode",
    ["Explore one model", "Compare all models"],
    horizontal=True,
    help="Use comparison mode to hold the budget constant and see how preferences change the optimum.",
)

with st.sidebar:
    st.header("Budget")
    px = st.number_input("Price of x (pₓ)", min_value=0.01, max_value=1_000_000.0, value=2.0, step=0.25)
    py = st.number_input("Price of y (pᵧ)", min_value=0.01, max_value=1_000_000.0, value=4.0, step=0.25)
    income = st.number_input("Income (m)", min_value=0.0, max_value=1_000_000.0, value=100.0, step=5.0)

    if study_mode == "Explore one model":
        model = st.selectbox("Preference model", list(SOLVERS))
    else:
        model = None

    st.header("Preference parameters")
    parameters = {}
    comparison_parameters = {}
    if study_mode == "Compare all models":
        st.caption("These settings affect their corresponding comparison panels.")
        comparison_parameters["Cobb–Douglas"] = {
            "alpha": st.slider("Cobb–Douglas α", 0.01, 0.99, 0.60, key="compare_cd_alpha")
        }
        comparison_parameters["Addictive (non-convex preferences)"] = {}
        comparison_parameters["Neutral good"] = {
            "neutral_good": st.radio("Neutral good", ["y", "x"], horizontal=True, key="compare_neutral")
        }
        st.markdown("**Perfect substitutes**")
        comparison_parameters["Perfect substitutes"] = {
            "a": st.number_input("Substitutes a", min_value=0.01, max_value=100.0, value=1.0, step=0.1),
            "b": st.number_input("Substitutes b", min_value=0.01, max_value=100.0, value=1.0, step=0.1),
        }
        st.markdown("**Perfect complements**")
        comparison_parameters["Perfect complements"] = {
            "a": st.number_input("Complements a", min_value=0.01, max_value=100.0, value=1.0, step=0.1),
            "b": st.number_input("Complements b", min_value=0.01, max_value=100.0, value=1.0, step=0.1),
        }
        comparison_parameters["Quasi-linear"] = {
            "alpha": st.number_input(
                "Quasi-linear α", min_value=0.01, max_value=10_000.0, value=10.0, step=0.5
            )
        }
    elif model == "Cobb–Douglas":
        parameters["alpha"] = st.slider("α: weight on x", 0.01, 0.99, 0.60)
    elif model == "Neutral good":
        parameters["neutral_good"] = st.radio("Neutral good", ["y", "x"], horizontal=True)
    elif model in ("Perfect substitutes", "Perfect complements"):
        parameters["a"] = st.number_input("a", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        parameters["b"] = st.number_input("b", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    elif model == "Quasi-linear":
        parameters["alpha"] = st.number_input(
            "α in u = α ln(x) + y", min_value=0.01, max_value=10_000.0, value=10.0, step=0.5
        )

if study_mode == "Explore one model":
    try:
        result = SOLVERS[model](px, py, income, **parameters)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    left, right = st.columns([0.9, 1.5], gap="large")
    with left:
        st.subheader("Optimal choice")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("x*", f"{result.x:.3f}")
        metric_right.metric("y*", f"{result.y:.3f}")
        st.metric("Maximum utility", f"{result.utility:.3f}")

        st.write(f"**Solution type:** {result.solution_type.title()}")
        st.write(result.explanation)
        st.write(f"**Spending:** {result.spending:.2f} of {income:.2f}")
        if result.optimal_set:
            st.info(f"Optimal set: {result.optimal_set}")

    with right:
        figure = make_plot(model, px, py, income, parameters, result)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
else:
    st.subheader("Preference comparison")
    st.caption(
        "Every panel uses the same prices and income. Only preferences change, "
        "so differences between the red optima come from the utility model."
    )

    results = {}
    errors = []
    for comparison_model, solver in SOLVERS.items():
        model_parameters = comparison_parameters[comparison_model]
        try:
            results[comparison_model] = solver(px, py, income, **model_parameters)
        except ValueError as error:
            errors.append(f"{comparison_model}: {error}")

    if errors:
        for error in errors:
            st.error(error)
    else:
        columns = st.columns(2, gap="large")
        for index, (comparison_model, result) in enumerate(results.items()):
            with columns[index % 2]:
                st.markdown(f"#### {comparison_model}")
                figure = make_plot(
                    comparison_model,
                    px,
                    py,
                    income,
                    comparison_parameters[comparison_model],
                    result,
                )
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
                st.caption(
                    f"Optimum: ({result.x:.2f}, {result.y:.2f}) · "
                    f"{result.solution_type.title()} — {result.explanation}"
                )

        st.subheader("Compare the solutions")
        rows = []
        for comparison_model, result in results.items():
            rows.append(
                {
                    "Preference model": comparison_model,
                    "x*": round(result.x, 3),
                    "y*": round(result.y, 3),
                    "Solution type": result.solution_type.title(),
                    "Spending": round(result.spending, 2),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

with st.expander("Model definitions"):
    st.markdown(
        """
        - **Cobb–Douglas:** `u = x^α y^(1-α)`
        - **Addictive/non-convex preferences:** `u = x₁² + x₂²`
        - **Neutral good:** utility depends on only one good
        - **Perfect substitutes:** `u = ax + by`
        - **Perfect complements:** `u = min(x/a, y/b)`
        - **Quasi-linear:** `u = α ln(x) + y`
        """
    )
