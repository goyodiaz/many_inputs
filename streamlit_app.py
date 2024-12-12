import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


@st.cache_resource(max_entries=1)
def get_data(lenght):
    names = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg', 'Sex_Encoded']
    rng = np.random.default_rng(seed=0)
    data = rng.random(size=(lenght, len(names)))
    return pd.DataFrame(data, columns=names)


lenght = st.number_input("Data lenght", value=10000)
df_new_powerlift = get_data(lenght=lenght)
tab = "Lift Prediction Calculator"
if tab == "Lift Prediction Calculator":
    st.subheader("Predict Your Missing Lift")
    st.write("""
    Use this calculator to predict one of your lifts (Squat, Bench, or Deadlift) based on your body weight, sex, and the other two lifts. This can be useful if you want to predict where you should be on a lift, or to identify which of your lifts is the weakest.
    """)

    # Prepare data for training
    df_data = df_new_powerlift[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg', 'Sex_Encoded']].dropna()

    # Split data into predictors and target
    def data_split(target):
        X = df_data.drop(columns=[target])
        y = df_data[target]
        return X, y

    # Create models for the lift based on the model
    models = {}
    for target in ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']:
        X, y = data_split(target)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        models[target] = model

    # Form for user input
    with st.form(key="lift_prediction_form"):
        input_lift = st.selectbox("Which lift do you want to predict?", ["Squat", "Bench", "Deadlift"])
        bodyweight = st.number_input("Enter your body weight (kg):", min_value=20.0, max_value=200.0, step=0.1)
        sex = st.selectbox("Select your sex:", ["Male", "Female"])
        sex_encoded = 0 if sex == "Male" else 1

        if input_lift == "Squat":
            bench = st.number_input("Enter your Bench (kg):", min_value=20.0, max_value=600.0, step=0.1, key="bench_input")
            deadlift = st.number_input("Enter your Deadlift (kg):", min_value=20.0, max_value=600.0, step=0.1, key="deadlift_input")
        elif input_lift == "Bench":
            squat = st.number_input("Enter your Squat (kg):", min_value=20.0, max_value=600.0, step=0.1, key="squat_input")
            deadlift = st.number_input("Enter your Deadlift (kg):", min_value=20.0, max_value=600.0, step=0.1, key="deadlift_input")
        elif input_lift == "Deadlift":
            squat = st.number_input("Enter your Squat (kg):", min_value=20.0, max_value=600.0, step=0.1, key="squat_input")
            bench = st.number_input("Enter your Bench (kg):", min_value=20.0, max_value=600.0, step=0.1, key="bench_input")

        # Submit button
        submit = st.form_submit_button(label="Predict")

    # Perform prediction only after the form is submitted
    if submit:
        if input_lift == "Squat":
            predicted_value = models['Best3SquatKg'].predict([[bench, deadlift, bodyweight, sex_encoded]])[0]
            st.write("Predicted Squat:", round(predicted_value, 2), "kg")
        elif input_lift == "Bench":
            predicted_value = models['Best3BenchKg'].predict([[squat, deadlift, bodyweight, sex_encoded]])[0]
            st.write("Predicted Bench:", round(predicted_value, 2), "kg")
        elif input_lift == "Deadlift":
            predicted_value = models['Best3DeadliftKg'].predict([[squat, bench, bodyweight, sex_encoded]])[0]
            st.write("Predicted Deadlift:", round(predicted_value, 2), "kg")
