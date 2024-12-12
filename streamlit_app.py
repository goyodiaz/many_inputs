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
df_data = df_new_powerlift[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg', 'Sex_Encoded']].dropna()

# Split data into predictors and target
def data_split(target):
    X = df_data.drop(columns=[target])
    y = df_data[target]
    return X, y

# Create models for the lift based on the model
models = {}
# for target in ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']:
target = 'Best3SquatKg'
X, y = data_split(target)
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
models[target] = model

# Form for user input
with st.form(key="lift_prediction_form"):
    input_lift = "Squat"
    bodyweight = 20
    sex_encoded = 0
    bench = st.number_input("Enter your Bench (kg):", min_value=20.0, max_value=600.0, step=0.1, key="bench_input")
    deadlift = st.number_input("Enter your Deadlift (kg):", min_value=20.0, max_value=600.0, step=0.1, key="deadlift_input")

    # Submit button
    submit = st.form_submit_button(label="Predict")

if submit:
    predicted_value = models['Best3SquatKg'].predict([[bench, deadlift, bodyweight, sex_encoded]])[0]
    st.write("Predicted Squat:", round(predicted_value, 2), "kg")
