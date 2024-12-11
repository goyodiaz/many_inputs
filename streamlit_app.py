import time

import numpy as np
import streamlit as st


def main():
    size = st.number_input(label="Size", value=1000)
    delay = st.number_input(label="Delay", value=10)
    if st.button("Calculate"):
        x = calculate(size=size, delay=delay)
        st.dataframe(x[:10, :10])


def calculate(size, delay):
    x = np.empty([size, size])
    time.sleep(delay)
    return x


if __name__ == "__main__":
    main()
