import streamlit as st
import pandas as pd
import numpy as np

import time

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]


st.bar_chart(hist_values)