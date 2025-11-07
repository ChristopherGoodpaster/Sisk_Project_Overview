# sisk_ballistics_cant.py
import matplotlib
matplotlib.use("Agg")  # avoids GUI backend issues

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Constants ---
G_FT_S2 = 32.174         # ft/s^2
IN_PER_FT = 12.0
FT_PER_YD = 3.0
MUZZLE_V_FPS = 2650.0    # locked muzzle velocity

st.set_page_config(page_title="Sisk — Cant Ballistics Demo", layout="wide")
st.title("Sisk Ballistics — Cant effect demo")
st.caption("If you see this line, Streamlit is running the script 👍")

st.markdown(
    "One aimed shot is placed center-mass on the left target. The right target shows where that "
    "same shot would hit if the rifle were canted by the angle below (gravity-only model — no drag)."
)

# --- Controls ---
col_controls, col_empty = st.columns([1, 3])
with col_controls:
    range_yd = st.number_input("Range (yards)", min_value=10, max_value=2000, value=200, step=10)
    cant_deg = st.slider("Cant (degrees clockwise)", -20.0, 20.0, 0.0, 0.1)
    show_grid = st.checkbox("Show target grid (1 in squares)", value=False)
    st.write("Muzzle velocity is locked to {:.0f} fps".format(MUZZLE_V_FPS))
    st.markdown("**Notes**: Simplified gravity-only ballistic model (no drag/BC).")

# --- Physics (simplified) ---
range_ft = range_yd * FT_PER_YD
t_sec = range_ft / MUZZLE_V_FPS
drop_ft = 0.5 * G_FT_S2 * (t_sec ** 2)
drop_in = drop_ft * IN_PER_FT

cant_rad = math.radians(cant_deg)
horizontal_offset_in = drop_in * math.sin(cant_rad)
vertical_offset_in = -drop_in * math.cos(cant_rad)

inches_per_moa = range_yd * 1.0472
horizontal_moa = horizontal_offset_in / inches_per_moa
vertical_moa = vertical_offset_in / inches_per_moa

# --- Results ---
st.subheader("Results (simplified gravity-only model)")
col1, col2, col3 = st.columns(3)
col1.metric("Time of flight (s)", f"{t_sec:.4f}")
col2.metric("Vertical drop (in)", f"{drop_in:.2f}")
col3.metric("Cant (deg)", f"{cant_deg:+.2f}")

col4, col5 = st.columns(2)
col4.write(f"Horizontal offset due to cant: **{horizontal_offset_in:.2f} in** ({horizontal_moa:+.2f} MOA)")
col5.write(f"Vertical offset at target: **{vertical_offset_in:.2f} in** ({vertical_moa:+.2f} MOA)")

# --- Plot targets ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

def draw_target(ax, title):
    radius_in = 12.0
    ax.set_xlim(-radius_in, radius_in)
    ax.set_ylim(-radius_in, radius_in)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    for r in [1, 3, 6, 9, 12]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.6))
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    if show_grid:
        ticks = np.arange(-radius_in, radius_in + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linewidth=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

draw_target(axes[0], "Aimed shot — no cant (center-mass)")
axes[0].plot(0, 0, marker='x', markersize=10, color='black')

draw_target(axes[1], f"Same aim, rifle canted {cant_deg:+.2f}°")
axes[1].plot(0, 0, marker='x', markersize=10, color='black')
axes[1].plot(horizontal_offset_in, vertical_offset_in, marker='o', markersize=8, color='red')
axes[1].annotate(
    f"{horizontal_offset_in:+.2f} in, {vertical_offset_in:+.2f} in",
    xy=(horizontal_offset_in, vertical_offset_in),
    xytext=(10, -10),
    textcoords='offset points'
)

st.pyplot(fig)

st.markdown(
    "### Math used\n"
    "- Time of flight: `t = range_ft / v0`\n"
    "- Drop: `drop_ft = 0.5 * g * t^2` → `drop_in = drop_ft * 12`\n"
    "- Cant θ: `x = drop_in * sin(θ)`, `y = -drop_in * cos(θ)`"
)
