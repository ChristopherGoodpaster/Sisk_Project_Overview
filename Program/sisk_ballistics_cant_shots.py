# sisk_ballistics_cant_shots.py
import math
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Constants ----------
G_FT_S2 = 32.174
IN_PER_FT = 12.0
FT_PER_YD = 3.0
MUZZLE_V_FPS = 2650.0

# Colors for shots (1,2,3)
SHOT_COLORS = ["red", "blue", "green"]

# ---------- Helpers ----------
def compute_offsets(range_yd: float, cant_deg: float):
    """Return (horizontal_offset_in, vertical_offset_in, t_sec, drop_in)."""
    range_ft = range_yd * FT_PER_YD
    t_sec = range_ft / MUZZLE_V_FPS
    drop_ft = 0.5 * G_FT_S2 * (t_sec ** 2)
    drop_in = drop_ft * IN_PER_FT
    cant_rad = math.radians(cant_deg)
    horizontal_offset_in = drop_in * math.sin(cant_rad)
    vertical_offset_in = -drop_in * math.cos(cant_rad)
    return horizontal_offset_in, vertical_offset_in, t_sec, drop_in

def init_session_state():
    if "shots" not in st.session_state:
        st.session_state.shots = []  # list of dicts: {'index', 'cant', 'h_in', 'v_in', 'color'}
    if "range_yd" not in st.session_state:
        st.session_state.range_yd = 200
    if "cant_deg" not in st.session_state:
        st.session_state.cant_deg = 0.0
    if "show_grid" not in st.session_state:
        st.session_state.show_grid = False

def reset_all():
    st.session_state.shots = []
    st.session_state.range_yd = 200
    st.session_state.cant_deg = 0.0
    st.session_state.show_grid = False
    # also update widgets that use these keys (they'll reflect new values automatically)

# ---------- App ----------
st.set_page_config(page_title="Sisk — Cant Ballistics (Multi-shot)", layout="wide")
st.title("Sisk Ballistics — Cant effect (multi-shot demo)")
st.caption("Lock: muzzle velocity 2650 fps. Gravity-only model (no drag). Aim is center-mass on left target.")

init_session_state()

# Controls column
with st.sidebar:
    st.header("Controls")
    range_yd = st.number_input(
        "Range (yards)", min_value=10, max_value=2000, value=st.session_state.range_yd, step=10,
        key="range_yd")
    cant_deg = st.slider(
        "Cant (degrees clockwise)", -20.0, 20.0, value=st.session_state.cant_deg, step=0.1,
        key="cant_deg")
    show_grid = st.checkbox("Show target grid (1 in squares)", value=st.session_state.show_grid, key="show_grid")
    st.write(f"Muzzle velocity locked at **{MUZZLE_V_FPS:.0f} fps**")
    st.markdown("---")

    # Shoot buttons for 1..3
    st.markdown("### Fire shots (click while cant is set)")
    cols = st.columns(3)
    for i in range(3):
        idx = i + 1
        def make_shoot_callback(index=idx):
            def shoot():
                # Don't record more than 3 shots
                if len(st.session_state.shots) >= 3:
                    st.warning("Maximum 3 shots recorded. Reset to start again.")
                    return
                h_in, v_in, t_sec, drop_in = compute_offsets(st.session_state.range_yd, st.session_state.cant_deg)
                shot = {
                    "index": index,
                    "cant": float(st.session_state.cant_deg),
                    "h_in": float(h_in),
                    "v_in": float(v_in),
                    "t_sec": float(t_sec),
                    "drop_in": float(drop_in),
                    "color": SHOT_COLORS[len(st.session_state.shots)]  # assign next available color
                }
                st.session_state.shots.append(shot)
            return shoot
        cols[i].button(f"Shoot {idx}", on_click=make_shoot_callback())

    st.markdown("---")
    if st.button("Reset all", on_click=reset_all):
        st.success("Reset to default settings.")

# ---------- Physics & main display ----------
# Compute current (instant) offsets for the slider's cant (useful to preview)
cur_h_in, cur_v_in, cur_t, cur_drop_in = compute_offsets(st.session_state.range_yd, st.session_state.cant_deg)
inches_per_moa = st.session_state.range_yd * 1.0472
cur_h_moa = cur_h_in / inches_per_moa
cur_v_moa = cur_v_in / inches_per_moa

st.subheader("Current preview (gravity-only)")
c1, c2, c3 = st.columns(3)
c1.metric("Time of flight (s)", f"{cur_t:.4f}")
c2.metric("Vertical drop (in)", f"{cur_drop_in:.2f}")
c3.metric("Cant (deg)", f"{st.session_state.cant_deg:+.2f}")

st.write(f"Preview horizontal offset: **{cur_h_in:+.2f} in** ({cur_h_moa:+.2f} MOA) — preview only; use *Shoot* to record.")

# Show recorded shots table (if any)
if st.session_state.shots:
    st.markdown("### Recorded shots")
    shot_rows = []
    for s in st.session_state.shots:
        shot_rows.append(
            f"Shot {s['index']}: cant={s['cant']:+.2f}°, offset={s['h_in']:+.2f} in (H) , {s['v_in']:+.2f} in (V) — color: {s['color']}"
        )
    for row in shot_rows:
        st.write(row)
else:
    st.info("No shots recorded yet. Set cant and click a 'Shoot' button in the sidebar to add a shot.")

# ---------- Plot targets ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

def draw_target(ax, title, show_grid=False):
    radius_in = 12.0
    ax.set_xlim(-radius_in, radius_in)
    ax.set_ylim(-radius_in, radius_in)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    # rings
    for r in [1, 3, 6, 9, 12]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.7))
    # crosshair
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    if show_grid:
        ticks = np.arange(-radius_in, radius_in + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linewidth=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# Left: zeroed / aimed target — show bullet mark at center (zero)
draw_target(axes[0], "Aimed target — zeroed (aim at center)", show_grid=st.session_state.show_grid)
# draw a solid bullet mark for zeroed shot
axes[0].plot(0, 0, marker='o', markersize=10, label='Aim (zero)', color='black', markeredgecolor='white', zorder=5)

# Right: impacts for recorded shots (and also show aim point for reference)
draw_target(axes[1], "Impacts (same aim, rifle canted values recorded)", show_grid=st.session_state.show_grid)
axes[1].plot(0, 0, marker='x', markersize=10, color='black', label='Aim point', zorder=4)

# Plot each recorded shot impact (if any)
for s in st.session_state.shots:
    h = s["h_in"]
    v = s["v_in"]
    color = s["color"]
    marker = 'o'  # filled circle
    # plot with a slightly larger edge to stand out
    axes[1].scatter(h, v, s=90, c=color, edgecolors='k', linewidths=0.8, zorder=6)
    # label near the point
    axes[1].annotate(
        f"Shot {s['index']} ({s['cant']:+.2f}°)\n{h:+.2f} in, {v:+.2f} in",
        xy=(h, v),
        xytext=(8, -8),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        zorder=7
    )

# Add legend for shot colors
if st.session_state.shots:
    # construct legend handles
    handles = []
    labels = []
    for i, s in enumerate(st.session_state.shots):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=s['color'], markeredgecolor='k', markersize=8))
        labels.append(f"Shot {s['index']} ({s['cant']:+.2f}°)")
    axes[1].legend(handles, labels, loc='upper right', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# ---------- Math explanation ----------
st.markdown(
    "### Math (same as before)\n"
    "- Time of flight: `t = range_ft / v0`.\n"
    "- Vertical drop (ft): `drop_ft = 0.5 * g * t^2` → `drop_in = drop_ft * 12`.\n"
    "- With rifle canted by θ (radians), drop vector rotates and becomes horizontal + vertical offsets:\n"
    "  - `x_offset_in = drop_in * sin(θ)`\n"
    "  - `y_offset_in = -drop_in * cos(θ)`  (negative = downwards on target)\n\n"
    "Workflow: set cant slider, click a 'Shoot' button to record that shot at the current cant. You can record up to 3 shots and compare them visually."
)
