# sisk_ballistics_with_bore.py
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
TARGET_RADIUS_IN = 12.0
SHOT_COLORS = ["red", "blue", "green"]

# ---------- Helpers ----------
def compute_zero_bore_angle(v0, zero_range_yd, sight_height_in):
    """Return (theta_rad, drop_at_zero_in). Bore angled above LOS to intersect at zero range."""
    R0_ft = zero_range_yd * FT_PER_YD
    R0_in = R0_ft * IN_PER_FT
    t0 = R0_ft / v0
    drop_ft = 0.5 * G_FT_S2 * (t0 ** 2)
    drop_in = drop_ft * IN_PER_FT
    theta_rad = math.atan2(drop_in + sight_height_in, R0_in)
    return theta_rad, drop_in

def compute_impact_at_range(v0, bore_angle_rad, sight_height_in, range_yd, cant_deg):
    """
    Impact on target plane relative to LOS center.
    Returns: (h_in, v_in, t_sec, y_rel_LOS_in, off_target_bool)
    """
    R_ft = range_yd * FT_PER_YD
    cos_t = math.cos(bore_angle_rad)
    t_sec = float('inf') if cos_t <= 1e-12 else R_ft / (v0 * cos_t)

    # Vertical relative to bore
    y_ft = v0 * math.sin(bore_angle_rad) * t_sec - 0.5 * G_FT_S2 * (t_sec ** 2)
    y_in = y_ft * IN_PER_FT

    # Relative to LOS (LOS is sight_height above bore at muzzle)
    y_rel_LOS_in = y_in - sight_height_in  # +up, -down

    # Rotate that vertical offset by cant around the LOS axis
    cant = math.radians(cant_deg)
    x0, y0 = 0.0, y_rel_LOS_in
    h_in = x0 * math.cos(cant) - y0 * math.sin(cant)   # right(+)
    v_in = x0 * math.sin(cant) + y0 * math.cos(cant)   # up(+)

    off = math.hypot(h_in, v_in) > TARGET_RADIUS_IN
    return h_in, v_in, t_sec, y_rel_LOS_in, off

def init_session_state():
    defaults = {
        "shots": [],
        "range_yd": 200,     # INT
        "cant_deg": 0.0,
        "show_grid": False,
        "sight_height": 2.5, # FLOAT
        "zero_range": 100    # INT
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_all():
    st.session_state.shots = []
    st.session_state.range_yd = 200
    st.session_state.cant_deg = 0.0
    st.session_state.show_grid = False
    st.session_state.sight_height = 2.5
    st.session_state.zero_range = 100

def add_shot(index: int):
    theta_rad, _ = compute_zero_bore_angle(
        MUZZLE_V_FPS, st.session_state.zero_range, st.session_state.sight_height
    )
    h_in, v_in, t_sec, y_rel, off = compute_impact_at_range(
        MUZZLE_V_FPS, theta_rad, st.session_state.sight_height,
        st.session_state.range_yd, st.session_state.cant_deg
    )
    if len(st.session_state.shots) < 3:
        st.session_state.shots.append({
            "index": index,
            "cant": float(st.session_state.cant_deg),
            "h_in": float(h_in),
            "v_in": float(v_in),
            "t_sec": float(t_sec),
            "y_rel_LOS_in": float(y_rel),
            "off_target": bool(off),
            "color": SHOT_COLORS[len(st.session_state.shots)]
        })

# ---------- App ----------
st.set_page_config(page_title="Sisk — Bore vs LOS + Cant", layout="wide")
st.title("Sisk Ballistics — Bore axis vs Line-of-Sight & Cant")
st.caption("Muzzle velocity locked at 2650 fps. Gravity-only model (no drag).")

init_session_state()

# Normalize legacy types (ints for yards, float for inches)
if isinstance(st.session_state.range_yd, float):
    st.session_state.range_yd = int(round(st.session_state.range_yd))
if isinstance(st.session_state.zero_range, float):
    st.session_state.zero_range = int(round(st.session_state.zero_range))
if isinstance(st.session_state.sight_height, int):
    st.session_state.sight_height = float(st.session_state.sight_height)

# Sidebar
with st.sidebar:
    st.header("Controls")

    st.number_input("Range (yards) (display)", min_value=10, max_value=2000, step=10,
                    value=st.session_state.range_yd, key="range_yd")
    st.number_input("Sight height (in) — LOS above bore (2.5–5.0)",
                    min_value=2.5, max_value=5.0, step=0.1,
                    value=st.session_state.sight_height, key="sight_height")
    st.number_input("Zero range (yards) — bullet intersects LOS",
                    min_value=10, max_value=1000, step=10,
                    value=st.session_state.zero_range, key="zero_range")
    st.slider("Cant (degrees clockwise)", -60.0, 60.0, step=0.1,
              value=st.session_state.cant_deg, key="cant_deg")
    st.checkbox("Show 1-in grid", value=st.session_state.show_grid, key="show_grid")
    st.write(f"Muzzle velocity locked at **{MUZZLE_V_FPS:.0f} fps**")
    st.markdown("---")

    st.markdown("### Fire shots (click while cant is set)")
    cols = st.columns(3)
    for i, col in enumerate(cols, start=1):
        with col:
            st.button(f"Shoot {i}", on_click=add_shot, args=(i,), disabled=(len(st.session_state.shots) >= 3))

    st.markdown("---")
    st.button("Reset all", on_click=reset_all)

# Bore angle & preview
theta_rad, drop_at_zero = compute_zero_bore_angle(
    MUZZLE_V_FPS, st.session_state.zero_range, st.session_state.sight_height
)
theta_deg = math.degrees(theta_rad)
theta_moa = theta_deg * 60.0

st.subheader("Zeroing & Bore angle")
c1, c2, c3 = st.columns(3)
c1.metric("Zero range (yd)", f"{st.session_state.zero_range:.0f}")
c2.metric("Sight height (in)", f"{st.session_state.sight_height:.2f}")
c3.metric("Bore angle", f"{theta_deg:.4f}°  ({theta_moa:.2f} MOA)")
st.write(f"Gravity drop at zero range used in calc: **{drop_at_zero:.2f} in**")

h_in, v_in, t_sec, y_rel_LOS_in, off_preview = compute_impact_at_range(
    MUZZLE_V_FPS, theta_rad, st.session_state.sight_height,
    st.session_state.range_yd, st.session_state.cant_deg
)

st.subheader("Current preview (gravity-only)")
pp1, pp2, pp3 = st.columns(3)
pp1.metric("Time of flight (s)", f"{t_sec:.4f}")
pp2.metric("Bullet vs LOS (in)", f"{y_rel_LOS_in:+.2f}")
pp3.metric("Cant (deg)", f"{st.session_state.cant_deg:+.2f}")

st.caption("Tip: If **Range == Zero**, y vs. LOS ≈ 0 ⇒ cant windage is small. "
           "Try Zero = 100 yd, Range = 200 yd to see a larger effect.")

if off_preview:
    st.error(f"Preview impact at {st.session_state.range_yd} yd → OFF TARGET (h={h_in:+.2f} in, v={v_in:+.2f} in)")
else:
    st.write(f"Preview impact at {st.session_state.range_yd} yd → h = **{h_in:+.2f} in**, v = **{v_in:+.2f} in**")

# Shots summary
if st.session_state.shots:
    st.markdown("### Recorded shots")
    for s in st.session_state.shots:
        off_text = " — **OFF TARGET**" if s["off_target"] else ""
        st.write(
            f"Shot {s['index']}: cant={s['cant']:+.2f}°, "
            f"impact = {s['h_in']:+.2f} in (H), {s['v_in']:+.2f} in (V){off_text} — color: {s['color']}"
        )
else:
    st.info("No shots recorded yet. Set cant and click a 'Shoot' button in the sidebar.")

# Plot targets
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

def draw_target(ax, title, show_grid=False):
    ax.set_xlim(-TARGET_RADIUS_IN, TARGET_RADIUS_IN)
    ax.set_ylim(-TARGET_RADIUS_IN, TARGET_RADIUS_IN)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    for r in [1, 3, 6, 9, TARGET_RADIUS_IN]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.7))
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    if show_grid:
        ticks = np.arange(-TARGET_RADIUS_IN, TARGET_RADIUS_IN + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linewidth=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# Left: zero view
draw_target(axes[0], "Aimed target — LOS (zero view)", show_grid=st.session_state.show_grid)
axes[0].plot(0, 0, marker='x', markersize=10, color='black', zorder=6)
if abs(st.session_state.zero_range - st.session_state.range_yd) < 1e-9:
    axes[0].scatter(0, 0, s=120, marker='o', color='black', edgecolors='white', zorder=7)

# Right: impacts
draw_target(axes[1], f"Impacts at {st.session_state.range_yd} yd (same LOS aim)", show_grid=st.session_state.show_grid)
axes[1].plot(0, 0, marker='x', markersize=10, color='black', zorder=4)

axes[1].scatter(h_in, v_in, s=80, facecolors='none', edgecolors='k', linewidths=1.0, zorder=5)
if off_preview:
    axes[1].annotate("PREVIEW: OFF TARGET", xy=(0.5, 0.95), xycoords='axes fraction',
                     color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
else:
    axes[1].annotate(f"Preview\n{h_in:+.2f} in, {v_in:+.2f} in", xy=(h_in, v_in),
                     xytext=(8, -8), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

for s in st.session_state.shots:
    h, v, color = s["h_in"], s["v_in"], s["color"]
    if s["off_target"]:
        angle = math.atan2(v, h)
        clip_x = TARGET_RADIUS_IN * 0.98 * math.cos(angle)
        clip_y = TARGET_RADIUS_IN * 0.98 * math.sin(angle)
        axes[1].scatter(clip_x, clip_y, s=140, marker='X', c=color, edgecolors='k', linewidths=1.0, zorder=6)
        axes[1].annotate(f"Shot {s['index']} OFF\n{h:+.2f}, {v:+.2f} in", xy=(clip_x, clip_y),
                         xytext=(8, -8), textcoords='offset points', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
    else:
        axes[1].scatter(h, v, s=90, c=color, edgecolors='k', linewidths=0.8, zorder=6)
        axes[1].annotate(f"Shot {s['index']} ({s['cant']:+.2f}°)\n{h:+.2f}, {v:+.2f} in",
                         xy=(h, v), xytext=(8, -8), textcoords='offset points', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

if st.session_state.shots:
    handles, labels = [], []
    for s in st.session_state.shots:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=s['color'], markeredgecolor='k', markersize=8))
        labels.append(f"Shot {s['index']} ({s['cant']:+.2f}°)" + (" — OFF" if s['off_target'] else ""))
    axes[1].legend(handles, labels, loc='upper right', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "- Sight height constrained to **2.5–5.0 in**.\n"
    "- Cant range is **±60°**.\n"
    "- Impacts outside the target disk are labeled **OFF TARGET**.\n"
    "- Bore angle computed from sight height + zero range (gravity-only)."
)
