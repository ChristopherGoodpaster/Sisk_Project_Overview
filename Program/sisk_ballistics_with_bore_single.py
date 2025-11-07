# sisk_ballistics_with_bore_single.py
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
SHOT_COLORS = ["red", "blue", "green"]

# ---------- Helpers ----------
def compute_zero_bore_angle(v0, zero_range_yd, sight_height_in):
    R0_ft = zero_range_yd * FT_PER_YD
    R0_in = R0_ft * IN_PER_FT
    t0 = R0_ft / v0
    drop_ft = 0.5 * G_FT_S2 * (t0 ** 2)
    drop_in = drop_ft * IN_PER_FT
    theta_rad = math.atan2(drop_in + sight_height_in, R0_in)
    return theta_rad, drop_in

def compute_impact_at_range(v0, bore_angle_rad, sight_height_in, range_yd, cant_deg):
    R_ft = range_yd * FT_PER_YD
    cos_t = math.cos(bore_angle_rad)
    t_sec = float('inf') if cos_t <= 1e-12 else R_ft / (v0 * cos_t)
    y_ft = v0 * math.sin(bore_angle_rad) * t_sec - 0.5 * G_FT_S2 * (t_sec ** 2)
    y_in = y_ft * IN_PER_FT
    y_rel_LOS_in = y_in - sight_height_in
    cant = math.radians(cant_deg)
    x0, y0 = 0.0, y_rel_LOS_in
    h_in = x0 * math.cos(cant) - y0 * math.sin(cant)
    v_in = x0 * math.sin(cant) + y0 * math.cos(cant)
    return h_in, v_in, t_sec, y_rel_LOS_in

def init_session_state():
    defaults = {
        "shots": [],
        "range_yd": 200,
        "cant_deg": 0.0,
        "show_grid": False,
        "sight_height": 2.5,
        "zero_range": 100
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

def add_shot(index: int, target_radius_in: float):
    theta_rad, _ = compute_zero_bore_angle(
        MUZZLE_V_FPS, st.session_state.zero_range, st.session_state.sight_height
    )
    h_in, v_in, t_sec, y_rel = compute_impact_at_range(
        MUZZLE_V_FPS, theta_rad, st.session_state.sight_height,
        st.session_state.range_yd, st.session_state.cant_deg
    )
    off = math.hypot(h_in, v_in) > target_radius_in
    if len(st.session_state.shots) < 3:
        st.session_state.shots.append({
            "index": index,
            "cant": float(st.session_state.cant_deg),
            "h_in": float(h_in),           # stored relative to LOS center
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

# normalize types
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

    # adjustable target radius
    target_radius_in = st.number_input("Target radius (in)", min_value=2.0, max_value=36.0, value=12.0, step=0.5)

    # aim offsets (positive = up/right)
    st.markdown("### Aim offsets (point scope here relative to target center)")
    aim_up_in = st.number_input("Aim up (in)", min_value=-36.0, max_value=36.0, value=0.0, step=0.1)
    aim_right_in = st.number_input("Aim right (in)", min_value=-36.0, max_value=36.0, value=0.0, step=0.1)

    st.write(f"Muzzle velocity locked at **{MUZZLE_V_FPS:.0f} fps**")
    st.markdown("---")

    st.markdown("### Fire shots (click while cant is set)")
    cols = st.columns(3)
    for i, col in enumerate(cols, start=1):
        with col:
            st.button(
                f"Shoot {i}",
                on_click=add_shot,
                args=(i, target_radius_in,),
                disabled=(len(st.session_state.shots) >= 3)
            )

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

h_in, v_in, t_sec, y_rel_LOS_in = compute_impact_at_range(
    MUZZLE_V_FPS, theta_rad, st.session_state.sight_height,
    st.session_state.range_yd, st.session_state.cant_deg
)
off_preview = math.hypot(h_in, v_in) > target_radius_in

st.subheader("Current preview (gravity-only)")
pp1, pp2, pp3 = st.columns(3)
pp1.metric("Time of flight (s)", f"{t_sec:.4f}")
pp2.metric("Bullet vs LOS (in)", f"{y_rel_LOS_in:+.2f}")
pp3.metric("Cant (deg)", f"{st.session_state.cant_deg:+.2f}")

# cant error estimate (renamed)
cant_rad = math.radians(st.session_state.cant_deg)
cant_error_in = abs(y_rel_LOS_in) * abs(math.sin(cant_rad))
st.write(f"Estimated cant error: **{cant_error_in:.2f} in**  *(|y_vs_LOS| × sin|cant|)*")

# will it be off target given current aim offsets?
adjusted_h = h_in - aim_right_in
adjusted_v = v_in - aim_up_in
will_off = (math.hypot(adjusted_h, adjusted_v) > target_radius_in) or (abs(v_in) > target_radius_in)
if will_off:
    st.warning("At this range/zero/cant/aim, the impact is **outside** the target radius. Enable auto-zoom to see it.")

st.caption("Tip: If **Range == Zero**, y vs. LOS ≈ 0 ⇒ cant error is small. Try Zero = 100 yd, Range = 400–600 yd to see a larger effect.")

if math.hypot(adjusted_h, adjusted_v) > target_radius_in:
    st.error(f"Preview at {st.session_state.range_yd} yd → OFF TARGET (impact rel center = h={adjusted_h:+.2f} in, v={adjusted_v:+.2f} in)")
else:
    st.write(f"Preview at {st.session_state.range_yd} yd → impact rel center = h = **{adjusted_h:+.2f} in**, v = **{adjusted_v:+.2f} in**")

# ---------- Single Target Plot (Impacts Only) ----------
auto_zoom = st.checkbox("Auto-zoom to include all impacts", value=True)

fig, ax = plt.subplots(figsize=(6.8, 6.8))

def draw_target(ax, title, show_grid=False, radius=12.0):
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    rings = [1, 3, 6, 9, radius]
    for r in rings:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.7))
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    if show_grid:
        ticks = np.arange(-radius, radius + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linewidth=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

title = f"Impacts at {st.session_state.range_yd} yd (target center = 0,0)"
draw_target(ax, title, show_grid=st.session_state.show_grid, radius=target_radius_in)

# draw aim marker (where LOS intersects target plane relative to center)
ax.plot(aim_right_in, aim_up_in, marker='+', markersize=14, color='gray', markeredgewidth=2, zorder=5)
ax.annotate("Aim point", xy=(aim_right_in, aim_up_in), xytext=(8, 8), textcoords='offset points', fontsize=8)

# draw preview impact (adjusted by aim offsets)
adj_h_in = h_in - aim_right_in
adj_v_in = v_in - aim_up_in
ax.scatter(adj_h_in, adj_v_in, s=80, facecolors='none', edgecolors='k', linewidths=1.0, zorder=6)
if math.hypot(adj_h_in, adj_v_in) > target_radius_in:
    ax.annotate("PREVIEW: OFF TARGET", xy=(0.5, 0.95), xycoords='axes fraction',
                color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
else:
    ax.annotate(f"Preview\n{adj_h_in:+.2f} in, {adj_v_in:+.2f} in", xy=(adj_h_in, adj_v_in),
                xytext=(8, -8), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# Recorded shots: display adjusted positions using current aim offsets
for s in st.session_state.shots:
    raw_h, raw_v, color = s["h_in"], s["v_in"], s["color"]
    display_h = raw_h - aim_right_in
    display_v = raw_v - aim_up_in
    if math.hypot(display_h, display_v) > target_radius_in:
        angle = math.atan2(display_v, display_h)
        clip_x = target_radius_in * 0.98 * math.cos(angle)
        clip_y = target_radius_in * 0.98 * math.sin(angle)
        ax.scatter(clip_x, clip_y, s=140, marker='X', c=color, edgecolors='k', linewidths=1.0, zorder=7)
        ax.annotate(f"Shot {s['index']} OFF\n{display_h:+.2f}, {display_v:+.2f} in", xy=(clip_x, clip_y),
                    xytext=(8, -8), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
    else:
        ax.scatter(display_h, display_v, s=90, c=color, edgecolors='k', linewidths=0.8, zorder=7)
        ax.annotate(f"Shot {s['index']} ({s['cant']:+.2f}°)\n{display_h:+.2f}, {display_v:+.2f} in",
                    xy=(display_h, display_v), xytext=(8, -8), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

# auto-zoom logic
xs = [adj_h_in] + [s["h_in"] - aim_right_in for s in st.session_state.shots]
ys = [adj_v_in] + [s["v_in"] - aim_up_in for s in st.session_state.shots]
if auto_zoom and xs and ys:
    max_abs = max(1.0, max(map(abs, xs + ys)))
    pad = 0.1 * max_abs
    xmin = min(-target_radius_in, min(xs) - pad)
    xmax = max(target_radius_in, max(xs) + pad)
    ymin = min(-target_radius_in, min(ys) - pad)
    ymax = max(target_radius_in, max(ys) + pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
else:
    ax.set_xlim(-target_radius_in, target_radius_in)
    ax.set_ylim(-target_radius_in, target_radius_in)

# legend
if st.session_state.shots:
    handles, labels = [], []
    for s in st.session_state.shots:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=s['color'], markeredgecolor='k', markersize=8))
        labels.append(f"Shot {s['index']} ({s['cant']:+.2f}°)" + (" — OFF" if s['off_target'] else ""))
    ax.legend(handles, labels, loc='upper right', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "- Target radius is adjustable.\n"
    "- Aim offsets shift the LOS intersection point on the target plane; impacts shown are relative to target center.\n"
    "- Recorded shots are stored relative to LOS; their displayed positions update when aim offsets change.\n"
    "- Estimated cant error shown: |y_vs_LOS| × sin(|cant|).\n"
)
