"""
============================================================
AI-Powered Smart Hydration Tracker
Python Dashboard — Production Code

Receives sensor data from ESP32-CAM via Serial,
runs TinyML inference, pushes to Firebase,
and displays real-time analytics.

Authors: Shaik Tasneem Kauser, V G Manjusree, Dharani S
Supervisor: Dr. Poornima N
VIT Vellore — Capstone Project BCSE498J
============================================================
"""

import streamlit as st
import serial
import pandas as pd
import numpy as np
import time
import joblib
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from firebase_helper import (
    init_firebase, push_sip,
    push_daily_summary, push_ml_prediction
)

# ============================================================
# CONFIGURATION
# ============================================================
PORT                  = 'COM5'          # Update to your COM port
BAUD                  = 115200
TILT_DRINK_THRESHOLD  = 35.0            # degrees
TILT_REST_THRESHOLD   = 20.0            # degrees
SETTLE_READINGS       = 4               # consecutive stable readings
WEIGHT_SAMPLES        = 8               # samples for median baseline
MAX_WEIGHT_G          = 1500.0          # sanity cap
DAILY_GOAL_ML         = 2500.0          # recommended daily intake
ALERT_THRESHOLD_SECS  = 2700            # 45 minutes
USER_ID               = 'tasneem'

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="AquaTrack — Smart Hydration",
    page_icon="💧",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container { padding-top:1rem; padding-bottom:1rem; }
    [data-testid="metric-container"] {
        background:#161b27; border:1px solid #1f2937;
        border-radius:12px; padding:14px 18px;
    }
    [data-testid="metric-container"] label {
        color:#6b7280 !important; font-size:12px !important;
        text-transform:uppercase; letter-spacing:0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color:#f9fafb !important; font-size:1.5rem !important;
        font-weight:600 !important;
    }
    section[data-testid="stSidebar"] { background:#0d1117; }
    section[data-testid="stSidebar"] * { color:#d1d5db !important; }
    .goal-bar-wrap {
        background:#1f2937; border-radius:8px;
        height:10px; width:100%; overflow:hidden; margin-top:4px;
    }
    .goal-bar-fill {
        height:100%; border-radius:8px;
        background:linear-gradient(90deg,#3b82f6,#06b6d4);
    }
    .pred-card {
        background:#0f172a; border:1px solid #1e3a5f;
        border-radius:10px; padding:14px; margin-bottom:8px;
    }
    .pred-label {
        color:#60a5fa; font-size:11px;
        text-transform:uppercase; letter-spacing:.08em;
    }
    .pred-value { color:#f0f9ff; font-size:22px; font-weight:700; margin:2px 0; }
    .pred-sub   { color:#6b7280; font-size:12px; }
    .model-badge {
        background:#052e16; border:1px solid #166534;
        border-radius:8px; padding:8px 12px; margin-bottom:8px; font-size:12px;
    }
    .firebase-badge {
        background:#1a1a2e; border:1px solid #f5a623;
        border-radius:8px; padding:8px 12px; margin-bottom:8px; font-size:12px;
    }
    .inference-box {
        background:#0f172a; border:1px solid #1e3a5f;
        border-radius:10px; padding:12px; margin-top:8px; font-size:12px;
    }
    .fb-log-item {
        background:#0f172a; border-left:3px solid #f5a623;
        padding:6px 10px; margin-bottom:4px;
        border-radius:0 6px 6px 0; font-size:12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INIT FIREBASE
# ============================================================
firebase_ok = init_firebase()

# ============================================================
# LOAD TINYML MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model  = joblib.load('hydration_model.pkl')
        scaler = joblib.load('hydration_scaler.pkl')
        with open('model_meta.json') as f:
            meta = json.load(f)
        return model, scaler, meta
    except Exception as e:
        st.error(f"Model files not found: {e}. Run train_model.py first.")
        st.stop()

model, scaler, meta = load_model()

def run_tinyml_inference(mins_since_last, hour_of_day, drinks_today,
                          avg_interval, activity_level):
    """
    Runs trained GradientBoosting model to predict hydration risk.
    Features match training schema exactly.
    Returns: (will_drink_soon, risk_score, confidence, pred_mins)
    """
    if   6  <= hour_of_day < 10: tod = 1
    elif 10 <= hour_of_day < 14: tod = 2
    elif 14 <= hour_of_day < 18: tod = 3
    else:                        tod = 4

    X = np.array([[
        mins_since_last, hour_of_day, drinks_today,
        avg_interval, activity_level, tod
    ]])
    X_scaled     = scaler.transform(X)
    prediction   = model.predict(X_scaled)[0]
    proba        = model.predict_proba(X_scaled)[0]
    confidence   = float(proba[prediction] * 100)
    risk_score   = float(proba[0] * 100)

    pred_mins = (max(1.0, avg_interval * (1 - proba[1]) * 2)
                 if prediction == 1
                 else avg_interval * proba[0])
    pred_mins = round(float(np.clip(pred_mins, 0.5, 45.0)), 1)

    return bool(prediction), risk_score, confidence, pred_mins

# ============================================================
# SERIAL CONNECTION
# ============================================================
@st.cache_resource
def get_serial(port, baud):
    try:
        s = serial.Serial(port, baud, timeout=0.1)
        s.reset_input_buffer()
        return s
    except Exception as e:
        return None

ser = get_serial(PORT, BAUD)
if not ser:
    st.error(
        f"❌ Cannot open {PORT}. "
        "Close Arduino Serial Monitor, check USB connection, then restart."
    )
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
def build_hourly_history():
    """Pre-populate hourly chart with sample data for demo."""
    now   = datetime.now()
    hours, mls = [], []
    for h in range(max(0, now.hour - 5), now.hour):
        hours.append(f"{h:02d}:00")
        mls.append(round(np.random.uniform(150, 340)))
    return hours, mls

_h, _m = build_hourly_history()

defaults = {
    'total_intake':        0.0,
    'state':               'IDLE',
    'start_weight':        0.0,
    'settle_count':        0,
    'last_sip_time':       datetime.now(),
    'sip_log':             [],
    'firebase_log':        [],
    'weight_buffer':       [],
    'df':                  pd.DataFrame(columns=['Time','Weight','Tilt','State']),
    'hourly_hours':        _h,
    'hourly_mls':          _m,
    # TinyML state
    'ml_prediction':       True,
    'ml_risk_score':       10.0,
    'ml_confidence':       85.0,
    'ml_pred_mins':        8.0,
    'ml_inference_count':  0,
    'ml_last_features':    {},
    'avg_interval':        35.0,
    'activity_level':      1,
    # Firebase state
    'firebase_ok':         firebase_ok,
    'fb_push_count':       0,
    'fb_last_push':        '—',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# LAYOUT
# ============================================================
with st.sidebar:
    st.markdown("## 💧 AquaTrack")
    st.markdown("---")

    # Daily goal progress bar
    goal_pct = min(st.session_state.total_intake / DAILY_GOAL_ML, 1.0)
    st.markdown("**Daily Goal Progress**")
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:13px;color:#9ca3af;margin-bottom:4px'>"
        f"<span>{st.session_state.total_intake:.0f} ml</span>"
        f"<span>{DAILY_GOAL_ML:.0f} ml goal</span></div>"
        f"<div class='goal-bar-wrap'>"
        f"<div class='goal-bar-fill' style='width:{goal_pct*100:.1f}%'></div>"
        f"</div>"
        f"<div style='font-size:12px;color:#6b7280;margin-top:4px'>"
        f"{(1-goal_pct)*DAILY_GOAL_ML:.0f} ml remaining</div>",
        unsafe_allow_html=True)

    st.markdown("---")

    # TinyML model badge
    st.markdown(
        f"<div class='model-badge'>"
        f"✅ TinyML Model Active<br>"
        f"<span style='color:#86efac'>"
        f"Accuracy: {meta['accuracy']}% · {meta['n_samples']} samples</span>"
        f"</div>", unsafe_allow_html=True)

    # Firebase badge
    fb_color = "#f5a623" if st.session_state.firebase_ok else "#f87171"
    fb_label = "Connected" if st.session_state.firebase_ok else "Offline"
    st.markdown(
        f"<div class='firebase-badge'>"
        f"<span style='color:{fb_color}'>🔥 Firebase: {fb_label}</span><br>"
        f"<span style='color:#6b7280;font-size:11px'>"
        f"Pushes: {st.session_state.fb_push_count} · "
        f"Last: {st.session_state.fb_last_push}</span>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("### 🧠 Live TinyML Inference")
    pred_panel   = st.empty()
    risk_panel   = st.empty()
    alert_banner = st.empty()

    st.markdown("---")
    st.markdown("### 🔬 Model Input Features")
    features_panel = st.empty()

    st.markdown("---")
    st.markdown("### 🔥 Firebase Live Feed")
    fb_feed_panel = st.empty()

    st.markdown("---")
    st.markdown(f"**User:** {USER_ID.title()}")
    st.markdown("**Device:** ESP32-CAM + MPU6050 + HX711")
    st.markdown(f"**Model:** {meta['model']}")
    st.markdown(f"**Accuracy:** {meta['accuracy']}%")
    st.markdown("**Cloud:** Firebase Realtime DB")
    st.markdown("**Firmware:** v2.1.4")
    st.markdown("---")
    st.markdown("### 🔌 Live Diagnostics")
    diag_panel = st.empty()
    st.markdown("---")
    st.markdown("### 📋 Sip Log")
    log_panel = st.empty()

# Main area
st.markdown("## 💧 AquaTrack — Smart Hydration Monitor")
col1, col2, col3, col4 = st.columns(4)
intake_box    = col1.empty()
status_box    = col2.empty()
last_sip_box  = col3.empty()
sip_count_box = col4.empty()

chart_col, bar_col = st.columns([2, 1])
chart_box  = chart_col.empty()
hourly_box = bar_col.empty()

# ============================================================
# MAIN LOOP
# ============================================================
inference_tick = 0
last_tilt      = 5.0

while True:

    # ── 1. Read serial data from ESP32 ───────────────────────
    curr_weight, curr_tilt, hw_state = None, None, None

    try:
        if ser.in_waiting > 0:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()

            # Skip header and event lines
            if (not raw or raw.startswith('Weight') or
                    raw.startswith('EVENT') or raw.startswith('INFO')):
                time.sleep(0.01)
                continue

            parts = raw.split(',')
            if len(parts) >= 2:
                curr_weight = float(parts[0])
                curr_tilt   = float(parts[1])
                hw_state    = parts[2] if len(parts) > 2 else 'IDLE'
                last_tilt   = curr_tilt

    except Exception as e:
        diag_panel.error(f"Serial error: {e}")
        time.sleep(0.05)
        continue

    if curr_weight is None:
        diag_panel.info("⏳ Waiting for sensor data from ESP32...")
        time.sleep(0.05)
        continue

    # ── 2. Sanitize weight ────────────────────────────────────
    if curr_weight > MAX_WEIGHT_G or curr_weight < -50:
        curr_weight = (st.session_state.df['Weight'].iloc[-1]
                       if not st.session_state.df.empty else 0.0)

    # ── 3. Append to rolling dataframe ───────────────────────
    new_row = {
        'Time':   datetime.now(),
        'Weight': curr_weight,
        'Tilt':   curr_tilt,
        'State':  hw_state or 'IDLE'
    }
    st.session_state.df = pd.concat(
        [st.session_state.df, pd.DataFrame([new_row])],
        ignore_index=True
    ).tail(300)

    st.session_state.weight_buffer.append(curr_weight)
    if len(st.session_state.weight_buffer) > 50:
        st.session_state.weight_buffer.pop(0)

    recent  = st.session_state.df['Weight'].tail(WEIGHT_SAMPLES)
    inference_tick += 1

    # ── 4. Python-side state machine ─────────────────────────
    # (mirrors ESP32 state machine — Python side logs to Firebase)
    state = st.session_state.state

    if state == 'IDLE':
        if curr_tilt < TILT_REST_THRESHOLD:
            st.session_state.start_weight = float(recent.median())
        if curr_tilt > TILT_DRINK_THRESHOLD:
            st.session_state.state        = 'DRINKING'
            st.session_state.settle_count = 0
            st.session_state.activity_level = min(
                7, st.session_state.activity_level + 1)

    elif state == 'DRINKING':
        if curr_tilt < TILT_REST_THRESHOLD:
            st.session_state.state        = 'RETURNING'
            st.session_state.settle_count = 0

    elif state == 'RETURNING':
        if curr_tilt < TILT_REST_THRESHOLD:
            st.session_state.settle_count += 1
        else:
            st.session_state.settle_count = 0

        if st.session_state.settle_count >= SETTLE_READINGS:
            end_weight = float(recent.median())
            sip_ml     = st.session_state.start_weight - end_weight

            if 15 < sip_ml < 600:
                st.session_state.total_intake  += sip_ml
                st.session_state.last_sip_time  = datetime.now()
                ts = datetime.now().strftime('%H:%M:%S')

                st.session_state.sip_log.append(
                    f"{ts}  ·  **{sip_ml:.0f} ml**  "
                    f"({st.session_state.start_weight:.0f}g → {end_weight:.0f}g)"
                )

                # ── Firebase: push drink event ────────────────
                if st.session_state.firebase_ok:
                    ok = push_sip(
                        user_id      = USER_ID,
                        sip_ml       = sip_ml,
                        start_weight = st.session_state.start_weight,
                        end_weight   = end_weight,
                        tilt_angle   = last_tilt
                    )
                    push_daily_summary(
                        user_id   = USER_ID,
                        total_ml  = st.session_state.total_intake,
                        sip_count = len(st.session_state.sip_log),
                        goal_ml   = DAILY_GOAL_ML
                    )
                    if ok:
                        st.session_state.fb_push_count += 1
                        st.session_state.fb_last_push   = ts
                        st.session_state.firebase_log.append(
                            f"{ts} · {sip_ml:.0f} ml ✅"
                        )
                # ─────────────────────────────────────────────

                # Update hourly chart
                hour_label = datetime.now().strftime('%H:00')
                if (st.session_state.hourly_hours and
                        st.session_state.hourly_hours[-1] == hour_label):
                    st.session_state.hourly_mls[-1] += sip_ml
                else:
                    st.session_state.hourly_hours.append(hour_label)
                    st.session_state.hourly_mls.append(sip_ml)

                # Update average interval
                total_sips = len(st.session_state.sip_log)
                if total_sips > 1:
                    elapsed = (datetime.now() -
                               st.session_state.last_sip_time
                               + timedelta(seconds=1)).total_seconds()
                    st.session_state.avg_interval = round(
                        elapsed / 60.0, 1)

                st.toast(f"💧 {sip_ml:.0f} ml logged!", icon="✅")

            st.session_state.state        = 'IDLE'
            st.session_state.settle_count = 0

    # ── 5. Dehydration alert ──────────────────────────────────
    elapsed_secs = (datetime.now() -
                    st.session_state.last_sip_time).total_seconds()

    if elapsed_secs > ALERT_THRESHOLD_SECS:
        try:
            ser.write(b'A')  # Trigger buzzer on ESP32
        except Exception:
            pass

    # ── 6. TinyML inference every 10 ticks (~3 seconds) ──────
    if inference_tick % 10 == 0:
        mins_since   = elapsed_secs / 60.0
        hour_now     = datetime.now().hour
        drinks_today = len(st.session_state.sip_log)
        activity     = st.session_state.activity_level

        # Decay activity level slowly over time
        if inference_tick % 50 == 0:
            st.session_state.activity_level = max(
                0, st.session_state.activity_level - 1)

        pred, risk, conf, pred_mins = run_tinyml_inference(
            mins_since_last = mins_since,
            hour_of_day     = hour_now,
            drinks_today    = drinks_today,
            avg_interval    = st.session_state.avg_interval,
            activity_level  = activity
        )

        st.session_state.ml_prediction      = pred
        st.session_state.ml_risk_score      = risk
        st.session_state.ml_confidence      = conf
        st.session_state.ml_pred_mins       = pred_mins
        st.session_state.ml_inference_count += 1
        st.session_state.ml_last_features   = {
            'mins_since_last': round(mins_since, 1),
            'hour_of_day':     hour_now,
            'drinks_today':    drinks_today,
            'avg_interval':    st.session_state.avg_interval,
            'activity_level':  activity,
        }

        # Push ML log to Firebase every 5 inferences
        if (st.session_state.firebase_ok and
                st.session_state.ml_inference_count % 5 == 0):
            push_ml_prediction(
                user_id    = USER_ID,
                risk_score = risk,
                pred_mins  = pred_mins,
                confidence = conf,
                features   = st.session_state.ml_last_features
            )

    # ── 7. Update sidebar ─────────────────────────────────────
    risk  = st.session_state.ml_risk_score
    conf  = st.session_state.ml_confidence
    pmins = st.session_state.ml_pred_mins
    pred  = st.session_state.ml_prediction

    risk_color = ("#4ade80" if risk < 35
                  else "#fbbf24" if risk < 65
                  else "#f87171")
    risk_label = ("Low" if risk < 35
                  else "Moderate" if risk < 65
                  else "High")
    pred_label = "✅ Drink predicted soon" if pred else "⚠️ Lapse risk detected"
    pred_color = "#4ade80" if pred else "#f87171"

    pred_panel.markdown(
        f"<div class='pred-card'>"
        f"<div class='pred-label'>TinyML Prediction</div>"
        f"<div class='pred-value' style='color:{pred_color};font-size:16px'>"
        f"{pred_label}</div>"
        f"<div class='pred-sub'>"
        f"Next sip ~{pmins:.1f} min · Confidence: {conf:.1f}%</div>"
        f"<div class='pred-sub' style='color:#4b5563;margin-top:4px'>"
        f"Total inferences: {st.session_state.ml_inference_count}</div>"
        f"</div>", unsafe_allow_html=True)

    risk_panel.markdown(
        f"<div class='pred-card'>"
        f"<div class='pred-label'>Dehydration Risk Score</div>"
        f"<div class='pred-value' style='color:{risk_color}'>{risk:.1f}%</div>"
        f"<div class='pred-sub'>Level: {risk_label}</div>"
        f"</div>", unsafe_allow_html=True)

    if risk >= 70:
        alert_banner.error("⚠️ High dehydration risk — Alert sent to device!")
    else:
        alert_banner.empty()

    feats = st.session_state.ml_last_features
    if feats:
        features_panel.markdown(
            f"<div class='inference-box'>"
            f"<span style='color:#60a5fa'>mins_since_last:</span> "
            f"<b>{feats.get('mins_since_last','—')}</b><br>"
            f"<span style='color:#60a5fa'>hour_of_day:</span> "
            f"<b>{feats.get('hour_of_day','—')}</b><br>"
            f"<span style='color:#60a5fa'>drinks_today:</span> "
            f"<b>{feats.get('drinks_today','—')}</b><br>"
            f"<span style='color:#60a5fa'>avg_interval:</span> "
            f"<b>{feats.get('avg_interval','—')} min</b><br>"
            f"<span style='color:#60a5fa'>activity_level:</span> "
            f"<b>{feats.get('activity_level','—')}</b>"
            f"</div>", unsafe_allow_html=True)

    if st.session_state.firebase_log:
        feed_html = "".join([
            f"<div class='fb-log-item'>{e}</div>"
            for e in reversed(st.session_state.firebase_log[-5:])
        ])
        fb_feed_panel.markdown(feed_html, unsafe_allow_html=True)
    else:
        fb_feed_panel.caption("Waiting for first sync...")

    # ── 8. Main metrics ───────────────────────────────────────
    STATE_LABEL = {
        'IDLE':      '✅  Monitoring',
        'DRINKING':  '🥤  Drinking...',
        'RETURNING': '🔄  Settling...'
    }
    goal_pct_now = min(
        st.session_state.total_intake / DAILY_GOAL_ML * 100, 100)
    last_sip_str = "—"
    if st.session_state.sip_log:
        parts = st.session_state.sip_log[-1].split('**')
        last_sip_str = parts[1] if len(parts) >= 2 else "—"

    total_sips = len(st.session_state.sip_log)
    avg_sip    = (st.session_state.total_intake / total_sips
                  if total_sips > 0 else 0)

    intake_box.metric("Total Intake Today",
                      f"{st.session_state.total_intake:.0f} ml",
                      delta=f"{goal_pct_now:.1f}% of daily goal")
    status_box.metric("System Status",
                      STATE_LABEL.get(st.session_state.state, '—'))
    last_sip_box.metric("Last Sip", last_sip_str)
    sip_count_box.metric("Total Sips", str(total_sips),
                         delta=f"Avg {avg_sip:.0f} ml/sip"
                         if total_sips > 0 else None)

    # ── 9. Sensor fusion chart ────────────────────────────────
    df       = st.session_state.df
    smooth_w = df['Weight'].rolling(window=4, min_periods=1).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df['Time'], y=smooth_w, name='Weight (g)',
        line=dict(color='#38bdf8', width=2.5),
        fill='tozeroy', fillcolor='rgba(56,189,248,0.07)'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['Tilt'], name='Tilt angle (°)',
        line=dict(color='#fb923c', width=1.5, dash='dot')
    ), secondary_y=True)
    fig.add_hline(
        y=TILT_DRINK_THRESHOLD, line_dash="dash",
        line_color="#ef4444", line_width=1,
        annotation_text="Drink threshold (35°)",
        annotation_position="bottom right",
        annotation_font_color="#ef4444",
        secondary_y=True)
    fig.update_layout(
        title=dict(text="Live Sensor Fusion — Weight & Tilt",
                   font=dict(size=14, color='#9ca3af')),
        height=340, plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#d1d5db', margin=dict(l=0, r=10, t=36, b=0),
        legend=dict(orientation="h", y=1.14,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
        xaxis=dict(gridcolor='#1f2937', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1f2937', showgrid=True, zeroline=False),
    )
    fig.update_yaxes(title_text="Weight (g)", range=[0, 600],
                     secondary_y=False, gridcolor='#1f2937',
                     title_font=dict(color='#38bdf8'))
    fig.update_yaxes(title_text="Tilt (°)", range=[0, 90],
                     secondary_y=True, gridcolor='rgba(0,0,0,0)',
                     title_font=dict(color='#fb923c'))
    chart_box.plotly_chart(fig, use_container_width=True,
                           key=f"main_{time.time()}")

    # ── 10. Hourly intake bar chart ───────────────────────────
    if st.session_state.hourly_hours:
        n = len(st.session_state.hourly_hours)
        bar_fig = go.Figure(go.Bar(
            x=st.session_state.hourly_hours,
            y=st.session_state.hourly_mls,
            marker_color=[
                '#38bdf8' if i == n - 1 else '#1e3a5f'
                for i in range(n)
            ],
            text=[f"{v:.0f}" for v in st.session_state.hourly_mls],
            textposition='outside',
            textfont=dict(color='#9ca3af', size=11)
        ))
        bar_fig.update_layout(
            title=dict(text="Hourly Intake (ml)",
                       font=dict(size=14, color='#9ca3af')),
            height=340, plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117', font_color='#d1d5db',
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis=dict(gridcolor='#1f2937'),
            yaxis=dict(gridcolor='#1f2937', title='ml', range=[0, 450])
        )
        bar_fig.add_hline(
            y=250, line_dash="dot", line_color="#4ade80",
            annotation_text="Recommended/hr",
            annotation_font_color="#4ade80", line_width=1)
        hourly_box.plotly_chart(bar_fig, use_container_width=True,
                                key=f"hourly_{time.time()}")

    # ── 11. Diagnostics ───────────────────────────────────────
    diag_panel.markdown(
        f"**Weight:** `{curr_weight:.1f} g`  \n"
        f"**Tilt:** `{curr_tilt:.1f}°`  \n"
        f"**HW State:** `{hw_state}`  \n"
        f"**SW State:** `{st.session_state.state}`  \n"
        f"**Baseline:** `{st.session_state.start_weight:.1f} g`  \n"
        f"**Settle:** `{st.session_state.settle_count}/{SETTLE_READINGS}`"
    )

    # ── 12. Sip log ───────────────────────────────────────────
    if st.session_state.sip_log:
        log_panel.markdown(
            "\n\n".join(reversed(st.session_state.sip_log[-8:]))
        )
    else:
        log_panel.caption("Waiting for first sip...")

    time.sleep(0.05)