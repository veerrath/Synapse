import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from plyer import notification
import simpleaudio as sa
import plotly.graph_objs as go

st.set_page_config(layout="wide", page_title="Synapse — Focus Tracker", page_icon="🧠")

defaults = {
    'posture_history': [], 'head_tilt_history': [], 'yawn_times': [], 'phone_glances': [],
    'start_time': time.time(), 'session_active': False, 'session_ended': False,
    'phone_glance_active': False, 'yawn_active': False,
    'calibrated_posture': None, 'calibrated_head_tilt': None,
    'latest_pose_landmarks': None, 'baseline_nose_y': None,
    'last_notified': {}, 'notification_cooldown': 20.0, 'calibration_done': False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# MediaPipe

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
NOSE = mp_pose.PoseLandmark.NOSE.value
FOREHEAD_IDX = 10
MOUTH_IDX = [61, 291, 13, 14, 78, 308, 311, 402, 0, 17]

def now(): return time.time()

def can_notify(event_key: str):
    last = st.session_state.get('last_notified', {}).get(event_key, 0)
    cooldown = float(st.session_state.get('notification_cooldown', 20.0))
    return (now() - last) >= cooldown

def record_notification(event_key: str):
    if 'last_notified' not in st.session_state:
        st.session_state['last_notified'] = {}
    st.session_state['last_notified'][event_key] = now()

def play_alert_sound(file):
    try:
        sa.WaveObject.from_wave_file(file).play()
    except Exception:
        pass

def notify_desktop_plyer(title, message):
    try:
        notification.notify(title=title, message=message, timeout=4)
    except Exception:
        pass

def notify_browser(title, message, play_beep=False):
    beep_js = """
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(880, ctx.currentTime);
      o.connect(ctx.destination);
      o.start();
      setTimeout(()=>{o.stop(); ctx.close();}, 150);
    } catch(e) { console.log(e); }
    """ if play_beep else ""
    html = f"""
    <script>
    const sendNotification = async () => {{
      if (!("Notification" in window)) return;
      if (Notification.permission === "granted") {{
        new Notification({title!r}, {{ body: {message!r} }} );
      }} else if (Notification.permission !== "denied") {{
        let p = await Notification.requestPermission();
        if (p === "granted") new Notification({title!r}, {{ body: {message!r} }} );
      }}
      {beep_js}
    }};
    sendNotification();
    </script>
    """
    st.components.v1.html(html, height=0)

def notify(title, message, event_key, sound_file="alert.wav", play_beep_browser=True):
    if not can_notify(event_key):
        return
    record_notification(event_key)
    threading.Thread(target=play_alert_sound, args=(sound_file,)).start()
    threading.Thread(target=notify_desktop_plyer, args=(title, message)).start()
    notify_browser(title, message, play_beep=play_beep_browser)

def compute_posture_score(lm, initial_y, factor=5):
    avg_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
    score = np.clip(0.5 + (initial_y - avg_y) * factor, 0, 1) * 100
    return score

def compute_head_tilt(nose, forehead, left_sh, right_sh):
    mid_x = (left_sh.x + right_sh.x) / 2
    mid_y = (left_sh.y + right_sh.y) / 2
    vec = np.array([forehead.x - mid_x, forehead.y - mid_y])
    vertical = np.array([0, 1])
    denom = (np.linalg.norm(vec) * np.linalg.norm(vertical))
    if denom == 0:
        return 0.0
    cos_theta = np.dot(vec, vertical) / denom
    cos_theta = np.clip(cos_theta, -1, 1)
    tilt = np.degrees(np.arccos(cos_theta))
    if tilt > 90:
        tilt = 180 - tilt
    return tilt

def compute_head_tilt_score(nose, forehead, left_sh, right_sh, per_degree=4):
    return compute_head_tilt(nose, forehead, left_sh, right_sh) * per_degree

def compute_mouth_ratio(lm, mouth_points):
    coords = np.array([[lm[i].x, lm[i].y] for i in mouth_points])
    if coords.shape[0] < 8:
        return 0
    ver = np.mean([
        np.linalg.norm(coords[2]-coords[3]),
        np.linalg.norm(coords[4]-coords[5]),
        np.linalg.norm(coords[6]-coords[7])
    ])
    hor = np.linalg.norm(coords[0]-coords[1]) if coords.shape[0] > 1 else 1.0
    return ver / hor if hor != 0 else 0

# Sidebar

st.title("Synapse — Focus Tracker")
with st.sidebar:
    st.markdown("## Controls & Settings")
    start_btn = st.button("Start Session", key="start_btn")
    stop_btn = st.button("End Session", key="stop_btn")
    st.markdown("---")
    st.markdown("### Detection Sensitivity")
    st.session_state['notification_cooldown'] = st.slider("Notification cooldown (s)", 5, 60, int(st.session_state['notification_cooldown']))
    yawn_threshold = st.slider("Yawn threshold (mouth ratio)", 0.2, 1.5, 0.6, 0.05)
    st.markdown("---")
    calibrate_btn = st.button("Calibrate Posture & Head (optional)")
    st.markdown("Tips: calibrate while sitting upright and looking at the camera.")


# Landing dashboard

if not st.session_state['session_active'] and not st.session_state['session_ended']:
    st.header("Welcome to Synapse")
    st.write("Synapse monitors your posture and neck position to ensure your body stays healthy. It also detecs when you are tired and suggests breaks to ensure you don't burn out. Lastly, it tracks when you go on your phone and notifies to to stay focused.")
    st.write("How to use:")
    st.write("- Choose the cooldown in between notifications using the slider.")
    st.write("- Choose how sensitive the yawn detection will be using the slider. Lower value = more sensitive, Higher value = less sensitive.")
    st.write("- Click **Start Session** to begin tracking.")
    st.write("- Click **End Session** once you are done and see your session summary.")
    st.info("Make sure your webcam is connected and your face is visible.")


# Start/End session

if start_btn:
    st.session_state['posture_history'] = []
    st.session_state['head_tilt_history'] = []
    st.session_state['yawn_times'] = []
    st.session_state['phone_glances'] = []
    st.session_state['start_time'] = now()
    st.session_state['session_active'] = True
    st.session_state['session_ended'] = False
    st.session_state['baseline_nose_y'] = None
    st.session_state['calibration_done'] = False
    st.session_state['last_notified'] = {}

if stop_btn:
    st.session_state['session_active'] = False
    st.session_state['session_ended'] = True

# Live session UI

if st.session_state['session_active']:
    main_col, right_col = st.columns([2.2, 1])
    frame_placeholder = main_col.empty()
    with right_col:
        st.markdown("### Live Metrics")
        posture_ph = st.empty()
        headtilt_ph = st.empty()
        glance_ph = st.empty()
        yawn_ph = st.empty()
        if st.session_state['calibrated_posture'] is not None:
            st.info(f"Calibrated posture score: {st.session_state['calibrated_posture']:.1f}")
        if st.session_state['calibrated_head_tilt'] is not None:
            st.info(f"Calibrated head-tilt score: {st.session_state['calibrated_head_tilt']:.1f}")

    # Camera
    cap = cv2.VideoCapture(0)
    initial_shoulder_y = None
    update_interval = 0.28
    GLANCE_THRESHOLD = 0.015   # small nose drop
    POSTURE_IGNORE = 5
    YAWN_THRESHOLD = yawn_threshold

    # Main streaming loop
    try:
        while st.session_state['session_active']:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read webcam frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame_rgb)
            results_face = face_mesh.process(frame_rgb)
            debug_frame = frame.copy()
            current_time = now()

            posture_val = None
            tilt_score = None

            if results_pose.pose_landmarks:
                lm_pose = results_pose.pose_landmarks.landmark
                st.session_state['latest_pose_landmarks'] = lm_pose

                nose = lm_pose[NOSE]
                try:
                    forehead = lm_pose[FOREHEAD_IDX]
                except Exception:
                    forehead = nose

                left_sh = lm_pose[LEFT_SHOULDER]
                right_sh = lm_pose[RIGHT_SHOULDER]

                if initial_shoulder_y is None:
                    initial_shoulder_y = (left_sh.y + right_sh.y) / 2

                # initialize baseline_nose_y
                if st.session_state.get('baseline_nose_y', None) is None:
                    st.session_state['baseline_nose_y'] = nose.y

                #AUTO-CALIBRATE
                if not st.session_state.get('calibration_done', False):
                    st.session_state['calibrated_posture'] = compute_posture_score(lm_pose, initial_shoulder_y)
                    st.session_state['calibrated_head_tilt'] = compute_head_tilt_score(nose, forehead, left_sh, right_sh)
                    st.session_state['baseline_nose_y'] = nose.y
                    st.session_state['calibration_done'] = True
                    st.success("Calibration complete!")

                # posture/tilt
                posture_val = compute_posture_score(lm_pose, initial_shoulder_y)
                tilt_score = compute_head_tilt_score(nose, forehead, left_sh, right_sh)

                #Phone Glance
                GLANCE_THRESHOLD = 0.015 
                POSTURE_IGNORE = 5

                if st.session_state.get('baseline_nose_y', None) is None:
                    st.session_state['baseline_nose_y'] = nose.y

                if 'posture_history' not in st.session_state:
                    st.session_state['posture_history'] = []

                if st.session_state['posture_history']:
                    prev_posture = st.session_state['posture_history'][-1][1]
                    posture_drop = prev_posture - posture_val
                else:
                    posture_drop = 0.0

                if 'phone_glance_active' not in st.session_state:
                    st.session_state['phone_glance_active'] = False
                if 'phone_glances' not in st.session_state:
                    st.session_state['phone_glances'] = []

                # Detection
                if not st.session_state['phone_glance_active']:
                    if (st.session_state['baseline_nose_y'] - nose.y) > GLANCE_THRESHOLD and abs(posture_drop) < POSTURE_IGNORE:
                        st.session_state['phone_glances'].append(current_time)
                        st.session_state['phone_glance_active'] = True

                        # Noti/Sound
                        notify(
                            "You looked at your phone.",
                            "Get off of the phone and stay focused.",
                            event_key="phone_glance",
                            sound_file="alert.wav",
                            play_beep_browser=True
                        )
                else:
                    # Reset active flag
                    if nose.y >= st.session_state['baseline_nose_y'] - (GLANCE_THRESHOLD / 2):
                        st.session_state['phone_glance_active'] = False
                        st.session_state['baseline_nose_y'] = nose.y

                st.session_state['posture_history'].append((current_time, posture_val))
                st.session_state['head_tilt_history'].append((current_time, tilt_score))



                # shoulder line
                try:
                    pts = np.array([[int(left_sh.x * frame.shape[1]), int(left_sh.y * frame.shape[0])],
                                    [int(right_sh.x * frame.shape[1]), int(right_sh.y * frame.shape[0])]])
                    cv2.polylines(debug_frame, [pts], isClosed=False, color=(0, 200, 0), thickness=2)
                except Exception:
                    pass

                # Post/Head tilt alerts
                if st.session_state.get('calibrated_posture') is not None:
                    deviation = posture_val - st.session_state['calibrated_posture']
                    if deviation < -12:
                        notify("Fix your posture", "You're slouching. Sit up straight!", event_key="posture_bad", sound_file="alert.wav")

                if st.session_state.get('calibrated_head_tilt') is not None:
                    tilt_deviation = tilt_score - st.session_state['calibrated_head_tilt']
                    if abs(tilt_deviation) > 30:
                        notify("You're tilting your head", "Keep your head and neck straight.", event_key="head_tilt", sound_file="alert.wav")

            # ZZZZZ
            if results_face.multi_face_landmarks:
                lm_face = results_face.multi_face_landmarks[0].landmark
                mouth_ratio = compute_mouth_ratio(lm_face, MOUTH_IDX)
                if not st.session_state.get('yawn_active', False):
                    if mouth_ratio > YAWN_THRESHOLD:
                        st.session_state['yawn_times'].append(current_time)
                        st.session_state['yawn_active'] = True
                        notify("You yawned", "You seem a bit tired. Consider taking a short break and coming back in a bit.", event_key="yawn", sound_file="alert.wav")
                else:
                    if mouth_ratio <= (YAWN_THRESHOLD / 1.4):
                        st.session_state['yawn_active'] = False

            # Display metrics
            frame_placeholder.image(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            posture_ph.metric("Posture", f"{posture_val:.1f}" if posture_val is not None else "—")
            headtilt_ph.metric("Head Tilt", f"{tilt_score:.1f}" if tilt_score is not None else "—")
            glance_ph.metric("Phone Glances", len(st.session_state.get('phone_glances', [])))
            yawn_ph.metric("Yawns", len(st.session_state.get('yawn_times', [])))

            if stop_btn:
                st.session_state['session_active'] = False
                st.session_state['session_ended'] = True
                break

            time.sleep(update_interval)

    except Exception as e:
        st.error(f"Streaming error: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass

# Summary

if st.session_state['session_ended'] and st.session_state['posture_history']:
    st.markdown("## Session Summary")
    start_t = st.session_state['start_time'] or st.session_state['posture_history'][0][0]
    p_times, p_vals = zip(*st.session_state['posture_history'])
    h_times, h_vals = zip(*st.session_state['head_tilt_history'])
    p_sec = [t - start_t for t in p_times]
    h_sec = [t - start_t for t in h_times]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_sec, y=p_vals, mode='lines', name='Posture Score'))
    fig.update_layout(title="Posture Over Time", yaxis=dict(range=[0,100]), template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h_sec, y=h_vals, mode='lines', name='Head Tilt Score'))
    fig2.update_layout(title="Head Tilt Over Time", yaxis=dict(range=[0, max(150, max(h_vals) + 10)]), template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Summary Counts")
    st.write(f"Total Yawns: {len(st.session_state.get('yawn_times', []))}")
    st.write(f"Total Phone Glances: {len(st.session_state.get('phone_glances', []))}")
