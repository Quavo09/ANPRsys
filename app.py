# app.py
import time, os, re, tempfile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pytesseract
import csv
import os
import pandas as pd

from datetime import datetime
# CSV file path
CSV_FILE = "recognized_plates.csv"
# ========= BASIC CONFIG (put this FIRST) =========
st.set_page_config(page_title="Cameroon ANPR", layout="wide")

# --- Paths ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CSV_FILE = "vehicle_database.csv"  # keep next to app.py

# --- Tesseract path (show a friendly warning if missing) ---
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except Exception:
    pass
# Function to log recognized plates
def log_plate(plate_text):
    plate_text = plate_text.strip().upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Plate Number", "Timestamp"])

def log_plate(plate_number, log_file="recognized_plates.csv"):
    """Logs new plates into a CSV file and gives feedback."""
    
    # If file doesn't exist, create it with headers
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=["Plate Number"])
        df.to_csv(log_file, index=False)

    # Read the CSV
    df = pd.read_csv(log_file)

    # Check if plate already exists
    if plate_number in df["Plate Number"].values:
        print(f"ℹ️ Plate {plate_number} already in database.")
        return False  # Not new

    # Append new plate using concat (pandas 2.0+)
    new_row = pd.DataFrame({"Plate Number": [plate_number]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(log_file, index=False)
    
    print(f"✅ Plate {plate_number} registered.")
    return True  # New plate

# Example usage after recognition
# Replace "recognized_text" with your OCR result
recognized_text = "CE789OH"
log_plate(recognized_text)
# ===== CAMEROON PLATE REGEX PATTERNS =====
plate_patterns = {
    "Regular": r"^[A-Z]{2}\d{3,4}[A-Z]{1,2}$",
    "Custom Number": r"^\d[A-Z0-9]{5}[A-Z]{2}\d$",
    "Cargo": r"^[A-Z]{2}(?:\d{4}[A-Z]{2}|[A-Z]{2}\d{4}[A-Z])$",
    "State Transport": r"^[A-Z]{2}\d{4}[A-Z]$",
    "Postal": r"^RT\d{6}$",
    "Military": r"^\d{7}$",
    "Police": r"^SN\d{4}$",
    "Diplomatic": r"^\d{1,3}[A-Z]{2}\d{1,3}$"
}

# Region codes for Regular/Custom/Cargo/State plates
region_codes = {
    "AD": "Adamawa",
    "CE": "Center",
    "EN": "Far North",
    "ES": "East",
    "LT": "Littoral",
    "NO": "North",
    "NW": "Northwest",
    "OU": "West",
    "SU": "South",
    "SW": "Southwest"
}
def classify_cameroon_plate(plate):
    for plate_type, pattern in plate_patterns.items():
        if re.fullmatch(pattern, plate):
            return plate_type
    return None

def get_region_from_plate(plate):
    if len(plate) >= 2 and plate[:2] in region_codes:
        return region_codes[plate[:2]]
    return "Unknown"   # <-- ensures it never stays blank

def get_region_from_plate(plate):
    if len(plate) >= 2 and plate[:2] in region_codes:
        return region_codes[plate[:2]]
    return None

# --- Cameroon region codes ---
REGION_CODES = ["AD", "CE", "EN", "ES", "LT", "NO", "NW", "OU", "SU", "SW"]

# ========= DEV UTILITIES =========

# 1) Force CSS to re-apply on every rerun (nonce changes each reload)
def apply_custom_styles():
    if "css_nonce" not in st.session_state:
        st.session_state.css_nonce = 0
    st.session_state.css_nonce += 1
    nonce = st.session_state.css_nonce
st.markdown("""
    <style>
       /* Full background with left-to-right green → red → yellow gradient */
        .stApp {
    background: linear-gradient(to right,
        #006B3F 0%,   #006B3F 33.33%,   /* green */
        #FF0000 33.33%, #FF0000 66.66%, /* red */
        #FFD700 66.66%, #FFD700 100%    /* yellow */
    );
    background-attachment: fixed;
    color: white !important;
    position: relative;
    overflow: hidden;
}

/* Yellow star behind content */
.stApp::before {
    content: "★";
    position: absolute;   /* relative to .stApp */
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 8rem;
    color: #FFD700;
    z-index: 0;           /* behind content */
    opacity: 0.9;         /* slightly transparent */
}

/* Make sure app content stays above the star */
.stApp > * {
    position: relative;
    z-index: 1;
}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
        }

        /* Sidebar images styling */
        section[data-testid="stSidebar"] img {
            border: 2px solid #FFD700;
            border-radius: 6px;
            filter: brightness(1.2) sepia(1) hue-rotate(20deg); /* yellow tint */
        }

        /* Sidebar text bold + yellow + font size */
        section[data-testid="stSidebar"] * {
            color: #FFD700 !important;
            font-weight: bold;
            font-size: 14px !important;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #FFD700 !important;
            text-shadow: 1px 1px 2px black;
        }

        /* Image borders */
        img {
            border-radius: 10px;
            border: 3px solid #FFD700;
        }

        /* Dataframe container */
        div[data-testid="stDataFrame"] {
            background-color: white !important;
            color: black !important;
            border-radius: 8px;
            padding: 10px;
        }

        /* Buttons */
        .stButton>button {
            background-color: #006B3F; /* green */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FFD700; /* gold */
            color: black;
            transform: scale(1.05);
        }

        /* File uploader (browse button) */
        [data-testid="stFileUploader"] section div {
            background-color: #FFD700 !important;
            color: black !important;
            border-radius: 8px;
            padding: 8px;
            font-weight: bold;
            font-size: 14px;
        }
        [data-testid="stFileUploader"] button {
            background-color: #FFD700 !important;
            color: black !important;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)


# 2) Robust CSV loader that hot-reloads when the file changes
@st.cache_data(show_spinner=False)
def load_vehicle_data(file_path: str, file_mtime: float):
    # file_mtime is part of the cache key; changing CSV invalidates the cache automatically
    return pd.read_csv(file_path)

def get_csv():
    try:
        mtime = os.path.getmtime(CSV_FILE)
    except FileNotFoundError:
        st.warning(f"⚠️ CSV not found: `{CSV_FILE}`. Create it with a `plate_number` column.")
        return pd.DataFrame(columns=["plate_number"])
    return load_vehicle_data(CSV_FILE, mtime)

# 3) One-click reload button (clears cache + reruns)
with st.sidebar:
    if st.button("🔄 Reload app / clear cache"):
        st.cache_data.clear()
        st.rerun()

# Apply styles (after sidebar content so colors apply there too)
apply_custom_styles()

# ========= OCR / PLATE PROCESSING =========
def process_plate_image(image_bgr: np.ndarray):
    """Find orange plate region and OCR it. Returns (crop, text or None)."""
    if image_bgr is None:
        return None, None

    # Detect orange plate by HSV range
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_orange = (5, 100, 100)
    upper_orange = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    plate_crop = image_bgr[y:y+h, x:x+w]

    # Preprocess for OCR (and “repair” stencil breaks)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Otsu bin + closing to connect breaks
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    raw = pytesseract.image_to_string(bin_img, config=config).strip()
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
    log_plate(cleaned)

    # Validate region code if present
    if len(cleaned) >= 2 and cleaned[:2] not in REGION_CODES:
        return plate_crop, None

    return plate_crop, cleaned or None
def get_region_from_plate(plate):
    if len(plate) >= 2 and plate[:2] in region_codes:
        return region_codes[plate[:2]]
    return "Unknown"   # <-- ensures it never stays blank
# ========= UI =========

st.title("ANPR Cameroon Number Plate Recognition System")

vehicle_data = get_csv()

st.sidebar.header("Upload Options")
upload_type = st.sidebar.radio("Choose Upload Type", ["Image(s)", "Video", "Camera"], horizontal=False)

# ---- IMAGES ----
def get_region_from_plate(plate):
    if len(plate) >= 2 and plate[:2] in region_codes:
        return region_codes[plate[:2]]
    return "Unknown"   # <-- ensures it never stays blank
if upload_type == "Image(s)":
    files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">Upload one or more images</span>', unsafe_allow_html=True) 

    if files:
        for f in files:
            bytes_np = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(bytes_np, cv2.IMREAD_COLOR)

            st.image(img, channels="BGR", caption=f"Uploaded: {f.name}", width=400)
            crop, plate = process_plate_image(img)

            if crop is not None:
                st.image(crop, channels="BGR", caption="Detected Plate", width=250)
                if plate:
                    st.write(f"**Detected Plate:** {plate}")
                    match = vehicle_data[vehicle_data["plate_number"].str.upper() == plate.upper()]
                    if not match.empty:
                        st.success("✅ Match found in database")
                        st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">✅ Match found in database</span>', unsafe_allow_html=True) 
                        st.dataframe(match)
                    else:
                        st.error("❌ No match found in database")
                        st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">❌ No match found in database</span>', unsafe_allow_html=True) 

                    # --- Console printout for debugging/trace ---
                    plate_type = classify_cameroon_plate(plate)
                    plate_region = get_region_from_plate(plate) if plate_type else None
                    print(f"\n📷 Uploaded: {f.name}")
                    print(f"🛠 Cleaned plate: '{plate}'")
                    if plate_type:
                        print(f"🇨🇲 Recognized plate type: {plate_type}")
                        st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">🇨🇲 Recognized plate type:</span>', unsafe_allow_html=True) 

                        if plate_region:
                            print(f"📍 Detected region: {plate_region}")
                            st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">📍 Detected region:</span>', unsafe_allow_html=True) 

                        if not match.empty:
                            print("✅ Match found in database:\n", match.to_string(index=False))
                            st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">✅ Match found in database.</span>', unsafe_allow_html=True) 
                        else:
                            print("❌ No match found in database.")
                            st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">❌ No match found in database.</span>', unsafe_allow_html=True) 

                    else:
                        print("⚠️ Plate format not recognized as valid Cameroon plate")
                        st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">⚠️ Plate format not recognized as valid Cameroon plate</span>', unsafe_allow_html=True) 

                        print("⏭ Skipping DB match due to invalid plate format")
                        st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">⏭ Skipping DB match due to invalid plate format.</span>', unsafe_allow_html=True) 
                       
                    print("-" * 50)
                else:
                    st.warning("⚠️ Plate detected but region code is invalid or OCR failed.")
                    st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">⚠️ Plate detected but region code is invalid or OCR failed.</span>', unsafe_allow_html=True) 
                 
            else:
                st.warning("⚠️ No plate detected in this image.")
                st.markdown('<span style="color:Black; font-size:16px font-weight:bold;">⚠️ No plate detected in this image.</span>', unsafe_allow_html=True) 
                
# ---- VIDEO FILE ----
elif upload_type == "Video":
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if vid:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(vid.read()); tmp.flush()

        cap = cv2.VideoCapture(tmp.name)

        # Playback timing for smooth video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        delay = max(1.0 / fps, 0.01)

        frame_count = 0
        ocr_stride = 10               # OCR every Nth frame for speed
        blink = False                 # Blink highlight on/off
        last_text = None              # Last good OCR text
        detected_plates = []          # (frame_idx, text)

        # Load vehicle database
        vehicle_data = get_csv()   # ensure it has a "plate_number" column

        # Layout: small video (left), results (right)
        vid_col, info_col = st.columns([1, 2])

        with vid_col:
            stframe = st.empty()   # live video

        with info_col:
            result_placeholder = st.empty()     # live results
            crop_placeholder = st.empty()       # show plate crop
            summary_placeholder = st.container()

        import time

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            blink = not blink  # toggle

            # --- Detect orange plate region (for rectangle + continuity) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_orange = (5, 100, 100)
            upper_orange = (20, 255, 255)
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bbox = None
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = (x, y, w, h)

                if blink:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if last_text:
                    cv2.putText(frame, last_text, (x, y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # --- OCR every Nth frame ---
            if frame_count % ocr_stride == 0:
                plate_crop, detected_plate = process_plate_image(frame)

                if plate_crop is not None and detected_plate:
                    last_text = detected_plate
                    detected_plates.append((frame_count, detected_plate))

                    # Lookup in vehicle database
                    match = vehicle_data[vehicle_data["plate_number"].str.upper() == detected_plate.upper()]

                    # Live update results
                    with info_col:
                        if not match.empty:
                            result_placeholder.markdown(
                                f"### ✅ Plate Detected: **{detected_plate}** _(Frame {frame_count})_"
                            )
                            st.success("✅ Match found in database")
                            st.dataframe(match, use_container_width=True)
                        else:
                            result_placeholder.markdown(
                                f"### ⚠️ Plate Detected: **{detected_plate}** _(Frame {frame_count})_"
                            )
                            st.error("❌ No match found in database")

                        crop_placeholder.image(plate_crop, channels="BGR", use_container_width=True)

            # --- Display video (resized smaller) ---
            target_w = 640
            h, w = frame.shape[:2]
            disp_h = int(h * (target_w / w))
            display_frame = cv2.resize(frame, (target_w, disp_h))

            with vid_col:
                stframe.image(display_frame, channels="BGR", use_container_width=True)

            time.sleep(delay)

        cap.release()
        cv2.destroyAllWindows()
        tmp.close()
        try:
            os.remove(tmp.name)
        except PermissionError:
            print("⚠️ Could not delete temp file immediately.")

        # Final summary list
        with info_col:
            if detected_plates:
                st.subheader("Detected Plates (Summary)")
                for fc, plate in detected_plates:
                    st.write(f"• Frame {fc}: **{plate}**")
            else:
                st.warning("No plates detected in the video.")


# ---- CAMERA STREAM ----
elif upload_type == "Camera":
    st.write("🎥 Live Camera Feed")

    # Load vehicle database
    vehicle_data = get_csv()  # must contain "plate_number"

    # Layout
    cam_col, info_col = st.columns([1, 2])
    with cam_col:
        stframe = st.empty()
    with info_col:
        result_placeholder = st.empty()
        crop_placeholder = st.empty()
        summary_placeholder = st.container()

    cap = cv2.VideoCapture(0)   # 0 = default webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    delay = max(1.0 / fps, 0.01)

    frame_count = 0
    ocr_stride = 10
    blink = False
    last_text = None
    detected_plates = []

    import time
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not access webcam.")
            break

        frame_count += 1
        blink = not blink

        # --- Detect orange plate region (optional: tweak color range) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = (5, 100, 100)
        upper_orange = (20, 255, 255)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)

            if blink:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if last_text:
                cv2.putText(frame, last_text, (x, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- OCR every Nth frame ---
        if frame_count % ocr_stride == 0:
            plate_crop, detected_plate = process_plate_image(frame)

            if plate_crop is not None and detected_plate:
                last_text = detected_plate
                detected_plates.append((frame_count, detected_plate))

                # Lookup
                match = vehicle_data[vehicle_data["plate_number"].str.upper() == detected_plate.upper()]

                with info_col:
                    if not match.empty:
                        result_placeholder.markdown(
                            f"### ✅ Plate Detected: **{detected_plate}** _(Frame {frame_count})_"
                        )
                        st.success("✅ Match found in database")
                        st.dataframe(match, use_container_width=True)
                    else:
                        result_placeholder.markdown(
                            f"### ⚠️ Plate Detected: **{detected_plate}** _(Frame {frame_count})_"
                        )
                        st.error("❌ No match found in database")

                    crop_placeholder.image(plate_crop, channels="BGR", use_container_width=True)

        # --- Display resized live frame ---
        target_w = 640
        h, w = frame.shape[:2]
        disp_h = int(h * (target_w / w))
        display_frame = cv2.resize(frame, (target_w, disp_h))

        with cam_col:
            stframe.image(display_frame, channels="BGR", use_container_width=True)

        time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    with info_col:
        if detected_plates:
            st.subheader("Detected Plates (Summary)")
            for fc, plate in detected_plates:
                st.write(f"• Frame {fc}: **{plate}**")
        else:
            st.warning("No plates detected from live camera.")

