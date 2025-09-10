import cv2
import pytesseract
import pandas as pd
import os
import re
import numpy as np

# ===== CONFIG =====
CSV_FILE = "vehicle_database.csv"   # Path to your CSV
PLATES_FOLDER = r"C:\Users\Dell\Desktop\ANPR_Project\plates"  # Path to plates folder
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Path to tesseract.exe

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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
    return None

def find_orange_plate(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = (5, 100, 100)
    upper_orange = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Morph to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6 and w > 100 and h > 30:
            plate_crop = image[y:y+h, x:x+w]
            plates.append(plate_crop)
    return plates

def preprocess_plate(plate_img):
    """Enhance the plate so black characters are vivid for OCR"""
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    lower_orange = (5, 100, 100)
    upper_orange = (20, 255, 255)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    plate_only = cv2.bitwise_and(plate_img, plate_img, mask=mask)

    gray = cv2.cvtColor(plate_only, cv2.COLOR_BGR2GRAY)

    # Isolate dark letters
    _, char_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Morph to connect strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)

    # Upscale for clarity
    char_mask = cv2.resize(char_mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    return char_mask

# ===== LOAD DATABASE =====
try:
    vehicle_data = pd.read_csv(CSV_FILE, encoding="utf-8")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

vehicle_data.columns = [col.strip().lower() for col in vehicle_data.columns]
if "plate_number" not in vehicle_data.columns:
    print("❌ 'plate_number' column not found in CSV. Please check your CSV headers.")
    exit()

# ===== PROCESS PLATES =====
image_files = [f for f in os.listdir(PLATES_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"📂 Found {len(image_files)} plate image(s) in '{PLATES_FOLDER}': {image_files}")

for filename in image_files:
    file_path = os.path.join(PLATES_FOLDER, filename)
    img = cv2.imread(file_path)

    if img is None:
        print(f"⚠️ Could not read image: {filename}")
        continue

    plate_images = find_orange_plate(img)
    if not plate_images:
        print(f"\n📷 Image: {filename}")
        print("⚠️ No orange plate found, skipping OCR...")
        print("-" * 50)
        continue

    for idx, plate_img in enumerate(plate_images):
        processed = preprocess_plate(plate_img)
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        detected_text = pytesseract.image_to_string(processed, config=config).strip()
        detected_plate = re.sub(r"[^A-Z0-9]", "", detected_text.upper())

        plate_type = classify_cameroon_plate(detected_plate)
        plate_region = get_region_from_plate(detected_plate) if plate_type else None

        print(f"\n📷 Image: {filename} (Plate Region #{idx + 1})")
        print(f"🔍 OCR detected (raw): '{detected_text}'")
        print(f"🛠 Cleaned plate: '{detected_plate}'")

        if plate_type:
            print(f"🇨🇲 Recognized plate type: {plate_type}")
            if plate_region:
                print(f"📍 Detected region: {plate_region}")
            match = vehicle_data[vehicle_data["plate_number"].str.upper() == detected_plate.upper()]
            if not match.empty:
                print("✅ Match found in database:\n", match.to_string(index=False))
            else:
                print("❌ No match found in database.")
        else:
            print("⚠️ Plate format not recognized as valid Cameroon plate")
            print("⏭ Skipping DB match due to invalid plate format")

        print("-" * 50)

print(f"\n📊 Finished processing {len(image_files)} image(s).")