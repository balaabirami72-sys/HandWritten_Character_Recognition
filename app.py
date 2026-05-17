import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import difflib  # Built-in sequence matching for language correction

# Set page configurations for a clean dashboard view
st.set_page_config(page_title="Handwritten Sheet Recognition", layout="centered")

st.title("📝 Intelligent Handwritten Word Recognition")
st.write("Equipped with Stroke-Merging Segments and AI Language Auto-Correction.")

# ==========================================
# 1. LOAD THE TRAINED CNN MODEL (.KERAS FORMAT)
# ==========================================
MODEL_PATH = "character_model.keras"  
class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# Vocabulary Dictionary for Auto-Correction Context Mapping
VALID_WORDS = ["Love", "LIFE", "HELLO", "WORLD", "COMPUTER", "SCIENCE", "JAVA", "PYTHON"]

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ '{MODEL_PATH}' not found! Please run your training script to save the model.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# ==========================================
# 2. CORE IMAGE PREPROCESSING FOR INDIVIDUAL CHARACTERS
# ==========================================
def preprocess_segmented_character(roi_gray):
    """Prepares an isolated grayscale character crop matching EMNIST dimensions."""
    _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    corner_pixels = [roi_bin[0, 0], roi_bin[0, -1], roi_bin[-1, 0], roi_bin[-1, -1]]
    if np.mean(corner_pixels) > 127:
        roi_bin = cv2.bitwise_not(roi_bin)
        
    roi_resized = cv2.resize(roi_bin, (28, 28), interpolation=cv2.INTER_AREA)
    
    # EMNIST Matrix Transposition
    roi_transposed = cv2.rotate(roi_resized, cv2.ROTATE_90_CLOCKWISE)
    roi_transposed = cv2.flip(roi_transposed, 1)
    
    roi_normalized = roi_transposed.astype("float32") / 255.0
    roi_final = np.expand_dims(roi_normalized, axis=0)
    roi_final = np.expand_dims(roi_final, axis=-1)
    return roi_final

# ==========================================
# 3. SHEET SEGMENTATION ENGINE WITH MORPHOLOGICAL CLOSING
# ==========================================
def process_full_sheet(image, thresh_block_size, thresh_c, morph_kernel_size):
    """Detects characters using robust structural closure filters to join fractured strokes."""
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, thresh_block_size, thresh_c
    )
    
    # 🎯 STRATEGY 1: MORPHOLOGICAL CLOSING (Dilation followed by Erosion)
    # This specifically seals internal micro-gaps and welds fragmented pen strokes back into one solid block!
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, morph_kernel)
    
    contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    frame_h, frame_w = img_gray.shape[:2]
    bounding_boxes = [
        b for b in bounding_boxes 
        if 15 < b[2] < (frame_w - 5) and 15 < b[3] < (frame_h - 5)
    ]
    
    if not bounding_boxes:
        return img_bgr, "No clean text blocks found. Tweak the sensitivity sliders!"
        
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    raw_extracted_text = ""
    
    for x, y, w, h in bounding_boxes:
        padding = 3
        y1 = max(0, y - padding)
        y2 = min(img_gray.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(img_gray.shape[1], x + w + padding)
        
        roi_gray = img_gray[y1:y2, x1:x2]
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        processed_roi = preprocess_segmented_character(roi_gray)
        preds = model.predict(processed_roi, verbose=0)
        raw_extracted_text += class_mapping[np.argmax(preds[0])]

    img_annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_annotated_rgb, raw_extracted_text

# ==========================================
# 4. USER INTERFACE ARCHITECTURE
# ==========================================
uploaded_file = st.file_uploader("Upload Image File...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    uploaded_image = Image.open(uploaded_file)
    
    if uploaded_image.width > 800:
        w_percent = (800 / float(uploaded_image.width))
        h_size = int((float(uploaded_image.height) * float(w_percent)))
        uploaded_image = uploaded_image.resize((800, h_size), Image.Resampling.LANCZOS)

    # Manual Rotation Check
    rotation_angle = st.sidebar.selectbox("🔄 Rotate Image Check:", options=[0, 90, 180, 270], index=0)
    if rotation_angle != 0:
        uploaded_image = uploaded_image.rotate(rotation_angle, expand=True)

    # Sidebar Fine-Tuning Elements
    st.sidebar.header("✂️ Fine-Tune Crop Windows")
    width, height = uploaded_image.width, uploaded_image.height
    x_range = st.sidebar.slider("Horizontal (X)", 0, width, (int(width*0.05), int(width*0.95)))
    y_range = st.sidebar.slider("Vertical (Y)", 0, height, (int(height*0.05), int(height*0.95)))
    
    st.sidebar.header("⚙️ OpenCV Stroke Welders")
    b_size = st.sidebar.slider("Block Size (Must be odd)", 3, 51, 25, step=2)
    c_val = st.sidebar.slider("Constant C", 1, 20, 7)
    # New slider to dynamically control structural merging weight
    k_size = st.sidebar.slider("Stroke Merging Intensity", 1, 10, 4)

    cropped_image = uploaded_image.crop((x_range[0], y_range[0], x_range[1], y_range[1]))
    
    annotated_img, raw_text = process_full_sheet(cropped_image, b_size, c_val, k_size)
    
    # 🎯 STRATEGY 2: THE AI LANGUAGE AUTO-CORRECTION LAYER
    # Compares the messy string against valid terms using close-match similarity ratios
    corrected_matches = difflib.get_close_matches(raw_text, VALID_WORDS, n=1, cutoff=0.2)
    final_output_word = corrected_matches[0] if corrected_matches else raw_text

    # Layout Rendering Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(cropped_image, caption="Target Crop Windows", use_container_width=True)
    with col2:
        st.image(annotated_img, caption="Segmented Output Map (Closed Outlines)", use_container_width=True)
        
    st.success("🎉 Extraction Pipeline Complete!")
    
    # Show the structural difference side-by-side
    col_text1, col_text2 = st.columns(2)
    with col_text1:
        st.markdown("**Raw Neural Network Reading:**")
        st.code(raw_text, language="text")
    with col_text2:
        st.markdown("**✨ Auto-Corrected Human-Eye Output:**")
        st.code(final_output_word, language="text")