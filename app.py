import csv
from io import BytesIO, StringIO
from pathlib import Path

import streamlit as st
from PIL import Image

from src.detector import detect_image


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


st.set_page_config(page_title="SmartPlate Marine & Road", page_icon="🚘", layout="wide")
st.title("SmartPlate Marine & Road")
st.caption("Stage 1: image-based object detection with Ultralytics YOLO")

st.info(
    "The default COCO model detects general objects, not license plates or ship markings. "
    "Use a compatible, locally trained YOLO weights file for those classes. "
    "A detected text region is not proof of a vehicle or vessel identity."
)

with st.sidebar:
    st.header("Detection settings")
    weights = st.text_input(
        "YOLO weights",
        value="yolo11n.pt",
        help="Use the default general-object model or enter a local .pt weights path.",
    ).strip()
    confidence = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.25)
    st.caption("The default weights may be downloaded by Ultralytics on first use.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload an image containing objects you want the selected YOLO model to detect.",
)

if uploaded_file is not None:
    if not weights:
        st.error("Enter a YOLO weights name or local path before running detection.")
    else:
        image_bytes = uploaded_file.getvalue()
        try:
            model = load_model(weights)
            with st.spinner("Running detection..."):
                annotated_rgb, detections = detect_image(image_bytes, model, confidence)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Detection failed: {error}")
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Original image")
                st.image(image_bytes, use_container_width=True)
            with right:
                st.subheader("Annotated image")
                st.image(annotated_rgb, channels="RGB", use_container_width=True)

            image_buffer = BytesIO()
            Image.fromarray(annotated_rgb).save(image_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name=f"{Path(uploaded_file.name).stem}_detected.png",
                mime="image/png",
            )

            if detections:
                st.subheader("Detections")
                st.dataframe(detections, use_container_width=True, hide_index=True)

                csv_buffer = StringIO()
                writer = csv.DictWriter(csv_buffer, fieldnames=list(detections[0]))
                writer.writeheader()
                writer.writerows(detections)
                st.download_button(
                    "Download detections as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{Path(uploaded_file.name).stem}_detections.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No objects met the selected confidence threshold.")
