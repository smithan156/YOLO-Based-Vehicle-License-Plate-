from typing import Any


def _number(value: Any) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return float(value)


def detect_image(
    image_bytes: bytes,
    model: Any,
    confidence: float = 0.25,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run a YOLO model on uploaded image bytes and return RGB annotations and detections."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    import cv2
    import numpy as np

    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")

    results = model(image, conf=confidence, verbose=False)
    if not results:
        raise RuntimeError("The YOLO model returned no result for the uploaded image.")

    result = results[0]
    names = result.names
    detections = []
    for box in result.boxes:
        class_id = int(_number(box.cls[0]))
        coordinates = [_number(value) for value in box.xyxy[0]]
        detections.append(
            {
                "Detection type": names.get(class_id, str(class_id)),
                "Detection confidence": round(_number(box.conf[0]), 4),
                "x1": round(coordinates[0]),
                "y1": round(coordinates[1]),
                "x2": round(coordinates[2]),
                "y2": round(coordinates[3]),
            }
        )

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections
