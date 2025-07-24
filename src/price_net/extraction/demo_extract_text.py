from pathlib import Path

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def preprocess_and_save_image(image_path: str, output_path: str = None) -> str:
    """
    Preprocess image to improve OCR results and save to new file.

    Args:
        image_path: Path to the input image
        output_path: Path to save preprocessed image (optional)

    Returns:
        Path to the preprocessed image
    """
    # Read the image
    image = cv2.imread(image_path)

    # Get original dimensions
    height, width = image.shape[:2]
    print(f"Original image size: {width}x{height}")

    # Upscale the image (4x larger for better OCR)
    scale_factor = 4
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Use INTER_CUBIC for better quality when upscaling
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(f"Upscaled image size: {new_width}x{new_height}")

    # Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Apply adaptive thresholding to improve text clarity
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Save preprocessed image
    if output_path is None:
        output_path = image_path.replace(".jpg", "_enhanced.jpg")

    cv2.imwrite(output_path, cleaned)
    print(f"Enhanced image saved to: {output_path}")

    return output_path


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Doctr OCR.

    Args:
        image_path: Path to the image file

    Returns:
        Extracted text as a string
    """
    # Check if image file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load a pre-trained OCR model
    print("Loading OCR model...")
    ocr_model = ocr_predictor(pretrained=True)
    # ocr_model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    # Preprocess and enhance the image
    enhanced_image_path = preprocess_and_save_image(image_path)

    # Load the enhanced image
    print(f"Processing enhanced image: {enhanced_image_path}")
    doc = DocumentFile.from_images(enhanced_image_path)

    # Perform OCR
    print("Extracting text...")
    result = ocr_model(doc)
    print(result)

    # Extract text from the result
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ""
                for word in line.words:
                    line_text += word.value + " "
                extracted_text.append(line_text.strip())

    return "\n".join(extracted_text)


if __name__ == "__main__":
    fpath = "/Users/porterjenkins/data/price-attribution-scenes/test/price-images/0c0a4618-105f-4e6c-8047-d75c8d768489.jpg"
    text = extract_text_from_image(fpath)
    print(text)
