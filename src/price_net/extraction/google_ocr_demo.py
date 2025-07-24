import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict

from google.cloud import vision


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_google_vision(image_path: str) -> Dict:
    """
    Extract text from an image using Google Cloud Vision API.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing extracted text and metadata
    """
    # Check if image file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Initialize the client
    client = vision.ImageAnnotatorClient()

    # Load the image into memory
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    # Create Image object
    image = vision.Image(content=content)

    # Perform text detection
    print(f"Processing image: {image_path}")
    response = client.text_detection(image=image)

    # Check for errors
    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    # Extract text annotations
    texts = response.text_annotations

    if not texts:
        print("No text detected in the image.")
        return {"full_text": "", "individual_words": [], "bounding_boxes": []}

    # The first annotation contains the full text
    full_text = texts[0].description

    # Extract individual words with their bounding boxes
    individual_words = []
    bounding_boxes = []

    for text in texts[1:]:  # Skip the first one as it's the full text
        # Get bounding box vertices
        vertices = []
        for vertex in text.bounding_poly.vertices:
            vertices.append({"x": vertex.x, "y": vertex.y})

        word_info = {"text": text.description, "bounding_box": vertices}

        individual_words.append(word_info)
        bounding_boxes.append(vertices)

    return {
        "full_text": full_text,
        "individual_words": individual_words,
        "bounding_boxes": bounding_boxes,
    }


def extract_text_google_vision_detailed(image_path: str) -> Dict:
    """
    Extract text with detailed information using Google Cloud Vision API.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing detailed text analysis
    """
    # Check if image file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Initialize the client
    client = vision.ImageAnnotatorClient()

    # Load the image into memory
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    # Create Image object
    image = vision.Image(content=content)

    # Perform document text detection (more detailed)
    print(f"Processing image with detailed analysis: {image_path}")
    response = client.document_text_detection(image=image)

    # Check for errors
    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    document = response.full_text_annotation

    if not document:
        print("No text detected in the image.")
        return {
            "full_text": "",
            "pages": [],
            "blocks": [],
            "paragraphs": [],
            "words": [],
        }

    # Extract detailed information
    result = {
        "full_text": document.text,
        "pages": [],
        "blocks": [],
        "paragraphs": [],
        "words": [],
    }

    # Parse pages, blocks, paragraphs, and words
    for page in document.pages:
        page_info = {
            "confidence": page.confidence,
            "width": page.width,
            "height": page.height,
        }
        result["pages"].append(page_info)

        for block in page.blocks:
            block_info = {
                "confidence": block.confidence,
                "block_type": block.block_type.name,
                "bounding_box": [
                    {"x": v.x, "y": v.y} for v in block.bounding_box.vertices
                ],
            }
            result["blocks"].append(block_info)

            for paragraph in block.paragraphs:
                paragraph_info = {
                    "confidence": paragraph.confidence,
                    "bounding_box": [
                        {"x": v.x, "y": v.y} for v in paragraph.bounding_box.vertices
                    ],
                    "words": [],
                }

                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    word_info = {
                        "text": word_text,
                        "confidence": word.confidence,
                        "bounding_box": [
                            {"x": v.x, "y": v.y} for v in word.bounding_box.vertices
                        ],
                    }
                    paragraph_info["words"].append(word_info)
                    result["words"].append(word_info)

                result["paragraphs"].append(paragraph_info)

    return result


def print_ocr_results(result: Dict, detailed: bool = False):
    """
    Print OCR results in a formatted way.

    Args:
        result: Dictionary containing OCR results
        detailed: Whether to print detailed information
    """
    print("\n" + "=" * 50)
    print("OCR RESULTS")
    print("=" * 50)

    if not result["full_text"]:
        print("No text detected in the image.")
        return

    print(f"Full Text:\n{result['full_text']}")
    print("-" * 50)

    if detailed and "words" in result:
        print(f"Detected {len(result['words'])} words:")
        for i, word in enumerate(result["words"][:10]):  # Show first 10 words
            print(f"{i + 1}. '{word['text']}' (confidence: {word['confidence']:.2f})")

        if len(result["words"]) > 10:
            print(f"... and {len(result['words']) - 10} more words")

    elif "individual_words" in result:
        print(f"Detected {len(result['individual_words'])} text elements:")
        for i, word in enumerate(result["individual_words"][:10]):  # Show first 10
            print(f"{i + 1}. '{word['text']}'")

        if len(result["individual_words"]) > 10:
            print(f"... and {len(result['individual_words']) - 10} more elements")


def main():
    """Main function to run OCR on an image."""
    parser = argparse.ArgumentParser(
        description="Extract text from image using Google Cloud Vision API"
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--detailed", action="store_true", help="Use detailed document analysis"
    )
    parser.add_argument("--output", help="Output file to save results (JSON format)")

    args = parser.parse_args()

    try:
        if args.detailed:
            result = extract_text_google_vision_detailed(args.image_path)
        else:
            result = extract_text_google_vision(args.image_path)

        # Print results
        print_ocr_results(result, detailed=args.detailed)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test with the same image
    fpath = "/Users/porterjenkins/data/price-attribution-scenes/test/price-images/0c0a4618-105f-4e6c-8047-d75c8d768489.jpg"

    try:
        print("Testing Google Cloud Vision API OCR...")
        result = extract_text_google_vision(fpath)
        print_ocr_results(result)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Google Cloud Vision API enabled")
        print("2. Service account key configured")
        print("3. GOOGLE_APPLICATION_CREDENTIALS environment variable set")
