from fastapi import UploadFile
from PIL import Image
import pytesseract
import io
import logging


logger = logging.getLogger(__name__)


async def extract_text_from_image(file: UploadFile) -> str:
    """
    Extract text from an uploaded image file using OCR (Optical Character Recognition)
    
    Args:
        file: The uploaded image file
        
    Returns:
        str: The extracted text from the image
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Use pytesseract to extract text
        ocr_text = pytesseract.image_to_string(image)

        if not ocr_text.strip():
            logger.warning("OCR produced empty text result")
            return "No text could be extracted from the image."
        
        logger.info(f"successfully extracted {len(ocr_text)} characters from the image")
        return ocr_text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise
