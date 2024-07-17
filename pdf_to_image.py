import os
import base64
from io import BytesIO
from typing import List, Union

import fitz #PyMuPDF
from PIL import Image
import requests


def pdf_to_image(**kwargs) -> List[Union[Image.Image, bytes]]:
    """
    Convert a PDF to an page images.
    
    :param pdf_path str: Path to the PDF file.
    :param pdf_bytes bytes: Bytes of the PDF file.
    :param pdf_uri str: URI of the PDF file.
    :param pdf_base64 str: Base64 of the PDF file.
    
    :param image_dpi int: Resolution of the image (Dot per inch).
    :param image_width int: Width of the image, for resizing.
    :param image_height int: Height of the image, for resizing.
    
    :param path_to_save str: Path to save the image. If not provided, the PIL images will be returned.
    :param image_prefix str: Prefix of the image to save. Default is "image".
    :param image_extension str: Format of the image to save (jpg, png). Default is "jpg".
    
    :return_bytes bool: If True, return the image bytes.
    
    :return: List of PIL images or image bytes.
    """
    all_images = []
    save_format = kwargs.get("image_extension", "jpg")
    try:
        if "pdf_path" in kwargs:
            doc = fitz.open(kwargs["pdf_path"])
        elif "pdf_bytes" in kwargs:
            doc = fitz.open(stream=kwargs["pdf_bytes"], filetype="pdf")
        elif "pdf_uri" in kwargs:
            pdf_file = requests.get(kwargs["pdf_uri"])
            doc = fitz.open(stream=pdf_file.content, filetype="pdf")
        elif "pdf_base64" in kwargs:
            doc = fitz.open(stream=base64.b64decode(kwargs["pdf_base64"]), filetype="pdf")
        else:
            raise ValueError("No PDF provided")

        image_name = os.path.splitext(os.path.basename(doc.name))[0] if doc.name else "image"
        for page_index in range(doc.page_count):
            page = doc[page_index]
            image_pix_map = page.get_pixmap(dpi=kwargs.get("image_dpi"))
            image_bytes = image_pix_map.tobytes()
            image = Image.open(BytesIO(image_bytes))
            if "image_width" in kwargs or "image_height" in kwargs:
                image = image.resize((kwargs.get("image_width", image.width), kwargs.get("image_height", image.height)))
                
            if kwargs.get("path_to_save"):
                image_name = kwargs.get("image_prefix", image_name)
                path_to_save = os.path.join(kwargs.get("path_to_save"), f"{image_name}_p{page_index}.{save_format}")
                image.save(path_to_save)
                
            if kwargs.get("return_bytes"):
                b_file = BytesIO()
                b_save_format = "jpeg" if save_format == "jpg" else save_format
                image.save(b_file, format=b_save_format)
                all_images.append(b_file.getvalue())

            else:
                all_images.append(image)
                
    except Exception as e:
        print(f"Error: {e}")
        raise e
                
    return all_images


if __name__ == '__main__':
    pdf_to_image(
        pdf_uri="https://www.caceres.mt.gov.br/fotos_institucional_downloads/2.pdf",
        path_to_save="./",
        # return_bytes=True,
    )