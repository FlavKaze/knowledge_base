import os
import base64
from io import BytesIO
from typing import List, Union

import fitz #PyMuPDF
from PIL import Image, ImageFilter
import requests
from tqdm import tqdm


class ArkPDF:
    def __init__(self, doc, load_origin=None):
        self.doc = doc
        self.load_origin = load_origin
        
    def __repr__(self):
        return f"ArkPDF({self.doc})"
        
    @classmethod
    def create_pdf_from_images(cls, **kwargs) -> fitz.Document:
        """
        Create a PDF from a list of images.
        
        :param path_images List[str]: List of paths to the images.
        :param pil_images List[Image]: List of PIL images.
        :param bytes_images List[bytes]: List of image bytes.
        :param uri_images List[str]: List of URIs to the images.
        :param base64_images List[str]: List of base64 images.
        
        :return: PDF created from the images.
        """
            
        pdf = fitz.open()
        try:
            if "path_images" in kwargs:
                path_images = kwargs["path_images"]
                images = [Image.open(image_path) for image_path in path_images]
                
            elif "pil_images" in kwargs:
                images = kwargs["pil_images"]
                
            elif "bytes_images" in kwargs:
                bytes_images = kwargs["bytes_images"]
                images = [Image.open(BytesIO(image_bytes)) for image_bytes in bytes_images]
                
            elif "uri_images" in kwargs:
                uri_images = kwargs["uri_images"]
                images = [Image.open(BytesIO(requests.get(image_uri).content)) for image_uri in uri_images]
                
            elif "base64_images" in kwargs:
                base64_images = kwargs["base64_images"]
                images = [Image.open(BytesIO(base64.b64decode(image_base64))) for image_base64 in base64_images]
                
            else:
                raise ValueError("No images provided")
            
            for image in images:
                # Convert PIL image to bytes
                img_bytes = BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()
                
                # Create new PDF page
                page = pdf.new_page()
                
                # Insert image into page
                page.insert_image(page.rect, stream=img_bytes)
                
        except Exception as e:
            print(f"Error in creating PDF from images: {e}")

        return cls(pdf)
    
    @classmethod
    def open_pdf(cls, **kwargs):
        """
        Open a PDF file.
        
        :param pdf_path str: Path to the PDF file.
        :param pdf_bytes bytes: Bytes of the PDF file.
        :param pdf_uri str: URI of the PDF file.
        :param pdf_base64 str: Base64 of the PDF file.
        """
        
        if "pdf_path" in kwargs:
            doc = fitz.open(kwargs["pdf_path"])
            load_origin = "path"
        elif "pdf_bytes" in kwargs:
            doc = fitz.open(stream=kwargs["pdf_bytes"], filetype="pdf")
            load_origin = "bytes"
        elif "pdf_uri" in kwargs:
            pdf_file = requests.get(kwargs["pdf_uri"])
            doc = fitz.open(stream=pdf_file.content, filetype="pdf")
            load_origin = "uri"
        elif "pdf_base64" in kwargs:
            doc = fitz.open(stream=base64.b64decode(kwargs["pdf_base64"]), filetype="pdf")
            load_origin = "base64"
        else:
            raise ValueError("No PDF provided")
        return cls(doc, load_origin)

    def get_page(self, page_num: int):
        return self.doc[page_num]

    def show_page(self, page_num: int):
        page = self.get_page(page_num)
        image_pix_map = page.get_pixmap()
        image_bytes = image_pix_map.tobytes()
        image = Image.open(BytesIO(image_bytes))
        image.show()
    
    def get_num_pages(self):
        return self.doc.page_count

    def remove_pages(self, pages: List[int]) -> fitz.Document:
        """
        Remove pages from the PDF.
        
        :param pages List[int]: List of pages to remove.
        """
        for page in pages:
            self.doc.delete_page(page)
            
        return self.doc
    
    def close(self):
        self.doc.close()
        
    def save(self, path_to_save=None, return_bytes=False):
        return self._external_save(self.doc, path_to_save, return_bytes)
    
    @staticmethod
    def _external_save(doc: fitz.Document, path_to_save=None, return_bytes=False) -> Union[bytes, str]:
        """
        Save the PDF.
        
        :param doc fitz.Document: PDF to save.
        :param path_to_save str: Path to save the PDF.
        
        :return: PDF bytes or path to the PDF.
        """
        if return_bytes:
            pdf_bytes = doc.write()
            return pdf_bytes
        else:
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
                
            doc.save(path_to_save)
            return path_to_save
        
    @staticmethod
    def _normalize_image(func):
        """Decorator to normalize image input and output formats.
        This decorator handles different image input formats and ensures consistent output format based on the input type.
        It supports the following image formats:
        - Base64 encoded string
        - PIL Image object
        - Bytes
        - BytesIO object
        The decorator processes the image and returns it in the same format as the input.
        Args:
            func: The function to be decorated. The decorated function should accept an image parameter
                 and return a modified image.
        Returns:
            wrapper: A function that handles image format conversion before and after processing.
                    The return type matches the input type (str, Image.Image, bytes, or BytesIO).
        Example:
            @_normalize_image
            def process_image(self, image):
                # Process the image
                return modified_image
        Note:
            When the input format is string, it's assumed to be a base64 encoded image.
            The output format will preserve the original image format (defaulting to JPEG if not specified).
        """
        def wrapper(*args, **kwargs):
            args_index = None
            image = kwargs.get("image")
            if not image and len(args) > 0:
                for i, arg in enumerate(args):
                    if isinstance(arg, (Image.Image, bytes, BytesIO)):
                        args_index = i
                        break
                    elif isinstance(arg, str) and args[0].startswith("data:image"):
                        args_index = i
                        break
                if args_index is not None:
                    image = args[args_index]
            
            return_type = bytes
        
            if isinstance(image, str):
                image = base64.b64decode(image)
                original_image = Image.open(BytesIO(image))
                return_type = str
            elif isinstance(image, Image.Image):
                original_image = image
                return_type = Image.Image
            elif isinstance(image, bytes):
                original_image = Image.open(BytesIO(image))
                return_type = bytes    
            elif isinstance(image, BytesIO):
                original_image = Image.open(image)
                return_type = BytesIO             
            else:
                return func(*args, **kwargs)
            
            if kwargs.get("image"):
                kwargs["image"] = original_image
            elif args_index is not None:
                args = list(args)
                args[args_index] = original_image
                args = tuple(args)
                
            original_image_format = original_image.format or "JPEG"
            modfied_image = func(*args, **kwargs)
            if not isinstance(modfied_image, Image.Image):
                return modfied_image
            
            if return_type == Image.Image:
                return modfied_image
        
            elif return_type == bytes:
                return modfied_image.tobytes()
            
            elif return_type == BytesIO:
                buffered = BytesIO()
                modfied_image.save(buffered, format=original_image_format)
                return buffered.getvalue()
            
            elif return_type == str:
                buffered = BytesIO()
                modfied_image.save(buffered, format=original_image_format)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
        return wrapper   
        
    @staticmethod
    @_normalize_image
    def blur_image(
        image: Union[str, Image.Image, bytes, BytesIO],
        blur_intensity: int = 10,
        save_file_name: str = None
    ) -> Union[Image.Image, bytes, BytesIO]:
        """
        Blur an image.
        
        Args:
            :param image Union[Image.Image, bytes, BytesIO]: Image to blur.
            :param blur_intensity int: Intensity of the blur.
            :param save_file_name str: Name of the file to save the blurred image.
            
        Returns:
            Union[Image.Image, bytes, BytesIO]: Blurred image.
        """

        blurred_image = image.filter(ImageFilter.GaussianBlur(blur_intensity))
        if save_file_name:
            blurred_image.save(save_file_name)
        return blurred_image    
    
    @staticmethod
    @_normalize_image
    def image_add_alpha_channel(image: Union[str, Image.Image, bytes, BytesIO]) -> Union[str, Image.Image, bytes, BytesIO]:
        """
        Add the alpha channel to an image.
        
        :param image Image: Image to add the alpha channel.
        
        :return: Image with the alpha channel.
        """
        if image.mode == "RGB":
            image.putalpha(255)
        return image
    
    @staticmethod
    @_normalize_image
    def image_grayscale(image: Union[str, Image.Image, bytes, BytesIO]) -> Union[str, Image.Image, bytes, BytesIO]:
        """
        Convert an image to grayscale.
        
        :param image Image: Image to convert to grayscale.
        
        :return: Image in grayscale.
        """
        return image.convert("L")
    
    @staticmethod
    @_normalize_image
    def resize_pages(image: Union[str, Image.Image, bytes, BytesIO], **kwargs) -> Union[str, Image.Image, bytes, BytesIO]:
        """
        Resize pages from the PDF.
        
        :param image Image: Image to resize.
        :param width int: New width of the page.
        :param height int: New height of the page.
        """
        if "image_width" in kwargs or "image_height" in kwargs:
            image = image.resize((
                kwargs.get("image_width", image.width), kwargs.get("image_height", image.height)
            ))
        return image

    @staticmethod
    @_normalize_image
    def _save_image(image: Union[str, Image.Image, bytes, BytesIO], page_index: int, image_name, **kwargs):
        """
        Save an image.
        
        :param image Image: Image to save.
        :param page_index int: Index of the page.
        :param path_to_save str: Path to save the image.
        :param image_prefix str: Prefix of the image to save.
        :param image_extension str: Format of the image to save (jpg, png).
        """  
        save_format = "jpeg" if kwargs.get("image_extension") == "jpg" else kwargs.get("image_extension", "jpg")
        
        if kwargs.get("path_to_save"):
            if not os.path.exists(kwargs.get("path_to_save")):
                os.makedirs(kwargs.get("path_to_save"))
            image_name = kwargs.get("image_prefix", image_name)
            path_to_save = os.path.join(kwargs.get("path_to_save"), f"{image_name}_p{page_index}.{save_format}")
            image.save(path_to_save)
    
    def _image_to_bytes(self, image: Image.Image, save_format: str) -> bytes:
        """
        Convert an image to bytes.
        
        :param image Image: Image to convert to bytes.
        :param save_format str: Format of the image to save (jpg, png).
        
        :return: Image bytes.
        """   
        b_file = BytesIO()
        b_save_format = "jpeg" if save_format == "jpg" else save_format
        image.save(b_file, format=b_save_format)
        return b_file.getvalue()

    def pdf_to_image(self, **kwargs) -> List[Union[Image.Image, bytes]]:
        """
        Convert a PDF to an page images.
        
        :param image_dpi int: Resolution of the image (Dot per inch).
        :param image_width int: Width of the image, for resizing.
        :param image_height int: Height of the image, for resizing.
        
        :param path_to_save str: Path to save the image. If not provided, the PIL images will be returned.
        :param image_prefix str: Prefix of the image to save. Default is "image".
        :param image_extension str: Format of the image to save (jpg, png). Default is "jpg".
        
        :param start_page int: Start page to convert.
        :param end_page int: End page to convert.
        
        :param grayscale bool: If True, convert the image to grayscale.
        :param add_rgba bool: If True, add the alpha channel to the image.
        
        :param blur_image bool: If True, blur the image.
        :param blur_intensity int: Intensity of the blur. Default is 10.
        :param blur_pages_indices List[int]: List of page indices to blur.
        
        :return_bytes bool: If True, return the image bytes.
        
        :return: List of PIL images or image bytes.
        """
        all_images = []
        save_format = kwargs.get("image_extension", "jpg")
        try:
            image_name = os.path.splitext(os.path.basename(self.doc.name))[0] if self.doc.name else "image"
            start_page = kwargs.get("start_page", 0)
            end_page = kwargs.get("end_page", self.doc.page_count - 1)
            
            for page_index in tqdm(range(start_page, end_page + 1), desc="Converting pages"):
                page = self.doc[page_index]
                image_pix_map = page.get_pixmap(dpi=kwargs.get("image_dpi"))
                image_bytes = image_pix_map.tobytes()
                image = Image.open(BytesIO(image_bytes))
                image = self.image_add_alpha_channel(image) if kwargs.get("add_rgba") else image
                image = self.resize_pages(image, **kwargs) if kwargs.get("image_width") or kwargs.get("image_height") else image
                image = self.image_grayscale(image) if kwargs.get("grayscale") else image
                if kwargs.get("blur_image"):
                    if page_index in kwargs.get("blur_pages_indices", []) or not kwargs.get("blur_pages_indices"):
                        image = self.blur_image(image, blur_intensity=kwargs.get("blur_intensity"))
                    
                self._save_image(image, page_index, image_name, **kwargs)

                if kwargs.get("return_bytes"):
                    all_images.append(self._image_to_bytes(image, save_format))

                else:
                    all_images.append(image)
                
        except Exception as e:
            print(f"Error: {e}")
            raise e
                
        return all_images

    def edit_pdf(self, **kwargs) -> fitz.Document:
        """
        Receve pdf file and edit it.
        
        Args: 
            :param images_dpi int: Resolution of the images (Dot per inch).
            :param images_width int: Width of the images, for resizing.
            :param images_height int: Height of the images, for resizing.
            
            :param path_to_save str: Path to save pdf.
            
            :param start_page int: Start page to retrun.
            :param end_page int: End page to return.
            
            :param grayscale bool: If True, convert the images to grayscale.
            :param add_rgba bool: If True, add the alpha channel to the image.

            :param blur_image bool: If True, blur the image.
            :param blur_intensity int: Intensity of the blur. Default is 10.
            :param blur_pages_indices List[int]: List of page indices to blur.
         
        Returns:       
            :return: pdf BytesIO or pdf path.
        """
        paph_to_save = kwargs.pop("path_to_save", None)
        images = self.pdf_to_image(**kwargs)
        pdf = fitz.open()
        for image in tqdm(images, desc="editing pdf pages"):
            #insert pillow image into pdf
            page = pdf.new_page(width=image.width, height=image.height)
            img_bytes = BytesIO()
            # Save with maximum quality and PNG format for lossless compression
            image.save(img_bytes, format="PNG", optimize=False, quality=100)
            # Insert image with high DPI for better quality
            page.insert_image(page.rect, stream=img_bytes.getvalue())
            
        if paph_to_save:
            pdf.save(paph_to_save)
            pdf.close()
            return paph_to_save
        
        return pdf

    def split_pdf(self, **kwargs) -> List[Union[bytes, str]]:
        """
        Split a PDF file into multiple PDF files.

        Args:
            
            :param image_dpi int: Resolution of the image (Dot per inch).
            :param image_width int: Width of the image, for resizing.
            :param image_height int: Height of the image, for resizing.
            
            :param path_to_save str: Path to save the image. If not provided, the PIL images will be returned.
            :param pdf_prefix str: Prefix of the image to save. Default is "image".
            :param images_ranges (List[Tuple[int, int]]): List of tuples with the start and end page of each split.
            
            :param start_page int: Start page to convert.
            :param end_page int: End page to convert.
            
            :param grayscale bool: If True, convert the image to grayscale.
            :param add_rgba bool: If True, add the alpha channel to the image.
            
            :param blur_image bool: If True, blur the image.
            :param blur_intensity int: Intensity of the blur. Default is 10.
            :param blur_pages_indices List[int]: List of page indices to blur.
            
            :return_bytes bool: If True, return the image bytes.
            :return_fitz bool: If True, return the fitz.Document object.
            :return_arkpdf bool: If True, return the ArkPDF object.
            
        Returns:
            List[str]: List of paths to the split PDF files.
        """

        input_pdf = self.doc.name
        output_dir = kwargs.pop("path_to_save", None)
        output_dir = os.path.dirname(input_pdf) if not output_dir else output_dir

        self.doc = self.edit_pdf(**kwargs)
        
        if kwargs.get("return_fitz"):
            kwargs["return_bytes"] = True
        
        ranges = kwargs.get("images_ranges", [(0, self.doc.page_count-1)])
        if kwargs.get("pdf_prefix"):
            file_prefix = kwargs.get("pdf_prefix")
        else:
            file_prefix = os.path.splitext(os.path.basename(self.doc.name))[0] if self.doc.name else "new_pdf"

        # Split the PDF
        generated_files = []
        for i, (start, end) in tqdm(enumerate(ranges), desc="Splitting PDF"):
            # Create filename
            file_name = f"{file_prefix}_{i+1}.pdf"
            file_path = os.path.join(output_dir, file_name)

            # Create new PDF with pages from range
            new_pdf = fitz.open()
            for page in range(start, end+1):
                new_pdf.insert_pdf(self.doc, from_page=page, to_page=page)
                
            saved_pdf = self._external_save(new_pdf, path_to_save=file_path, return_bytes=kwargs.get("return_bytes"))
            
            if kwargs.get("return_fitz"):
                generated_files.append(new_pdf)
                
            elif kwargs.get("return_arkpdf"):
                generated_files.append(ArkPDF.open_pdf(pdf_path=saved_pdf))
            
            else:  
                generated_files.append(saved_pdf)

        return generated_files
    
if __name__ == '__main__':
    ark_pdf = ArkPDF.open_pdf(
        pdf_path="/home/flaviogaspareto/documents/vscode/knowledge_base/docs/324477 - AGE - 10_2016.pdf",
    )

    ark_pdf.split_pdf(
        image_dpi=500, 
        # image_width=1080, image_height=1200,
        # grayscale=True,
        path_to_save="./outputs_pdf",
        pdf_prefix="termo_de_posse",
        images_ranges=[(0, 0), (1, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    )
    
# if __name__ == '__main__':
#     # Example usage of ArkPDF class

#     # Open a PDF from a file path
#     pdf_path = "./docs/65f08f665cf6e92fbd3e0293.pdf"
#     pdf_bytes = open(pdf_path, "rb").read()
    
#     path_to_save = "./outputs"
#     # ark_pdf = ArkPDF.open_pdf(pdf_path=pdf_path)
#     ark_pdf = ArkPDF.open_pdf(pdf_bytes=pdf_bytes)


#     # Get number of pages
#     num_pages = ark_pdf.get_num_pages()
#     print(f"Number of pages: {num_pages}")

#     # Get a specific page
#     page = ark_pdf.get_page(0)
#     print(f"First page: {page}")

#     # # Convert PDF to images
#     images = ark_pdf.pdf_to_image(
#         image_dpi=300, image_extension="png", add_rgba=True,
#         path_to_save=path_to_save, start_page=0, end_page=1
#     )
#     print(f"Converted images: {images}")

#     # Edit PDF
#     edited_pdf_path = ark_pdf.edit_pdf(image_width=800, image_height=400, grayscale=True, path_to_save=f"{path_to_save}/edited_pdf.pdf")
#     print(f"Edited PDF saved at: {edited_pdf_path}")
    
#     edited_pdf_path = ark_pdf.edit_pdf(
#         image_width=1024, image_height=2048,
#         blur_image=True, blur_intensity=20, blur_pages_indices=[0],
#         path_to_save=f"{path_to_save}/edited_pdf2.pdf"
#     )
#     print(f"Edited PDF saved at: {edited_pdf_path}")

#     # Split PDF
#     split_pdfs = ark_pdf.split_pdf(
#         image_width=1080, image_height=1200, grayscale=True, 
#         images_ranges=[(0, 1), (2, 3)], path_to_save=f"{path_to_save}", pdf_prefix="split_pdf"
#     )
#     print(f"Split PDFs: {split_pdfs}")
    
#     # Split PDF2
#     split_pdfs = ark_pdf.split_pdf(
#         image_width=1080, image_height=1200, grayscale=True, 
#         images_ranges=[(0, 1), (2, 3)],
#         return_fitz=True
#     )
#     print(f"Split PDFs: {[type(pdf) for pdf in split_pdfs]}")
    
#     # Split PDF3
#     split_pdfs = ark_pdf.split_pdf(
#         image_width=1080, image_height=1200, grayscale=True, 
#         images_ranges=[(0, 1), (2, 3)],
#         return_arkpdf=True
#     )
#     print(f"Split PDFs: {[type(pdf) for pdf in split_pdfs]}")
    
#     # Create a PDF from images
#     pdf_images = [
#         Image.open(f"{path_to_save}/image_p0.png"),
#         Image.open(f"{path_to_save}/image_p1.png")
#     ]
#     ark_pdf = ArkPDF.create_pdf_from_images(pil_images=pdf_images)
#     ark_pdf.save(f"{path_to_save}/pdf_from_images.pdf")
#     print(f"PDF from images: {ark_pdf}")

#     # Create a PDF from images 2
#     ark_pdf = ArkPDF.create_pdf_from_images(path_images=[f"{path_to_save}/image_p0.png", f"{path_to_save}/image_p1.png"])
#     ark_pdf.save(f"{path_to_save}/pdf_from_images2.pdf")
#     print(f"PDF from images: {ark_pdf}")
#     ark_pdf.close()
    
    
#     #test blur_image
#     image = open(f"{path_to_save}/image_p0.png", "rb").read()
#     blurred_image = ArkPDF.blur_image(image, blur_intensity=10, save_file_name=f"{path_to_save}/blurred_image.png")
#     print(f"Blurred image: {type(blurred_image)}")

    
#     #test blur_image2
#     image = Image.open(f"{path_to_save}/image_p0.png")
#     blurred_image = ArkPDF.blur_image(image, blur_intensity=3, save_file_name=f"{path_to_save}/blurred_image2.png")
#     print(f"Blurred image: {type(blurred_image)}")
