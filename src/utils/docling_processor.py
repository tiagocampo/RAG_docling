from typing import Dict, Any, List
from pathlib import Path
import logging
import json
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption,
    HTMLFormatOption,
    ImageFormatOption
)
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from graphs.chat_graph import get_model
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoclingProcessor:
    def __init__(self):
        """Initialize Docling processor with proper format configurations."""
        # Configure pipeline options for PDF
        pdf_pipeline_options = PdfPipelineOptions()
        # Disable OCR completely
        pdf_pipeline_options.do_ocr = False
        # Enable table structure without cell matching for better performance
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = False
        # Enable image extraction
        pdf_pipeline_options.images_scale = 2.0
        pdf_pipeline_options.generate_page_images = True
        pdf_pipeline_options.generate_picture_images = True
        
        # Initialize document converter with format-specific options
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.MD
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pdf_pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.HTML: HTMLFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=StandardPdfPipeline
                )
            }
        )
        
        # Get vision-capable model for image analysis
        self.model = get_model()
    
    def analyze_image(self, image_data: bytes) -> str:
        """Analyze image using LLM with vision capabilities."""
        try:
            # Convert image to base64 for LLM processing
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create prompt for image analysis
            prompt = f"""Analyze this image and provide a detailed description of its contents.
            Focus on:
            1. Main elements and their relationships
            2. Any text visible in the image
            3. Charts, diagrams, or visual data
            4. Key information that would be relevant for document understanding
            
            Provide the description in a clear, concise format."""
            
            # Use model's vision capabilities
            response = self.model.invoke([{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            }, {
                "type": "text",
                "text": prompt
            }])
            
            return response.content
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return "Error analyzing image content"
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document using Docling and return structured information."""
        logger.info(f"Processing document: {file_path}")
        
        # Convert file path to Path object
        path = Path(file_path)
        
        # Convert and process the document
        conversion_result = self.converter.convert(path)
        
        if conversion_result.status != ConversionStatus.SUCCESS:
            logger.error(f"Document conversion failed: {conversion_result.error_message}")
            raise Exception(f"Document conversion failed: {conversion_result.error_message}")
        
        doc = conversion_result.document
        
        # Process images and tables
        image_analyses = []
        table_data = []
        
        # Process page images
        for page in doc.pages.values():
            if hasattr(page, 'image') and page.image and page.image.pil_image:
                try:
                    # Convert PIL image to bytes
                    img_byte_arr = BytesIO()
                    page.image.pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Analyze page image
                    analysis = self.analyze_image(img_byte_arr)
                    image_analyses.append({
                        "analysis": analysis,
                        "page": page.page_no,
                        "type": "page_image"
                    })
                except Exception as e:
                    logger.warning(f"Error processing page image: {str(e)}")
        
        # Process document elements (figures, tables, etc.)
        for element, _ in doc.iterate_items():
            try:
                # Get the page number for the element
                page_no = element.page_no if hasattr(element, 'page_no') else None
                
                # Handle tables
                if hasattr(element, 'cells') and element.cells:
                    table_dict = {
                        "page": page_no,
                        "location": element.bbox.to_dict() if hasattr(element.bbox, 'to_dict') else None,
                        "rows": len(element.cells),
                        "cols": len(element.cells[0]) if element.cells else 0,
                        "data": []
                    }
                    
                    # Extract table data
                    for row in element.cells:
                        row_data = []
                        for cell in row:
                            cell_text = cell.text if hasattr(cell, 'text') else ""
                            row_data.append(cell_text)
                        table_dict["data"].append(row_data)
                    
                    table_data.append(table_dict)
                
                # Handle figures and other images
                if hasattr(element, 'get_image'):
                    try:
                        image = element.get_image(doc)
                        if image and image.pil_image:
                            # Convert PIL image to bytes
                            img_byte_arr = BytesIO()
                            image.pil_image.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            analysis = self.analyze_image(img_byte_arr)
                            image_analyses.append({
                                "analysis": analysis,
                                "page": page_no,
                                "type": element.__class__.__name__,
                                "location": element.bbox.to_dict() if hasattr(element.bbox, 'to_dict') else None
                            })
                    except Exception as e:
                        logger.warning(f"Error processing element image: {str(e)}")
            
            except Exception as e:
                logger.warning(f"Error processing element: {str(e)}")
        
        # Export document structure
        doc_dict = doc.export_to_dict()
        doc_dict["images"] = image_analyses
        doc_dict["tables"] = table_data
        
        return doc_dict
    
    def extract_text_with_context(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text with context information."""
        logger.info(f"Extracting text with context from: {file_path}")
        
        # Process the document
        doc_info = self.process_document(file_path)
        
        # Extract text chunks with context
        chunks = []
        for page in doc_info["pages"]:
            page_no = page.get("page_no") if isinstance(page, dict) else getattr(page, "page_no", None)
            
            # Get sections for this page
            page_sections = [
                section for section in doc_info.get("sections", [])
                if section.get("page_number") == page_no
            ]
            
            # Get tables for this page
            page_tables = [
                table for table in doc_info.get("tables", [])
                if table.get("page") == page_no
            ]
            
            # Get images for this page
            page_images = [
                img for img in doc_info.get("images", [])
                if img.get("page") == page_no
            ]
            
            # Create chunk with context
            chunk = {
                "text": page.get("text") if isinstance(page, dict) else getattr(page, "text", ""),
                "context": {
                    "page_num": page_no,
                    "header": next((section.get("title") for section in page_sections if section.get("is_header")), None),
                    "section_type": next((section.get("type") for section in page_sections), "body"),
                    "tables": [{
                        "data": table.get("data", []),
                        "rows": table.get("rows", 0),
                        "cols": table.get("cols", 0),
                        "location": table.get("location")
                    } for table in page_tables],
                    "images": [{
                        "type": img.get("type"),
                        "analysis": img.get("analysis"),
                        "location": img.get("location")
                    } for img in page_images]
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def get_document_structure(self, file_path: str) -> Dict[str, Any]:
        """Get the high-level structure of the document."""
        doc_info = self.process_document(file_path)
        
        return {
            "title": doc_info["metadata"].get("title"),
            "author": doc_info["metadata"].get("author"),
            "date": doc_info["metadata"].get("date"),
            "sections": [
                {
                    "title": section.get("header"),
                    "level": section.get("level"),
                    "page": section.get("page_number")
                }
                for section in doc_info["layout"]["sections"]
                if section.get("header")
            ],
            "num_pages": len(doc_info["layout"]["pages"]),
            "num_tables": len(doc_info["tables"]),
            "num_charts": len(doc_info["charts"]),
            "num_entities": len(doc_info["entities"])
        }
    
    def _is_entity_in_section(self, entity: Dict, section: Dict) -> bool:
        """Check if an entity is within a section's bounds."""
        entity_start = entity.get("start", 0)
        entity_end = entity.get("end", 0)
        section_start = section.get("start", 0)
        section_end = section.get("end", 0)
        
        return (entity_start >= section_start and 
                entity_end <= section_end)
    
    def _is_element_in_section(self, element: Dict, section: Dict) -> bool:
        """Check if a table or chart is within a section's bounds."""
        element_page = element.get("page_number", 0)
        section_page = section.get("page_number", 0)
        
        return element_page == section_page