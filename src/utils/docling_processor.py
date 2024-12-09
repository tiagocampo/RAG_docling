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
from docling.utils.export import generate_multimodal_pages

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
        
        # Configure text extraction
        pdf_pipeline_options.do_ocr = False
        
        # Configure table detection
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = True

        # Configure image extraction
        pdf_pipeline_options.images_scale = 2.0  # Higher resolution for better quality
        pdf_pipeline_options.generate_page_images = True
        pdf_pipeline_options.generate_picture_images = True

        
        # Initialize document converter with format-specific options
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.HTML,
                InputFormat.MD
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
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
                )
            }
        )
        
        # Get vision-capable model for image analysis
        self.model = get_model()
    
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
        logger.info("Document converted successfully")
        
        # Process multimodal content
        multimodal_content = []
        for content_text, content_md, content_dt, page_cells, page_segments, page in generate_multimodal_pages(conversion_result):
            if hasattr(page, 'image') and page.image:
                try:
                    # Convert page image to bytes for analysis
                    img_byte_arr = BytesIO()
                    page.image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Analyze page content with vision model
                    analysis = self.analyze_image(img_byte_arr)
                    
                    multimodal_content.append({
                        "page": page.page_no,
                        "text": content_text,
                        "markdown": content_md,
                        "cells": page_cells,
                        "segments": page_segments,
                        "image_analysis": analysis
                    })
                except Exception as e:
                    logger.warning(f"Error processing page content: {str(e)}")
        
        # Extract metadata
        metadata = {}
        try:
            # Try to get title from document properties
            if hasattr(doc, 'title'):
                metadata['title'] = doc.title
            elif hasattr(doc, 'properties') and hasattr(doc.properties, 'title'):
                metadata['title'] = doc.properties.title
            else:
                # Try to find title in the first page
                first_page = next(iter(doc.pages.values()), None)
                if first_page and hasattr(first_page, 'text'):
                    lines = first_page.text.split('\n')
                    if lines:
                        metadata['title'] = lines[0].strip()
            
            # Try to get author
            if hasattr(doc, 'author'):
                metadata['author'] = doc.author
            elif hasattr(doc, 'properties') and hasattr(doc.properties, 'author'):
                metadata['author'] = doc.properties.author
            
            # Try to get date
            if hasattr(doc, 'date'):
                metadata['date'] = doc.date
            elif hasattr(doc, 'properties') and hasattr(doc.properties, 'created'):
                metadata['date'] = doc.properties.created
            
            logger.info(f"Extracted metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            metadata = {}
        
        # Export document structure
        doc_dict = doc.export_to_dict()
        doc_dict["metadata"] = metadata
        doc_dict["multimodal_content"] = multimodal_content
        
        logger.info(f"Document processing complete with {len(multimodal_content)} pages of content")
        return doc_dict
    
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
    

    def extract_text_with_context(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text with context from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries containing text chunks with metadata
        """
        logger.info(f"Extracting text with context from: {file_path}")
        
        # Convert file path to Path object
        path = Path(file_path)
        
        # Convert and process the document
        conversion_result = self.converter.convert(path)
        
        if conversion_result.status != ConversionStatus.SUCCESS:
            logger.error(f"Document conversion failed: {conversion_result.error_message}")
            raise Exception(f"Document conversion failed: {conversion_result.error_message}")
        
        chunks = []
        # Process multimodal content
        for content_text, content_md, content_dt, page_cells, page_segments, page in generate_multimodal_pages(conversion_result):
            # Extract metadata
            metadata = {
                "page_number": page.page_no,
                "markdown": content_md,
                "cells": page_cells,
                "segments": page_segments
            }
            
            # Add image analysis if available
            if hasattr(page, 'image') and page.image:
                try:
                    # Convert page image to bytes for analysis
                    img_byte_arr = BytesIO()
                    page.image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Analyze page content with vision model
                    image_analysis = self.analyze_image(img_byte_arr)
                    metadata["image_analysis"] = image_analysis
                except Exception as e:
                    logger.warning(f"Error analyzing page image: {str(e)}")
            
            # Create chunk with text and metadata
            chunk = {
                "text": content_text,
                "metadata": metadata
            }
            chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} text chunks with context")
        return chunks
