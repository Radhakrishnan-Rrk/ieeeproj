"""
PDF Parser Module
Uses PyMuPDF (fitz) for high-quality PDF text and image extraction
"""

import fitz  # PyMuPDF
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """Represents a block of text with formatting info"""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    bbox: tuple  # (x0, y0, x1, y1)
    page_num: int


@dataclass
class ImageInfo:
    """Represents an extracted image"""
    image_bytes: bytes
    width: int
    height: int
    page_num: int
    bbox: tuple
    caption: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class TableInfo:
    """Represents an extracted table"""
    rows: List[List[str]]
    page_num: int
    bbox: tuple
    caption: Optional[str] = None


@dataclass
class ParsedDocument:
    """Complete parsed document structure"""
    text_blocks: List[TextBlock] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)
    tables: List[TableInfo] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFParser:
    """
    High-quality PDF parser using PyMuPDF
    Extracts text with formatting, images, and attempts table detection
    """
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract all content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content
        """
        doc = fitz.open(pdf_path)
        result = ParsedDocument(
            page_count=len(doc),
            metadata=doc.metadata
        )
        
        full_text_parts = []
        
        for page_num, page in enumerate(doc):
            # Extract text blocks with formatting
            text_blocks = self._extract_text_blocks(page, page_num)
            result.text_blocks.extend(text_blocks)
            
            # Extract images
            images = self._extract_images(page, page_num, pdf_path)
            result.images.extend(images)
            
            # Attempt table extraction
            tables = self._extract_tables(page, page_num)
            result.tables.extend(tables)
            
            # Get full page text
            full_text_parts.append(page.get_text())
        
        result.full_text = "\n".join(full_text_parts)
        doc.close()
        
        return result
    
    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text blocks with formatting information"""
        blocks = []
        
        # Get detailed text with formatting info
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Not a text block
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    font_name = span.get("font", "")
                    font_size = span.get("size", 12)
                    flags = span.get("flags", 0)
                    
                    # Determine bold/italic from flags
                    is_bold = bool(flags & 2 ** 4) or "bold" in font_name.lower()
                    is_italic = bool(flags & 2 ** 1) or "italic" in font_name.lower()
                    
                    blocks.append(TextBlock(
                        text=text,
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        page_num=page_num
                    ))
        
        return blocks
    
    def _extract_images(self, page: fitz.Page, page_num: int, pdf_path: str) -> List[ImageInfo]:
        """Extract images from a PDF page"""
        images = []
        job_id = Path(pdf_path).stem
        image_dir = self.upload_dir / f"{job_id}_images"
        image_dir.mkdir(exist_ok=True)
        
        # Get the document from the page
        doc = page.parent
        
        image_list = page.get_images(full=True)
        print(f"[ImageExtract] Page {page_num + 1}: Found {len(image_list)} images")
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                if not base_image:
                    print(f"[ImageExtract] Could not extract image xref={xref}")
                    continue
                
                image_bytes = base_image.get("image", b"")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                ext = base_image.get("ext", "png")
                
                if not image_bytes:
                    print(f"[ImageExtract] Empty image bytes for xref={xref}")
                    continue
                
                # Save image to disk
                filename = f"image_p{page_num + 1}_{img_index + 1}.{ext}"
                image_path = image_dir / filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                print(f"[ImageExtract] Saved: {image_path} ({len(image_bytes)} bytes)")
                
                # Get actual bbox from page if possible
                img_rect = None
                try:
                    # Use get_image_rects(xref) which is more reliable for specific image instances
                    rects = page.get_image_rects(xref)
                    if rects:
                        # Use the first occurrence's rect
                        img_rect = rects[0]
                except:
                    # Fallback to get_image_info if needed
                    try:
                        img_info_list = page.get_image_info()
                        for info in img_info_list:
                            if info.get('xref') == xref:
                                img_rect = info.get('bbox')
                                break
                    except:
                        pass
                
                bbox = tuple(img_rect) if img_rect else (0, 0, width, height)
                
                # Try to find caption near image using the bbox
                caption = self._find_image_caption(page, bbox)
                
                images.append(ImageInfo(
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    page_num=page_num,
                    bbox=bbox,
                    caption=caption,
                    filename=str(image_path)
                ))
            except Exception as e:
                print(f"[ImageExtract] Error extracting image {img_index} from page {page_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # === VECTOR GRAPHICS EXTRACTION ===
        # Detect and extract vector drawings (flowcharts, diagrams)
        # Pass existing image bboxes to avoid extracting duplicates
        existing_bboxes = [img.bbox for img in images if img.bbox]
        vector_images = self._extract_vector_graphics(page, page_num, pdf_path, existing_bboxes)
        images.extend(vector_images)
        
        return images
    
    def _extract_vector_graphics(self, page: fitz.Page, page_num: int, pdf_path: str, existing_bboxes: List[tuple] = None) -> List[ImageInfo]:
        """Extract vector graphics (diagrams, flowcharts) by detecting drawing regions
        
        Args:
            existing_bboxes: List of bounding boxes of already extracted raster images to avoid duplicates
        """
        images = []
        job_id = Path(pdf_path).stem
        image_dir = self.upload_dir / f"{job_id}_images"
        image_dir.mkdir(exist_ok=True)
        existing_bboxes = existing_bboxes or []
        
        try:
            # Get page drawings (paths, lines, curves, shapes)
            drawings = page.get_drawings()
            
            if not drawings:
                return images
            
            print(f"[VectorExtract] Page {page_num + 1}: Found {len(drawings)} drawings")
            
            # Group drawings into regions (diagrams are typically clustered)
            if len(drawings) < 5:
                return images  # Not enough drawings to be a diagram
            
            # Find the bounding box of all drawings
            all_rects = []
            for d in drawings:
                if 'rect' in d:
                    all_rects.append(fitz.Rect(d['rect']))
                elif 'items' in d:
                    for item in d['items']:
                        if len(item) >= 2 and hasattr(item[1], '__iter__'):
                            try:
                                r = fitz.Rect(item[1])
                                if r.is_valid and not r.is_empty:
                                    all_rects.append(r)
                            except:
                                pass
            
            if not all_rects:
                return images
            
            # Combine all drawing rects into one bounding box
            combined_rect = all_rects[0]
            for r in all_rects[1:]:
                combined_rect = combined_rect | r  # Union
            
            # Only extract if the region is significant (not tiny graphics)
            if combined_rect.width < 100 or combined_rect.height < 100:
                return images
            
            # === CHECK FOR OVERLAP WITH EXISTING IMAGES ===
            # Skip if this region significantly overlaps with already extracted raster images
            for existing_bbox in existing_bboxes:
                if len(existing_bbox) >= 4:
                    existing_rect = fitz.Rect(existing_bbox)
                    intersection = combined_rect & existing_rect
                    if not intersection.is_empty:
                        # Calculate overlap percentage
                        overlap_area = intersection.width * intersection.height
                        combined_area = combined_rect.width * combined_rect.height
                        overlap_ratio = overlap_area / combined_area if combined_area > 0 else 0
                        
                        if overlap_ratio > 0.3:  # More than 30% overlap
                            print(f"[VectorExtract] Skipping region - overlaps with existing image ({overlap_ratio:.1%} overlap)")
                            return images
            
            # Expand the bounding box slightly
            combined_rect = combined_rect + fitz.Rect(-5, -5, 5, 5)
            combined_rect = combined_rect & page.rect  # Clip to page
            
            print(f"[VectorExtract] Rendering diagram region: {combined_rect}")
            
            # Render the region as an image
            matrix = fitz.Matrix(2.0, 2.0)  # 2x zoom for quality
            clip = combined_rect
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            
            # Save the image
            img_index = len([f for f in image_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
            filename = f"diagram_p{page_num + 1}_{img_index + 1}.png"
            image_path = image_dir / filename
            pixmap.save(str(image_path))
            
            print(f"[VectorExtract] Saved diagram: {image_path}")
            
            # Try to find caption near the diagram
            caption = self._find_vector_caption(page, combined_rect)
            
            images.append(ImageInfo(
                image_bytes=pixmap.tobytes(),
                width=pixmap.width,
                height=pixmap.height,
                page_num=page_num,
                bbox=tuple(combined_rect),
                caption=caption,
                filename=str(image_path)
            ))
            
        except Exception as e:
            print(f"[VectorExtract] Error extracting vector graphics from page {page_num}: {e}")
            import traceback
            traceback.print_exc()
        
        return images
    
    def _find_vector_caption(self, page: fitz.Page, rect: fitz.Rect) -> Optional[str]:
        """Find caption text below a vector diagram - returns just the caption text without Fig. prefix"""
        # Look for text below the diagram
        search_rect = fitz.Rect(rect.x0 - 20, rect.y1, rect.x1 + 20, rect.y1 + 60)
        text = page.get_text("text", clip=search_rect)
        
        # Look for "Fig." pattern and extract the caption text AFTER the figure number
        fig_match = re.search(r'(?:Fig\.?|Figure)\s*\d+[.:]?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if fig_match:
            # Return just the caption text (group 1), not the full match with "Fig. X"
            caption_text = fig_match.group(1).strip() if fig_match.group(1) else ""
            return caption_text if caption_text else text.strip()[:100]
        
        return text.strip()[:100] if text.strip() else None
    
    def _find_image_caption(self, page: fitz.Page, bbox: tuple) -> Optional[str]:
        """Attempt to find caption text near an image using its bbox"""
        if not bbox or bbox == (0, 0, 0, 0):
            return None
            
        x0, y0, x1, y1 = bbox
        # Define search area: below the image
        # Captions are usually immediately below.
        # Height: 100 points below should cover 2-3 lines of caption
        # Width: Expand slightly to catch centered captions that might be wider than the image
        search_rect = fitz.Rect(x0 - 50, y1, x1 + 50, y1 + 100)
        
        # Get text in this specific area
        try:
            text = page.get_text("text", clip=search_rect)
        except:
            return None
            
        if not text:
            return None
        
        # Look for common caption patterns - group(1) captures just the caption text
        # Only look for the first match, as it's the one closest to the image (top of search area)
        patterns = [
            r'Fig(?:ure)?\.?\s*\d+[.:]?\s*([^\n]+)',
            r'Figure\s+\d+[.:]?\s*([^\n]+)',
            r'FIGURE\s+\d+[.:]?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return just the caption text (group 1), not the full match with "Fig. X"
                caption_text = match.group(1).strip() if match.group(1) else None
                # Clean up newlines in caption
                if caption_text:
                    caption_text = ' '.join(caption_text.split())
                return caption_text
        
        # Fallback: if text starts with "Fig", take the whole block
        if text.strip().lower().startswith('fig'):
            return ' '.join(text.split())
            
        return None
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[TableInfo]:
        """
        Attempt to extract tables from the page
        This is a simplified extraction - complex tables may need additional processing
        """
        tables = []
        
        # Use PyMuPDF's table detection (available in newer versions)
        try:
            tab = page.find_tables()
            for table in tab:
                rows = []
                for row in table.extract():
                    rows.append([cell if cell else "" for cell in row])
                
                if rows:
                    # Try to find table caption
                    caption = self._find_table_caption(page, table.bbox)
                    
                    tables.append(TableInfo(
                        rows=rows,
                        page_num=page_num,
                        bbox=table.bbox,
                        caption=caption
                    ))
        except AttributeError:
            # Older PyMuPDF version doesn't have find_tables
            # Fall back to text-based table detection
            tables = self._detect_tables_from_text(page, page_num)
        
        return tables
    
    def _find_table_caption(self, page: fitz.Page, table_bbox: tuple) -> Optional[str]:
        """Find caption near a table"""
        text = page.get_text()
        
        patterns = [
            r'Table\s+\d+[.:]\s*([^\n]+)',
            r'TABLE\s+\d+[.:]\s*([^\n]+)',
            r'Tab(?:le)?\.?\s*\d+[.:]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _detect_tables_from_text(self, page: fitz.Page, page_num: int) -> List[TableInfo]:
        """
        Enhanced fallback table detection using text patterns
        Detects tables based on:
        - Tab-separated values
        - Multiple spaces indicating columns
        - Numeric patterns in rows
        """
        tables = []
        text = page.get_text()
        
        # First, look for explicitly labeled tables
        table_pattern = r'(?:TABLE|Table)\s+([IVX\d]+)[.:\s]*(.*?)(?=(?:TABLE|Table)\s+[IVX\d]+|$)'
        table_matches = re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in table_matches:
            table_num = match.group(1)
            table_content = match.group(2)
            caption = f"Table {table_num}"
            
            # Parse the table content
            rows = self._parse_table_content(table_content)
            if rows and len(rows) >= 2:
                tables.append(TableInfo(
                    rows=rows,
                    page_num=page_num,
                    bbox=(0, 0, 0, 0),
                    caption=caption
                ))
        
        # Also look for tabular data patterns
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if table_lines and len(table_lines) >= 2:
                    rows = self._parse_table_lines(table_lines)
                    if rows and len(rows) >= 2:
                        tables.append(TableInfo(
                            rows=rows,
                            page_num=page_num,
                            bbox=(0, 0, 0, 0),
                            caption=None
                        ))
                table_lines = []
                in_table = False
                continue
            
            # Check if line looks like table data
            if self._looks_like_table_row(line):
                table_lines.append(line)
                in_table = True
            elif in_table:
                # End of table
                if table_lines and len(table_lines) >= 2:
                    rows = self._parse_table_lines(table_lines)
                    if rows and len(rows) >= 2:
                        tables.append(TableInfo(
                            rows=rows,
                            page_num=page_num,
                            bbox=(0, 0, 0, 0),
                            caption=None
                        ))
                table_lines = []
                in_table = False
        
        return tables
    
    def _looks_like_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row"""
        # Has multiple columns separated by tabs or multiple spaces
        if '\t' in line:
            return line.count('\t') >= 1
        
        # Multiple items separated by 3+ spaces
        parts = re.split(r'\s{3,}', line)
        if len(parts) >= 2:
            return True
        
        # Contains multiple numbers (likely data)
        numbers = re.findall(r'\d+\.?\d*', line)
        if len(numbers) >= 2:
            # Check for column-like structure
            words = line.split()
            if len(words) >= 3:
                return True
        
        return False
    
    def _parse_table_content(self, content: str) -> List[List[str]]:
        """Parse table content into rows and cells"""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        return self._parse_table_lines(lines)
    
    def _parse_table_lines(self, lines: List[str]) -> List[List[str]]:
        """Parse a list of table lines into rows and cells"""
        rows = []
        
        for line in lines:
            # Try tab-separated first
            if '\t' in line:
                cells = [c.strip() for c in line.split('\t') if c.strip()]
            else:
                # Try multiple-space separation
                cells = [c.strip() for c in re.split(r'\s{2,}', line) if c.strip()]
            
            if cells:
                rows.append(cells)
        
        # Validate: all rows should have similar number of columns
        if rows:
            col_counts = [len(row) for row in rows]
            most_common = max(set(col_counts), key=col_counts.count)
            
            # Keep only rows with column count close to most common
            rows = [row for row in rows if abs(len(row) - most_common) <= 1]
            
            # Pad rows to have same number of columns
            if rows:
                max_cols = max(len(row) for row in rows)
                rows = [row + [''] * (max_cols - len(row)) for row in rows]
        
        return rows

