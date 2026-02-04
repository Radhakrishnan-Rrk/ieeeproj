"""
Image Handler Module
Processes and resizes images for IEEE column width
"""

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io


class ImageHandler:
    """
    Handles image processing for IEEE format
    - Resizes to IEEE column width (3.5 inches at 300 DPI = 1050 pixels)
    - Converts to compatible formats
    - Optimizes for PDF embedding
    """
    
    # IEEE column width: 3.5 inches at 300 DPI
    IEEE_COLUMN_WIDTH_PX = 1050
    
    # Maximum height to prevent overly tall images
    MAX_HEIGHT_PX = 1400
    
    # Supported output formats
    SUPPORTED_FORMATS = ['PNG', 'JPEG', 'PDF']
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_image(self, image_path: str, job_id: str, index: int) -> Optional[str]:
        """
        Process an image for IEEE format
        
        Args:
            image_path: Path to the source image
            job_id: Job ID for naming
            index: Image index for naming
            
        Returns:
            Path to processed image, or None if processing failed
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary (for JPEG output)
            if img.mode in ('RGBA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to fit IEEE column width
            img = self._resize_for_ieee(img)
            
            # Save processed image
            output_path = self.output_dir / f"{job_id}_fig_{index}.png"
            img.save(output_path, 'PNG', optimize=True)
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def _resize_for_ieee(self, img: Image.Image) -> Image.Image:
        """Resize image to fit IEEE column width while maintaining aspect ratio"""
        width, height = img.size
        
        # Only resize if image is wider than column width
        if width > self.IEEE_COLUMN_WIDTH_PX:
            ratio = self.IEEE_COLUMN_WIDTH_PX / width
            new_size = (self.IEEE_COLUMN_WIDTH_PX, int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Check if height is too tall after width resize
        if img.height > self.MAX_HEIGHT_PX:
            ratio = self.MAX_HEIGHT_PX / img.height
            new_size = (int(img.width * ratio), self.MAX_HEIGHT_PX)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            img = Image.open(image_path)
            return img.size
        except Exception:
            return (0, 0)
    
    def create_placeholder(self, width: int = 400, height: int = 300, text: str = "Image") -> bytes:
        """Create a placeholder image when original cannot be processed"""
        from PIL import ImageDraw
        
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle([0, 0, width-1, height-1], outline=(200, 200, 200), width=2)
        
        # Draw text
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=(150, 150, 150))
        
        # Return as bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
