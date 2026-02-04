"""
Table Handler Module
Processes tables for IEEE LaTeX format
"""

from typing import List, Optional


class TableHandler:
    """
    Handles table processing for IEEE format
    - Converts table data to LaTeX tabular environment
    - Handles column alignment and formatting
    - Supports multi-column tables
    """
    
    def __init__(self):
        self.default_alignment = 'l'  # Left-align by default
    
    def to_latex(self, rows: List[List[str]], caption: str = None, label: str = None) -> str:
        """
        Convert table rows to LaTeX tabular environment
        
        Args:
            rows: List of rows, each row is a list of cell values
            caption: Optional table caption
            label: Optional LaTeX label
            
        Returns:
            LaTeX table code
        """
        if not rows:
            return ""
        
        # Determine number of columns
        num_cols = max(len(row) for row in rows)
        
        # Generate column specification
        col_spec = self._generate_col_spec(rows, num_cols)
        
        # Build LaTeX table
        latex_parts = []
        
        # Table environment
        latex_parts.append("\\begin{table}[htbp]")
        latex_parts.append("\\centering")
        
        # Caption (IEEE style: caption at top)
        if caption:
            escaped_caption = self._escape_latex(caption)
            latex_parts.append(f"\\caption{{{escaped_caption}}}")
        
        if label:
            latex_parts.append(f"\\label{{{label}}}")
        
        # Tabular environment
        latex_parts.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_parts.append("\\toprule")
        
        # Process rows
        for i, row in enumerate(rows):
            # Pad row to have correct number of columns
            padded_row = row + [''] * (num_cols - len(row))
            
            # Escape special characters
            escaped_cells = [self._escape_latex(cell) for cell in padded_row]
            
            # Join cells with &
            row_str = ' & '.join(escaped_cells) + ' \\\\'
            latex_parts.append(row_str)
            
            # Add midrule after header (first row)
            if i == 0:
                latex_parts.append("\\midrule")
        
        # Close environments
        latex_parts.append("\\bottomrule")
        latex_parts.append("\\end{tabular}")
        latex_parts.append("\\end{table}")
        
        return '\n'.join(latex_parts)
    
    def _generate_col_spec(self, rows: List[List[str]], num_cols: int) -> str:
        """
        Generate column specification based on content
        
        Args:
            rows: Table rows
            num_cols: Number of columns
            
        Returns:
            LaTeX column specification string
        """
        # Analyze columns for content type
        col_specs = []
        
        for col_idx in range(num_cols):
            # Get all values in this column
            col_values = []
            for row in rows:
                if col_idx < len(row):
                    col_values.append(row[col_idx])
            
            # Determine alignment based on content
            alignment = self._determine_alignment(col_values)
            col_specs.append(alignment)
        
        return ''.join(col_specs)
    
    def _determine_alignment(self, values: List[str]) -> str:
        """
        Determine column alignment based on content
        
        Args:
            values: List of cell values in the column
            
        Returns:
            'l', 'c', or 'r' for alignment
        """
        if not values:
            return 'l'
        
        # Count numeric values
        numeric_count = 0
        for val in values[1:]:  # Skip header
            if val.strip():
                try:
                    float(val.replace(',', '').replace('%', ''))
                    numeric_count += 1
                except ValueError:
                    pass
        
        # If mostly numeric, right-align
        if len(values) > 1 and numeric_count > (len(values) - 1) * 0.5:
            return 'r'
        
        return 'l'
    
    def _escape_latex(self, text: str) -> str:
        """Escape LaTeX special characters"""
        if not text:
            return ""
        
        # LaTeX special characters
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def detect_header_row(self, rows: List[List[str]]) -> bool:
        """
        Detect if the first row is a header row
        
        Args:
            rows: Table rows
            
        Returns:
            True if first row appears to be a header
        """
        if len(rows) < 2:
            return False
        
        first_row = rows[0]
        second_row = rows[1]
        
        # Check if first row is all text and different from data rows
        first_row_numeric = sum(1 for cell in first_row if self._is_numeric(cell))
        second_row_numeric = sum(1 for cell in second_row if self._is_numeric(cell))
        
        # If first row has significantly fewer numbers, it's likely a header
        return first_row_numeric < second_row_numeric
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric"""
        if not value or not value.strip():
            return False
        try:
            float(value.replace(',', '').replace('%', ''))
            return True
        except ValueError:
            return False
    
    def merge_cells(self, rows: List[List[str]], start_col: int, end_col: int, row_idx: int) -> List[List[str]]:
        """
        Merge cells in a row (for multi-column support)
        
        Args:
            rows: Table rows
            start_col: Starting column index
            end_col: Ending column index
            row_idx: Row index to merge
            
        Returns:
            Modified rows with merged cell
        """
        if row_idx >= len(rows):
            return rows
        
        row = rows[row_idx]
        if start_col >= len(row) or end_col >= len(row):
            return rows
        
        # Get merged content
        merged_content = ' '.join(row[start_col:end_col + 1])
        span = end_col - start_col + 1
        
        # Create new row with multicolumn
        new_row = row[:start_col] + [f"\\multicolumn{{{span}}}{{c}}{{{merged_content}}}"]
        if end_col + 1 < len(row):
            new_row.extend(row[end_col + 1:])
        
        rows[row_idx] = new_row
        return rows
