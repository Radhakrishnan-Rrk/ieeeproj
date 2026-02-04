import re
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IEEEInvariantEnforcer:
    """
    Enforces IEEE conference formatting rules as hard constraints.
    Validates the parsed document state before generation.
    """
    
    def __init__(self):
        self.errors = []
        
    def validate(self, parsed_doc: Any) -> Dict[str, Any]:
        """
        Run all invariant checks on the parsed document.
        Returns:
            Dict with 'passed' (bool) and 'errors' (list of strings).
        """
        self.errors = []
        
        # 1. Title Formatting
        self._validate_title(parsed_doc.title)
        
        # 2. Abstract Formatting
        if hasattr(parsed_doc, 'abstract'):
            self._validate_abstract(parsed_doc.abstract)
            
        # 3. Index Terms
        if hasattr(parsed_doc, 'keywords'):
            self._validate_index_terms(parsed_doc.keywords)
            
        # 4. Section Numbering & Structure
        if hasattr(parsed_doc, 'sections'):
            self._validate_sections(parsed_doc.sections)
            
        # 5. Figure Placement & Captions
        if hasattr(parsed_doc, 'figures'):
            self._validate_figures(parsed_doc.figures)
            
        # 6. Referencing Style (Check full text)
        if hasattr(parsed_doc, 'full_text'):
            self._validate_referencing_style(parsed_doc.full_text)
            
        # 7. Normalization & Definitions
        if hasattr(parsed_doc, 'full_text'):
            self._validate_normalization(parsed_doc.full_text)
            
        return {
            "passed": len(self.errors) == 0,
            "errors": self.errors
        }
    
    def validate_latex(self, latex_content: str) -> Dict[str, Any]:
        """
        Validate generated LaTeX code for forbidden patterns as a final Hard Gate.
        """
        errors = []
        lines = latex_content.split('\n')
        
        # Track key elements for uniqueness
        title_count = 0
        maketitle_count = 0
        author_block_count = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('%'):
                continue
            
            # === TITLE UNIQUENESS CHECK ===
            if line_stripped.startswith('\\title{'):
                title_count += 1
            if '\\maketitle' in line_stripped:
                maketitle_count += 1
            if '\\author{' in line_stripped:
                author_block_count += 1
            
            # Strict checks on output
            if re.search(r'\bFigure\s+\d+', line):
                 errors.append(f"Line {i+1}: Found forbidden 'Figure X' style in output.")
            
            if re.search(r'\bFig\s+\d+', line):
                 errors.append(f"Line {i+1}: Found forbidden 'Fig X' style (missing dot) in output.")
                 
            if re.search(r'\b(TBD|XXX)\b', line):
                 errors.append(f"Line {i+1}: Found placeholder (TBD/XXX) in output.")
                 
            if re.search(r'\\section\{\s*\d+\.', line):
                 errors.append(f"Line {i+1}: Found manual section numbering in output.")
            
            # === AUTHOR BLOCK VALIDATION ===
            # Check for designations that shouldn't appear
            if re.search(r'\\IEEEauthorblockN\{[^}]*(Professor|Asst\.|Assoc\.|Dr\.|Prof\.)', line, re.IGNORECASE):
                 errors.append(f"Line {i+1}: Author name contains title/designation.")
            
            # Check for fragmented content in author blocks
            if re.search(r'\\IEEEauthorblockN\{(Computer Science|Engineering|Department|University)\}', line, re.IGNORECASE):
                 errors.append(f"Line {i+1}: Author block contains affiliation instead of name.")
        
        # Validate counts
        if title_count > 1:
            errors.append(f"Found {title_count} \\title commands. Must have exactly 1.")
        if maketitle_count > 1:
            errors.append(f"Found {maketitle_count} \\maketitle commands. Must have exactly 1.")
        if author_block_count > 1:
            errors.append(f"Found {author_block_count} \\author blocks. Must have exactly 1.")
                 
        return {"passed": len(errors) == 0, "errors": errors}
    
    def _validate_title(self, title: str):
        if not title:
            self.errors.append("Title is missing.")
            return

        # Check for banned content
        banned = ['Professor', 'Student', 'Department', 'University', 'College', 'IEEE']
        for word in banned:
            if word.lower() in title.lower():
                self.errors.append(f"Title contains banned word: '{word}'. Title must be singular and clean.")
        
        # Check for newlines (Multi-line title in data implies not cleaned)
        if '\n' in title.strip():
             self.errors.append("Title contains newlines. Must be a single continuous line.")

    def _validate_abstract(self, abstract: str):
        if not abstract:
            return # allowed?
            
        # Check for newlines (Single paragraph enforcement)
        # Note: We replaced \n with space in generator, but here we check INPUT state.
        # If input has newlines, generator FIXES it.
        # But User says "Assert invariants. Fail or approve".
        # If generator handles it, is it an error?
        # User: "After making fixes, you MUST re-analyze... If error exists -> Reject".
        # So this Enforcer should ideally run AFTER fixes are applied?
        # Or checks if current state is fixable?
        # "You must prevent previously fixed errors from reappearing."
        # I'll check for broken words "mag- netic".
        if re.search(r'\w+-\s+\w+', abstract):
             self.errors.append("Abstract contains broken words (e.g., 'mag- netic').")
             
    def _validate_index_terms(self, keywords: List[str]):
        # Check for newlines and improper formatting
        for kw in keywords:
            # Check for line breaks
            if '\n' in kw or '\r' in kw:
                self.errors.append(f"Index term '{kw[:30]}...' contains line breaks.")
            # Check for semicolons (should be comma-separated)
            if ';' in kw:
                self.errors.append(f"Index term contains semicolon (should use commas only).")
            # Check for excessive spacing
            if '  ' in kw:
                self.errors.append(f"Index term '{kw[:30]}...' contains double spaces.")

    def _validate_sections(self, sections: List[Any]):
        conclusion_count = 0
        
        for section in sections:
            # Check if title has leaked Arabic numbers "1. Intro" instead of Roman "I. Intro"
            if re.match(r'^\d+\.', section.title):
                 self.errors.append(f"Section title contains Arabic numbering: '{section.title}'. Must use Roman numerals.")
            
            # Check for conclusion section
            if 'conclusion' in section.title.lower():
                conclusion_count += 1
            
            # Check for broken words in section content
            if hasattr(section, 'content') and section.content:
                if re.search(r'\w+-\s+\w+', section.content):
                    self.errors.append(f"Section '{section.title[:30]}' contains broken words (e.g., 'mag- netic').")
        
        # Validate single conclusion section
        if conclusion_count > 1:
            self.errors.append(f"Found {conclusion_count} conclusion sections. Must have exactly one.")

    def _validate_figures(self, figures: List[Any]):
        for fig in figures:
            # Check captions for placeholders
            if fig.caption:
                placeholder_patterns = [
                    "Caption here", "Insert caption", "Add caption",
                    "TODO", "PLACEHOLDER", "Lorem ipsum", "[caption]"
                ]
                for pattern in placeholder_patterns:
                    if pattern.lower() in fig.caption.lower():
                        self.errors.append(f"Figure {fig.number} has placeholder caption: '{pattern}'.")
                        break
            
            # Check for empty or very short captions
            if not fig.caption or len(fig.caption.strip()) < 5:
                self.errors.append(f"Figure {fig.number} has missing or too short caption.")
                    
    def _validate_referencing_style(self, text: str):
        # Disallowed formats: "Fig X", "Figure X", "figure x"
        # ONLY allowed: "Fig. X" (with period and space)
        
        # Check for "Figure X" (full word)
        if re.search(r'\bFigure\s+\d+', text, re.IGNORECASE):
             self.errors.append("Found disallowed 'Figure X' reference style. Must use 'Fig. X'.")
        
        # Check for "Fig X" without period
        if re.search(r'\bFig\s+\d+', text):
             self.errors.append("Found disallowed 'Fig X' reference style (missing period). Must use 'Fig. X'.")
        
        # Check for "Fig1" without space
        if re.search(r'\bFig\d+', text):
             self.errors.append("Found 'FigX' format (missing period and space). Must use 'Fig. X'.")

    def _validate_normalization(self, text: str):
        # Check for ALL placeholder patterns - STRICT
        
        # Check for TBD, XXX (standalone)
        if re.search(r'\bTBD\b', text, re.IGNORECASE):
             self.errors.append("Found placeholder 'TBD'. Must use explicit range like [0, 1].")
        if re.search(r'\bXXX\b', text):
             self.errors.append("Found placeholder 'XXX'. Must use explicit range like [0, 1].")
        
        # Check for [?], [TBD], [XXX]
        if re.search(r'\[\s*\?\s*\]', text):
             self.errors.append("Found placeholder '[?]'. Must use explicit range like [0, 1].")
        if re.search(r'\[\s*(TBD|XXX)\s*\]', text, re.IGNORECASE):
             self.errors.append("Found placeholder in brackets. Must use explicit range like [0, 1].")
        
        # Check for incomplete ranges [0, ?] or [?, 1]
        if re.search(r'\[\s*\d+\s*,\s*\?\s*\]', text):
             self.errors.append("Found incomplete range with '?'. Must specify complete range.")
        if re.search(r'\[\s*\?\s*,\s*\d+\s*\]', text):
             self.errors.append("Found incomplete range with '?'. Must specify complete range.")
