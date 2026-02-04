"""
Content Analyzer Module
Analyzes parsed document content to extract structured IEEE paper components
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from text_processor import TextProcessor
from authorparser import IEEEAuthorParser


@dataclass
class Author:
    """Represents a paper author"""
    name: str
    affiliation: str = ""
    email: str = ""


@dataclass
class Section:
    """Represents a paper section"""
    title: str
    content: str
    level: int  # 1 = main section (I, II), 2 = subsection (A, B)
    number: str = ""  # e.g., "I", "II", "A", "B"


@dataclass
class Reference:
    """Represents a bibliographic reference"""
    number: int
    text: str
    authors: str = ""
    title: str = ""
    venue: str = ""
    year: str = ""


@dataclass
class Figure:
    """Represents a figure with caption"""
    number: int
    image_path: str
    caption: str
    width: int = 0
    height: int = 0


@dataclass
class Table:
    """Represents a table with caption"""
    number: int
    rows: List[List[str]]
    caption: str


@dataclass
class StructuredPaper:
    """Complete structured IEEE paper"""
    title: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    authors: List[Author] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)


class ContentAnalyzer:
    """
    Analyzes parsed document content and extracts structured components
    for IEEE paper formatting
    """
    
    # Common section titles in academic papers
    SECTION_PATTERNS = [
        r'^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.?\s+(.+)$',  # Roman numerals
        r'^(\d+)\.?\s+(.+)$',  # Arabic numerals
        r'^([A-Z])\.?\s+(.+)$',  # Letter subsections
    ]
    
    ABSTRACT_KEYWORDS = [
        'abstract', 'summary', 'overview'
    ]
    
    REFERENCE_SECTION_KEYWORDS = [
        'references', 'bibliography', 'works cited', 'literature cited'
    ]
    
    def __init__(self):
        self.roman_to_int = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
        }
    
    def analyze(self, parsed_doc) -> StructuredPaper:
        """
        Analyze a parsed document and extract structured content
        
        Args:
            parsed_doc: ParsedDocument from PDF or DOCX parser
            
        Returns:
            StructuredPaper with all components extracted
        """
        paper = StructuredPaper()
        
        # Extract title (usually largest font or first heading)
        paper.title = self._extract_title(parsed_doc)
        
        # Extract authors and affiliations
        paper.authors = self._extract_authors(parsed_doc)
        
        # Extract abstract
        paper.abstract = self._extract_abstract(parsed_doc)
        
        # Extract keywords
        paper.keywords = self._extract_keywords(parsed_doc)
        
        # Extract sections and clean content
        paper.sections = self._extract_sections(parsed_doc)
        for section in paper.sections:
            section.content = TextProcessor.clean_text(section.content)
        
        # Extract references
        paper.references = self._extract_references(parsed_doc)
        
        # Process figures and clean captions
        paper.figures = self._process_figures(parsed_doc)
        for fig in paper.figures:
            fig.caption = TextProcessor.clean_text(fig.caption)
        
        # Process tables
        paper.tables = self._process_tables(parsed_doc)
        
        # Extract equations
        paper.equations = self._extract_equations(parsed_doc)
        
        return paper
    
    def _extract_title(self, parsed_doc) -> str:
        """Extract the paper title"""
        if not parsed_doc.text_blocks:
            # Fallback to first line of full text
            lines = parsed_doc.full_text.strip().split('\n')
            return lines[0] if lines else "Untitled Paper"
        
        # Find the largest font text block on first page
        first_page_blocks = [b for b in parsed_doc.text_blocks if b.page_num == 0]
        
        if first_page_blocks:
            # Sort by font size (largest first)
            sorted_blocks = sorted(first_page_blocks, key=lambda x: x.font_size, reverse=True)
            
            # Title is usually the largest text that's reasonably short
            for block in sorted_blocks:
                text = block.text.strip()
                if len(text) < 300 and len(text) > 5:
                    cleaned_title = self._clean_extracted_title(text)
                    if cleaned_title:
                        return cleaned_title
        
        # Fallback: check for heading blocks
        for block in parsed_doc.text_blocks:
            if hasattr(block, 'is_heading') and block.is_heading:
                if hasattr(block, 'heading_level') and block.heading_level <= 1:
                    return self._clean_extracted_title(block.text)
        
        return "Untitled Paper"

    def _clean_extracted_title(self, text: str) -> str:
        """Clean title text by removing non-title tokens and deduplicating"""
        if not text:
            return ""
            
        # 1. Split into lines
        lines = text.split('\n')
        valid_lines = []
        
        # 2. Filter disallowed content (Designations, Affiliations, Author names)
        disallowed_keywords = [
            'professor', 'student', 'scholar', 'engineer', 'department', 
            'dept', 'university', 'college', 'institute', 'school',
            'faculty', 'abstract', 'introduction', 'ieee', 'conference',
            'vol.', 'no.', 'pp.', 'issue', 'doi', 'email', 'asst.',
            'research scholar', 'computer science', 'engineering', 'technology',
            'madanapalle', 'cyber security', 'gmail.com', '@', 'india'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for disallowed keywords
            if any(kw in line.lower() for kw in disallowed_keywords):
                continue
                
            # Check for email
            if '@' in line:
                continue
                
            # Check for generic labels
            if line.lower() in ['paper title', 'title']:
                continue
            
            # Skip lines that look like names (only 2-3 proper words)
            words = line.split()
            if len(words) <= 3 and all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
                continue
                
            valid_lines.append(line)
        
        if not valid_lines:
            return ""
            
        # 3. Deduplication (e.g. "My Title My Title")
        full_text = ' '.join(valid_lines)
        
        # Simple repeat check: "ABC ABC" -> "ABC"
        half = len(full_text) // 2
        if len(full_text) > 10 and full_text[:half].strip() == full_text[half:].strip():
            full_text = full_text[:half].strip()
        
        # 4. Remove any repeated words pattern (e.g., "Architecture With Reversible Logic Gate Integration M. Sri Lakshmi Preethi")
        # Check if last part looks like a name and remove it
        words = full_text.split()
        if len(words) > 3:
            # Check if last 2-4 words are a name pattern
            potential_name = ' '.join(words[-4:])
            if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z.]+){1,3}$', potential_name):
                # Remove the trailing name
                full_text = ' '.join(words[:-4]).strip() or full_text
            
        # 5. Enforce Title Case (Capitalize words)
        words = full_text.split()
        title_cased = []
        # Minor words for title casing
        keywords_lower = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'with', 'to', 'as'}
        
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word.lower() not in keywords_lower:
                title_cased.append(word.capitalize())
            else:
                title_cased.append(word.lower())
        
        return ' '.join(title_cased)
    
    def _extract_authors(self, parsed_doc) -> List[Author]:
        """Extract author names and affiliations"""
        authors = []
        full_text = parsed_doc.full_text
        
        # Common patterns for author lines
        # Pattern 1: "FirstName LastName, Affiliation"
        # Pattern 2: "FirstName LastName1, FirstName LastName2"
        
        # Look in the first 20% of the document
        search_area = full_text[:len(full_text) // 5]
        lines = search_area.split('\n')
        
        # Skip title (first non-empty line usually)
        found_title = False
        author_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if not found_title:
                found_title = True
                continue
            
            # Stop at "Abstract" or section headers
            if any(kw in line.lower() for kw in self.ABSTRACT_KEYWORDS):
                break
            if re.match(r'^(?:I{1,3}|IV|V|1)\.?\s+', line):
                break
            
            # Check if line looks like author names
            # Authors typically have: capitalized words, possibly with commas
            if self._looks_like_author_line(line):
                author_lines.append(line)
        
        # Parse author lines
        for line in author_lines[:5]:  # Limit to 5 author lines
            extracted = self._parse_author_line(line)
            authors.extend(extracted)
        
        return authors
    
    def _looks_like_author_line(self, line: str) -> bool:
        """Check if a line appears to contain author names"""
        # Single words are not author lines
        words = line.split()
        if len(words) < 2:
            return False
        
        # Filter out obvious non-author content
        non_author_keywords = [
            'abstract', 'introduction', 'keywords', 'index terms',
            'engineering', 'technology', 'sciences', 'systems',
            'chennai', 'mumbai', 'delhi', 'bangalore', 'india',
            'usa', 'china', 'germany', 'california', 'texas',
            'classification', 'detection', 'analysis', 'framework',
            'deep learning', 'neural', 'network', 'cnn', 'hybrid'
        ]
        if any(kw in line.lower() for kw in non_author_keywords):
            return False
        
        # Check for email patterns (often indicates author section)
        if '@' in line:
            return True
        
        # Author names typically have first and last names
        # Must have at least 2 capitalized words that look like names
        name_words = [w for w in words if w and w[0].isupper() and len(w) > 1]
        
        # Skip lines with numbers (publication years, page numbers)
        if any(c.isdigit() for c in line):
            return False
        
        return len(name_words) >= 2 and len(line) < 100
    
    def _parse_author_line(self, line: str) -> List[Author]:
        """Parse a line to extract author information using IEEE-compliant formatting"""
        authors = []
        parser = IEEEAuthorParser()
        
        # Try to split by comma, "and", or semicolon
        parts = re.split(r',\s*(?:and\s+)?|\s+and\s+|;\s*', line)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for email in the part
            email_match = re.search(r'[\w.-]+@[\w.-]+\.\w+', part)
            email = email_match.group() if email_match else ""
            
            if email:
                part = part.replace(email, '').strip()
            
            # Use IEEE Author Parser to clean the name
            cleaned_name = parser._clean_author_name(part)
            
            # Simple heuristic: if it looks like a name (2-4 words, capitalized)
            words = cleaned_name.split()
            if 1 <= len(words) <= 5 and cleaned_name:
                # Validate cleaned name doesn't contain titles/degrees
                if not any(re.search(p, cleaned_name, re.IGNORECASE) for p in parser.TITLE_PATTERNS):
                    authors.append(Author(
                        name=cleaned_name,
                        affiliation="",
                        email=email.lower() if email else ""
                    ))
        
        return authors
    
    def _extract_abstract(self, parsed_doc) -> str:
        """Extract the abstract section with validation for IEEE format"""
        full_text = parsed_doc.full_text
        
        # Look for "Abstract" keyword
        patterns = [
            r'(?:^|\n)\s*Abstract[:\s]*\n?(.*?)(?=\n\s*(?:Keywords?|Index Terms?|I+\.?\s+[A-Z]|\d+\.?\s+[A-Z]))',
            r'(?:^|\n)\s*ABSTRACT[:\s]*\n?(.*?)(?=\n\s*(?:KEYWORDS?|INDEX TERMS?|I+\.?\s+[A-Z]|\d+\.?\s+[A-Z]))',
        ]
        
        abstract = ""
        for pattern in patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Apply Cleaning
                abstract = TextProcessor.clean_text(abstract)
                # Force Single Paragraph (Replace newlines with space)
                abstract = re.sub(r'\s*\n\s*', ' ', abstract).strip()
                return abstract
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                break
        
        if not abstract:
            # Fallback: look for a paragraph after potential abstract header
            lines = full_text.split('\n')
            capture = False
            abstract_lines = []
            
            for line in lines:
                if any(kw in line.lower() for kw in self.ABSTRACT_KEYWORDS):
                    capture = True
                    continue
                
                if capture:
                    if re.match(r'^(?:I{1,3}|IV|V|1)\.?\s+[A-Z]', line):
                        break
                    if 'keyword' in line.lower() or 'index term' in line.lower():
                        break
                    abstract_lines.append(line)
                    if len(abstract_lines) > 10:  # Limit abstract search
                        break
            
            abstract = ' '.join(abstract_lines).strip()
        
        # === IEEE VALIDATION ===
        # Clean up abstract ending - ensure it ends with complete sentence
        abstract = self._clean_abstract_ending(abstract)
        
        # Validate word count (IEEE: 150-250 words)
        word_count = len(abstract.split())
        if word_count < 50 and abstract:  # Too short - might be truncated
            print(f"[Warning] Abstract may be truncated ({word_count} words)")
        elif word_count > 300:  # Too long - trim to last complete sentence around 250 words
            words = abstract.split()
            trimmed = ' '.join(words[:250])
            # Find last sentence boundary
            last_period = trimmed.rfind('.')
            if last_period > 100:
                abstract = trimmed[:last_period + 1]
        
        return abstract
    
    def _clean_abstract_ending(self, abstract: str) -> str:
        """Ensure abstract ends with a complete sentence"""
        if not abstract:
            return abstract
        
        # Remove trailing incomplete sentences (no period at end)
        abstract = abstract.strip()
        if abstract and not abstract.endswith(('.', '!', '?')):
            # Find last complete sentence
            last_period = abstract.rfind('.')
            last_exclaim = abstract.rfind('!')
            last_question = abstract.rfind('?')
            last_boundary = max(last_period, last_exclaim, last_question)
            if last_boundary > len(abstract) * 0.5:  # Only if we have substantial content
                abstract = abstract[:last_boundary + 1]
        
        # Remove any Index Terms that leaked into abstract
        abstract = re.sub(r'\s*Index Terms[—–-]?.*$', '', abstract, flags=re.IGNORECASE)
        abstract = re.sub(r'\s*Keywords?[:\s].*$', '', abstract, flags=re.IGNORECASE)
        
        return abstract.strip()
    
    def _extract_keywords(self, parsed_doc) -> List[str]:
        """Extract keywords from the paper with IEEE validation (4-6 terms)"""
        full_text = parsed_doc.full_text
        
        # Look for "Keywords" or "Index Terms" section
        # CRITICAL FIX: Capture everything until Introduction to avoid missing terms
        # This handles cases where terms span multiple lines or have weird formatting
        
        # 1. Find the start of keywords
        start_match = re.search(r'(?:Keywords?|Index\s+Terms?)[:\s—–-]*', full_text, re.IGNORECASE)
        keywords = []
        
        if start_match:
            start_pos = start_match.end()
            remaining_text = full_text[start_pos:]
            
            # 2. Find the end (Introduction or next section)
            # Look for "I. Introduction", "1. Introduction", or just "Introduction" as section header
            end_match = re.search(r'(?:\n\s*|^)(?:I\.?|1\.?)?\s*Introduction', remaining_text, re.IGNORECASE)
            
            if end_match:
                # Take everything between Keywords and Introduction
                keywords_block = remaining_text[:end_match.start()].strip()
            else:
                # Fallback: take next 500 chars or until double newline
                keywords_block = remaining_text[:500].split('\n\n')[0]
            
            # 3. Clean and split
            # Remove any leading punctuation often included like "—"
            keywords_block = keywords_block.lstrip('—–- :')
            
            # Split by comma, semicolon, bullet points, or double spaces
            # Also handle newlines as separators if no commas present
            if ',' in keywords_block or ';' in keywords_block:
                delimiters = r'[,;•·]'
            else:
                delimiters = r'[,;•·]|\n|\s{2,}'
                
            raw_keywords = re.split(delimiters, keywords_block)
            keywords = [kw.strip() for kw in raw_keywords if kw.strip() and len(kw.strip()) > 2]
        
        # === IEEE VALIDATION ===
        cleaned_keywords = []
        for kw in keywords:
            # Skip if looks truncated (ends with incomplete word pattern)
            if kw.endswith(('-', '–', '—')):
                continue
            
            # 1. Clean whitespace and aggressive punctuation from ends
            # Fixes "deep learning. " -> "deep learning"
            kw = kw.strip()
            kw = re.sub(r'^[\s.,;:+–—"\'`]+|[\s.,;:+–—"\'`]+$', '', kw)
            
            # Skip if too short to be meaningful
            if len(kw) < 2:
                continue
                
            # 2. Clean up internal formatting
            # Remove "and" if it's linking last term
            kw = re.sub(r'^and\s+', '', kw, flags=re.IGNORECASE)
            # Remove space before punctuation
            kw = re.sub(r'\s+([-.,;:!?])', r'\1', kw)
            # Merge multiline terms ("Deep\nLearning" -> "Deep Learning")
            kw = re.sub(r'\s+', ' ', kw)
            
            # Explicit fix for hyphenation in keywords
            kw = kw.replace('time- consuming', 'time-consuming')
            
            # 3. Enforce "Title Case Per Term" (e.g., "Deep Learning", "Neural Networks")
            # Custom title case to keep small words lowercase if not first/last
            words = kw.split()
            title_cased = []
            small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'with', 'to', 'as'}
            
            for i, word in enumerate(words):
                # Preserve acronyms like "MRI", "CNN" if they are already uppercase
                if word.isupper() and len(word) > 1:
                    title_cased.append(word)
                elif i == 0 or i == len(words) - 1 or word.lower() not in small_words:
                    title_cased.append(word.capitalize())
                else:
                    title_cased.append(word.lower())
            
            kw = ' '.join(title_cased)
            
            if kw:
                cleaned_keywords.append(kw)
        
        # IEEE recommends 4-6 keywords (but capture more if available)
        # Sort alphabetically (IEEE style)
        cleaned_keywords.sort(key=lambda x: x.lower())
        
        # Deduplicate while preserving order (case insensitive)
        unique_keywords = []
        seen = set()
        for kw in cleaned_keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)
                
        return unique_keywords[:8]  # cap at 8
    
    def _extract_sections(self, parsed_doc) -> List[Section]:
        """Extract paper sections with content"""
        sections = []
        full_text = parsed_doc.full_text
        lines = full_text.split('\n')
        
        current_section = None
        current_content = []
        section_count = 0
        subsection_count = 0
        found_first_section = False
        
        # Skip lines that are part of header (title, authors, abstract, keywords)
        skip_keywords = ['abstract', 'keyword', 'index term', 'email', '@gmail', '@yahoo', '@outlook', 
                        'department', 'university', 'institute', 'professor', 'student']
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # CRITICAL FIX: Stop processing sections if we hit References/Bibliography
            # This prevents the References content from being appended to the Conclusion section
            stop_pattern = r'^(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|ACKNOWLEDGMENT|Acknowledgment|ACKNOWLEDGEMENTS)\b'
            if re.match(stop_pattern, line_stripped) and len(line_stripped) < 30:
                # We hit the end of the main body
                if current_section:
                    current_section.content = ' '.join(current_content).strip()
                    if current_section.content and self._is_valid_section(current_section):
                        sections.append(current_section)
                break
            
            # === FILTER OUT CAPTIONS LEAKED IN TEXT ===
            # User reported: "Fig. 1. Proposed workflow." appearing inside paragraphs.
            # We filter out lines that strictly look like captions (start with Fig X. or Figure X:)
            # But preserve sentences like "Fig. 1 shows that..." (no dot/colon immediately after number)
            # Regex requires dot or colon AFTER the number: "Fig. 1." or "Fig 1:"
            # Also handles "Fig. 1 Sample image" (missing dot but looking like caption if short?)
            # Let's stick to strict separator or "Fig. 2 Sample..." as reported
            if re.match(r'^(?:Fig\.?|Figure|FIGURE)\s*\d+([.:]|\s{2,})\s+', line_stripped):
                continue
            # Also filter "Fig. 2 Sample..." (space but no dot? risky? User said "Fig. 2 Sample")
            # If line is short (<100 chars) and starts with Fig/Figure + Num, likely caption.
            if re.match(r'^(?:Fig\.?|Figure|FIGURE)\s*\d+\s+[A-Z]', line_stripped) and len(line_stripped) < 150:
                 # Check if it contains "shows", "illustrates", "demonstrates" (verb) -> Keep
                 # Else -> Skip
                 if not re.search(r'\b(shows|illustrates|demonstrates|depicts|presents)\b', line_stripped.lower()):
                     continue
            
            # Check for section headers
            section_match = self._match_section_header(line_stripped)
            
            if section_match:
                level, title = section_match
                
                # Only start capturing at first major section (level 1)
                # This skips any preamble content before INTRODUCTION
                if not found_first_section:
                    if level == 1:
                        found_first_section = True
                    else:
                        continue  # Skip subsections before first main section
                
                # Save previous section
                if current_section:
                    current_section.content = ' '.join(current_content).strip()
                    # Validate section has content before adding
                    if current_section.content and self._is_valid_section(current_section):
                        sections.append(current_section)
                
                if level == 1:
                    section_count += 1
                    subsection_count = 0
                    number = self._int_to_roman(section_count)
                else:
                    subsection_count += 1
                    number = chr(64 + subsection_count)  # A, B, C...
                
                # Validate section title is complete (not truncated)
                title = self._clean_section_title(title)
                
                current_section = Section(
                    title=title,
                    content="",
                    level=level,
                    number=number
                )
                current_content = []
            elif current_section and found_first_section:
                # Skip lines that look like header content
                line_lower = line_stripped.lower()
                if any(kw in line_lower for kw in skip_keywords):
                    continue
                # Skip very short lines that look like author separators
                if len(line_stripped) < 3:
                    continue
                current_content.append(line_stripped)
        
        # Save last section
        if current_section:
            current_section.content = ' '.join(current_content).strip()
            if current_section.content and self._is_valid_section(current_section):
                sections.append(current_section)
        
        # === DEDUPLICATION AND CONCLUSION MERGING ===
        sections = self._deduplicate_sections(sections)
        
        return sections
    
    def _deduplicate_sections(self, sections: List[Section]) -> List[Section]:
        """Remove duplicate sections and merge multiple conclusions into one"""
        if not sections:
            return sections
        
        # Track seen section titles (normalized)
        seen_titles = set()
        deduplicated = []
        conclusion_sections = []
        
        for section in sections:
            # Normalize title for comparison
            normalized_title = section.title.lower().strip()
            normalized_title = re.sub(r'\s+', ' ', normalized_title)
            
            # Check for conclusion sections
            if 'conclusion' in normalized_title:
                conclusion_sections.append(section)
                continue
            
            # Skip duplicates
            if normalized_title in seen_titles:
                continue
            
            seen_titles.add(normalized_title)
            deduplicated.append(section)
        
        # Merge all conclusion sections into one
        if conclusion_sections:
            merged_content = ' '.join([s.content for s in conclusion_sections if s.content])
            merged_conclusion = Section(
                title="CONCLUSION",
                content=merged_content,
                level=1,
                number=self._int_to_roman(len(deduplicated) + 1)
            )
            deduplicated.append(merged_conclusion)
        
        return deduplicated
    
    def _clean_section_title(self, title: str) -> str:
        """Clean and validate section title"""
        if not title:
            return title
        
        # Remove trailing punctuation that might indicate truncation
        title = title.strip()
        title = title.rstrip(':').strip()
        
        # Remove any embedded figure/table references that leaked into title
        title = re.sub(r'\s*Fig\.?\s*\d+.*$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*Table\s*[IVX\d]+.*$', '', title, flags=re.IGNORECASE)
        
        # Ensure title doesn't end mid-word (truncation indicator)
        if title and title[-1].islower() and len(title) > 20:
            # Might be truncated, take up to last space
            last_space = title.rfind(' ')
            if last_space > len(title) * 0.5:
                title = title[:last_space].strip()
        
        return title
    
    def _is_valid_section(self, section: Section) -> bool:
        """Validate that a section is complete and valid"""
        # Check title is not empty or just whitespace
        if not section.title or not section.title.strip():
            return False
        
        # Check title is not too short (likely truncated)
        if len(section.title) < 3:
            return False
        
        # Check content exists
        if not section.content or len(section.content) < 10:
            return False
        
        return True
    
    def _match_section_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Match a line against section header patterns"""
        if not line or len(line) > 100:
            return None
        
        # Clean the line
        line = line.strip()
        
        # REJECT sentences as section titles (Fix for "PERFORMANCE IS EVALUATED...")
        # If title contains verbs or is a long sentence ending in period
        # BUT explicitly allow patterns that look like headers (starting with numbering)
        is_header_pattern = re.match(r'^(?:I{1,3}|IV|V|VI|\d+|[A-Z])\.?\s+', line)
        
        if len(line) > 20 and (
            re.search(r'\b(is|are|was|were|has|have|shows|presents|illustrates)\b', line.lower()) or
            (line.endswith('.') and not is_header_pattern)
        ):
            return None
        
        # Roman numeral sections (I., II., etc.)
        match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X)\.?\s+(.+)$', line, re.IGNORECASE)
        if match:
            return (1, match.group(2).strip())
        
        # Arabic numeral sections (1., 2., etc.)
        # Only if followed by capital letter
        match = re.match(r'^(\d+)\.?\s+([A-Z].+)$', line)
        if match:
            return (1, match.group(2).strip())
        
        # Letter subsections (A., B., etc.)
        match = re.match(r'^([A-Z])\.?\s+(.+)$', line)
        if match and len(line) < 80:
            return (2, match.group(2).strip())
        
        return None
    
    def _int_to_roman(self, num: int) -> str:
        """Convert integer to Roman numeral"""
        roman_map = [
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        result = ''
        for value, numeral in roman_map:
            while num >= value:
                result += numeral
                num -= value
        return result
    
    def _extract_references(self, parsed_doc) -> List[Reference]:
        """Extract and parse references/bibliography with improved detection"""
        references = []
        full_text = parsed_doc.full_text
        
        # Find the references section with more patterns
        ref_start = None
        patterns_to_try = [
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*References\s*\n', 
            r'\n\s*Bibliography\s*\n',
            r'\bREFERENCES\b',
            r'\bReferences\b',
            r'REFERENCES\s*$',  # At end of line
        ]
        
        for pattern in patterns_to_try:
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                ref_start = match.end()
                break
        
        if ref_start is None:
            return references
        
        ref_text = full_text[ref_start:]
        
        # Try multiple reference patterns
        # Pattern 1: [1] Author, "Title," venue, year format
        matches = re.findall(r'\[(\d+)\]\s*([^\[\]]+?)(?=\[\d+\]|\Z)', ref_text, re.DOTALL)
        
        if not matches:
            # Pattern 2: Simple numbered list 1. 2. etc
            matches = re.findall(r'^(\d+)\.\s*(.+?)(?=^\d+\.|\Z)', ref_text, re.MULTILINE | re.DOTALL)
        
        if not matches:
            # Pattern 3: Look for citation patterns inline like \cite{ref1} followed by text
            cite_pattern = r'\\cite\{ref(\d+)\}\s*([^\\]+?)(?=\\cite|$)'
            matches = re.findall(cite_pattern, ref_text)
        
        for num, text in matches:
            # Clean up the text - remove newlines and extra whitespace
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split()).strip()
            
            # Remove trailing punctuation artifacts
            text = re.sub(r'[,;]$', '', text).strip()
            
            if text and len(text) > 10:  # Skip very short entries
                references.append(Reference(
                    number=int(num),
                    text=text,
                    authors=self._extract_ref_authors(text),
                    title=self._extract_ref_title(text),
                    year=self._extract_ref_year(text),
                    venue=self._extract_ref_venue(text)
                ))
        
        # === IEEE VALIDATION ===
        # Sort by reference number first
        references.sort(key=lambda r: r.number)
        
        # Deduplicate references (same title or very similar text)
        unique_refs = []
        seen_titles = set()
        seen_texts = set()
        
        for ref in references:
            # Check for duplicate title
            title_key = ref.title.lower().strip() if ref.title else ""
            text_key = ref.text[:50].lower().strip() if ref.text else ""
            
            if title_key and title_key in seen_titles:
                continue  # Skip duplicate
            if text_key and text_key in seen_texts:
                continue  # Skip duplicate
            
            if title_key:
                seen_titles.add(title_key)
            if text_key:
                seen_texts.add(text_key)
            
            unique_refs.append(ref)
        
        # Renumber sequentially (1, 2, 3, ...)
        for i, ref in enumerate(unique_refs):
            ref.number = i + 1
        
        return unique_refs
    
    def _extract_ref_authors(self, ref_text: str) -> str:
        """Extract authors from a reference"""
        # Authors are typically at the start, before quotes or title
        match = re.match(r'^([^"]+?)(?:,\s*"|\.?\s+")', ref_text)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_ref_title(self, ref_text: str) -> str:
        """Extract title from a reference"""
        match = re.search(r'"([^"]+)"', ref_text)
        if match:
            return match.group(1)
        return ""
    
    def _extract_ref_year(self, ref_text: str) -> str:
        """Extract year from a reference"""
        match = re.search(r'\b(19|20)\d{2}\b', ref_text)
        if match:
            return match.group()
        return ""
    
    def _extract_ref_venue(self, ref_text: str) -> str:
        """Extract venue (journal/conference name) from a reference"""
        # Look for italicized text or text after title and before year
        # Common patterns: ", Journal Name," or "in Proceedings of..."
        patterns = [
            r'in\s+(Proc\.|Proceedings\s+of\s+[^,]+)',
            r',\s*([A-Z][^,]+(?:Journal|Conference|Symposium|Workshop|Trans\.|Review)[^,]*)',
            r'\"[^\"]+\",\s*([^,]+),',  # Text after title in quotes
        ]
        for pattern in patterns:
            match = re.search(pattern, ref_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _process_figures(self, parsed_doc) -> List[Figure]:
        """Process extracted images into figures with IEEE validation"""
        figures = []
        seen_paths = set()  # Track image paths to prevent duplicates
        seen_sizes = set()   # Track image sizes to catch duplicate page screenshots
        seen_captions = {}  # Track captions to prevent duplicates
        
        for idx, img in enumerate(parsed_doc.images):
            # === DEDUPLICATION ===
            # Skip duplicate images based on file path
            img_path = img.filename or ""
            if img_path and img_path in seen_paths:
                continue
            seen_paths.add(img_path)
            
            # Skip duplicate images based on size (catches page duplicates)
            # Pages captured as images often have identical dimensions
            img_size = (getattr(img, 'width', 0), getattr(img, 'height', 0))
            if img_size[0] > 400 and img_size[1] > 400:  # Only check large images (likely page captures)
                if img_size in seen_sizes:
                    continue  # Skip this duplicate page image
            seen_sizes.add(img_size)
            
            # === CLEAN CAPTION ===
            caption = img.caption or ""
            caption = self._clean_figure_caption(caption, idx + 1)
            
            # === ENSURE ALL FIGURES HAVE VALID CAPTIONS ===
            # If caption is empty or too short, generate a descriptive one
            if not caption or len(caption.strip()) < 5:
                # Try to extract caption from nearby text in the document
                caption = self._find_figure_caption(parsed_doc, idx + 1, img_path)
            
            # If still no caption, generate a meaningful default
            if not caption or len(caption.strip()) < 5:
                caption = f"Illustration showing the system architecture and workflow (Part {len(figures) + 1})."
            
            # === DEDUPLICATE CAPTIONS ===
            # If we've seen this exact caption before, it's likely an issue.
            # Append distinctive text to make it unique and valid.
            if caption:
                clean_cap_key = caption.lower().strip()
                if clean_cap_key in seen_captions:
                    count = seen_captions[clean_cap_key]
                    seen_captions[clean_cap_key] += 1
                    # Modify the caption to be unique
                    if caption.endswith('.'):
                        caption = caption[:-1] + f" (Part {count + 1})."
                    else:
                        caption = f"{caption} (Part {count + 1})."
                else:
                    seen_captions[clean_cap_key] = 1
            
            figures.append(Figure(
                number=len(figures) + 1,  # Sequential numbering after dedup
                image_path=img_path,
                caption=caption,
                width=img.width,
                height=img.height
            ))
        
        return figures
    
    def _find_figure_caption(self, parsed_doc, fig_num: int, img_path: str) -> str:
        """Try to find a figure caption from the document text"""
        full_text = parsed_doc.full_text
        
        # Look for patterns like "Fig. X. Caption text" or "Figure X: Caption"
        patterns = [
            rf'Fig\.?\s*{fig_num}\.?\s*[-:]?\s*([^.\n]+(?:\.[^.\n]+)?)',
            rf'Figure\s*{fig_num}\.?\s*[-:]?\s*([^.\n]+(?:\.[^.\n]+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                if len(caption) > 10:  # Reasonable caption length
                    return caption
        
        return ""
    
    def _clean_figure_caption(self, caption: str, fig_num: int) -> str:
        """Clean figure caption for IEEE format"""
        if not caption:
            return ""
        
        # === REMOVE REDUNDANT PREFIXES ===
        # Remove "Fig. 1. Fig. 1." repeated patterns
        caption = re.sub(r'(Figure|Fig\.?)\s*\d+\.?\s*[-:]?\s*', '', caption, flags=re.IGNORECASE)
        # Run again to catch doubles
        caption = re.sub(r'(Figure|Fig\.?)\s*\d+\.?\s*[-:]?\s*', '', caption, flags=re.IGNORECASE)
        
        # Remove "Figure" word if it starts the caption without number
        caption = re.sub(r'^(Figure|Fig\.?)\s*[-:]?\s*', '', caption, flags=re.IGNORECASE)
        
        # === REMOVE PLACEHOLDERS ===
        placeholder_patterns = [
            r'From this qualitative study,?\s*it is evident that\s*(it is)?\.?',
            r'it is evident that it is\.?',
            r'Placeholder\s*(text|caption)?\.?',
            r'Caption\s*here\.?',
            r'Insert\s*caption\.?',
            r'Figure\s*description\.?',
            r'Figure\s*illustration\.?',
        ]
        for pattern in placeholder_patterns:
            caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
        
        # Clean up whitespace
        caption = ' '.join(caption.split()).strip()
        
        # Remove leading/trailing punctuation artifacts
        caption = caption.strip('.,;:-–—')
        
        # Ensure caption is a complete description (ends with proper punctuation)
        if caption and not caption.endswith(('.', '!', '?')):
            # Check if it looks like an incomplete sentence
            if len(caption.split()) > 3:  # More than a few words
                caption = caption + '.'
        
        return caption
    
    def _process_tables(self, parsed_doc) -> List[Table]:
        """Process extracted tables"""
        tables = []
        
        for idx, tbl in enumerate(parsed_doc.tables):
            tables.append(Table(
                number=idx + 1,
                rows=tbl.rows,
                caption=tbl.caption or f"Table {idx + 1}"
            ))
        
        return tables
    
    def _extract_equations(self, parsed_doc) -> List[str]:
        """Extract mathematical equations using EquationHandler"""
        try:
            from parsers.equation_handler import EquationHandler
            
            handler = EquationHandler()
            full_text = parsed_doc.full_text
            
            # Use equation handler for comprehensive detection
            equations = handler.extract_equations(full_text)
            
            # Also check section content for equations
            for section in getattr(parsed_doc, 'sections', []):
                if hasattr(section, 'content'):
                    section_equations = handler.extract_equations(section.content)
                    equations.extend(section_equations)
            
            # Remove duplicates and return LaTeX representations
            seen = set()
            unique_eqs = []
            for eq in equations:
                if eq.latex not in seen:
                    seen.add(eq.latex)
                    unique_eqs.append(eq.latex)
            
            return unique_eqs[:30]  # Limit to 30 equations
            
        except ImportError:
            # Fallback to basic detection
            equations = []
            full_text = parsed_doc.full_text
            
            patterns = [
                r'\$\$([^$]+)\$\$',
                r'\$([^$]+)\$',
                r'\\begin\{equation\}(.+?)\\end\{equation\}',
                r'([a-zA-Z]\s*=\s*[^,\n]+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.DOTALL)
                equations.extend(matches)
            
            return equations[:20]

