"""
IEEE Conference Author Block Formatter
Formats author information according to IEEE conference paper standards.
Ensures compliance with IEEE PDF eXpress requirements.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Author:
    """Represents an author with IEEE-compliant formatting"""
    name: str
    affiliation: str = ""
    department: str = ""
    institution: str = ""
    city: str = ""
    country: str = ""
    email: str = ""
    affiliation_number: int = 0


@dataclass
class AuthorBlock:
    """Represents the complete IEEE author block"""
    authors: List[Author] = field(default_factory=list)
    affiliations: List[Dict[str, str]] = field(default_factory=list)


class IEEEAuthorParser:
    """
    Parses and formats author information according to IEEE conference standards.
    
    IEEE Author Block Rules:
    1. Author names - full names, no titles/degrees
    2. Affiliations - Department, Institution, City, Country
    3. Emails - lowercase, after affiliations
    4. Multiple authors - superscript numerals for mapping
    """
    
    # Patterns for titles/designations to remove (order matters - longer patterns first)
    TITLE_PATTERNS = [
        r'\bAsst\.?\s*Professor\s*',  # Asst. Professor (before Prof)
        r'\bAssoc\.?\s*Professor\s*',  # Assoc. Professor (before Prof)
        r'\bAssistant\s+Professor\s*',  # Assistant Professor (before Prof)
        r'\bAssociate\s+Professor\s*',  # Associate Professor (before Prof)
        r'\bProfessor\s*',  # Professor (full word)
        r'\bProf\.\s*',  # Prof. (abbreviated)
        r'\bDr\.\s*',  # Dr.
        r'\bMr\.\s*',  # Mr.
        r'\bMrs\.\s*',  # Mrs.
        r'\bMs\.\s*',  # Ms.
        r'\bMiss\.\s*',  # Miss.
        r'\bSir\s+',  # Sir
        r'\bMadam\s+',  # Madam
        r'\bResearch\s+Scholar\s*',  # Research Scholar
    ]
    
    # Patterns for degrees to remove (at end of name)
    DEGREE_PATTERNS = [
        r',?\s+Ph\.?D\.?\s*$',
        r',?\s+M\.?Tech\.?\s*$',
        r',?\s+B\.?Tech\.?\s*$',
        r',?\s+M\.?S\s*$',
        r',?\s+B\.?S\s*$',
        r',?\s+M\.?Sc\.?\s*$',
        r',?\s+B\.?Sc\.?\s*$',
        r',?\s+MBA\s*$',
        r',?\s+MCA\s*$',
        r',?\s+BCA\s*$',
        r',?\s+M\.?E\s*$',
        r',?\s+B\.?E\s*$',
        r',?\s+MBBS\s*$',
        r',?\s+MD\s*$',
        r',?\s+M\.?Phil\.?\s*$',
        r'\s*\([^)]*\)\s*$',  # Remove anything in parentheses at end
    ]
    
    # Common institution keywords
    INSTITUTION_KEYWORDS = [
        'university', 'college', 'institute', 'institution', 'school',
        'academy', 'polytechnic', 'iit', 'nit', 'iiit', 'bits'
    ]
    
    # Common department keywords
    DEPARTMENT_KEYWORDS = [
        'department', 'dept', 'school of', 'faculty of', 'division of',
        'center for', 'centre for', 'lab', 'laboratory'
    ]
    
    def __init__(self):
        self.authors = []
        self.affiliations = []
    
    def parse_authors(self, raw_authors: List[str], raw_affiliations: List[str] = None) -> AuthorBlock:
        """
        Parse raw author and affiliation data into IEEE-compliant format.
        
        Args:
            raw_authors: List of author names/data as extracted
            raw_affiliations: Optional list of affiliation strings
            
        Returns:
            AuthorBlock with cleaned, IEEE-compliant data
        """
        author_block = AuthorBlock()
        affiliation_map = {}  # Map unique affiliations to numbers
        
        for raw_author in raw_authors:
            author = self._parse_single_author(raw_author)
            
            # Assign affiliation number
            if author.affiliation:
                aff_key = self._normalize_affiliation_key(author.affiliation)
                if aff_key not in affiliation_map:
                    affiliation_map[aff_key] = len(affiliation_map) + 1
                author.affiliation_number = affiliation_map[aff_key]
            
            author_block.authors.append(author)
        
        # Process standalone affiliations if provided
        if raw_affiliations:
            for aff in raw_affiliations:
                parsed_aff = self._parse_affiliation(aff)
                aff_key = self._normalize_affiliation_key(aff)
                if aff_key not in affiliation_map:
                    affiliation_map[aff_key] = len(affiliation_map) + 1
                parsed_aff['number'] = affiliation_map[aff_key]
                author_block.affiliations.append(parsed_aff)
        
        # Build affiliations from authors if not provided separately
        if not author_block.affiliations:
            for aff_key, aff_num in sorted(affiliation_map.items(), key=lambda x: x[1]):
                # Find an author with this affiliation to get details
                for author in author_block.authors:
                    if author.affiliation_number == aff_num:
                        author_block.affiliations.append({
                            'number': aff_num,
                            'department': author.department,
                            'institution': author.institution,
                            'city': author.city,
                            'country': author.country,
                            'full': author.affiliation
                        })
                        break
        
        return author_block
    
    def _parse_single_author(self, raw: str) -> Author:
        """Parse a single author entry"""
        author = Author(name="")
        
        if not raw:
            return author
        
        # Split by common delimiters
        parts = re.split(r'[,\n]', raw)
        
        # First part is usually the name
        name = parts[0].strip() if parts else raw.strip()
        name = self._clean_author_name(name)
        author.name = name
        
        # Rest might contain affiliation, email
        for part in parts[1:]:
            part = part.strip()
            if self._is_email(part):
                author.email = self._clean_email(part)
            elif part:
                # Accumulate as affiliation
                if author.affiliation:
                    author.affiliation += ", " + part
                else:
                    author.affiliation = part
        
        # Parse affiliation into components
        if author.affiliation:
            aff_parts = self._parse_affiliation(author.affiliation)
            author.department = aff_parts.get('department', '')
            author.institution = aff_parts.get('institution', '')
            author.city = aff_parts.get('city', '')
            author.country = aff_parts.get('country', '')
        
        return author
    
    def _clean_author_name(self, name: str) -> str:
        """
        Clean author name by removing titles, degrees, and designations.
        
        IEEE Rule: Use full names, no titles or degrees.
        """
        if not name:
            return ""
        
        cleaned = name.strip()
        
        # Remove titles/designations - apply multiple times to catch combinations
        prev_cleaned = ""
        while prev_cleaned != cleaned:
            prev_cleaned = cleaned
            for pattern in self.TITLE_PATTERNS:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            cleaned = ' '.join(cleaned.split())  # Normalize spaces
        
        # Remove degrees - apply multiple times
        prev_cleaned = ""
        while prev_cleaned != cleaned:
            prev_cleaned = cleaned
            for pattern in self.DEGREE_PATTERNS:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            cleaned = ' '.join(cleaned.split())
        
        # Remove superscript numbers (affiliation markers)
        cleaned = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰\*†‡§¶]+', '', cleaned)
        cleaned = re.sub(r'\s*\d+\s*$', '', cleaned)  # Trailing numbers
        
        # Remove commas that may be left over
        cleaned = cleaned.strip(',').strip()
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Ensure proper capitalization
        cleaned = self._proper_name_case(cleaned)
        
        return cleaned
    
    def _proper_name_case(self, name: str) -> str:
        """Convert name to proper title case"""
        if not name:
            return ""
        
        # Handle special cases like "van", "de", "von"
        words = name.split()
        result = []
        lowercase_words = {'van', 'de', 'von', 'der', 'den', 'la', 'le', 'du'}
        
        for i, word in enumerate(words):
            if i == 0:
                result.append(word.capitalize())
            elif word.lower() in lowercase_words:
                result.append(word.lower())
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _parse_affiliation(self, aff: str) -> Dict[str, str]:
        """
        Parse affiliation string into components.
        
        IEEE Rule: Department, Institution, City, Country
        """
        result = {
            'department': '',
            'institution': '',
            'city': '',
            'country': '',
            'full': aff
        }
        
        if not aff:
            return result
        
        # Split by common delimiters
        parts = [p.strip() for p in re.split(r'[,\n]', aff) if p.strip()]
        
        # Remove postal codes
        parts = [self._remove_postal_code(p) for p in parts]
        parts = [p for p in parts if p]  # Remove empty
        
        # Try to identify each part
        for part in parts:
            part_lower = part.lower()
            
            # Check for department
            if any(kw in part_lower for kw in self.DEPARTMENT_KEYWORDS):
                if not result['department']:
                    result['department'] = self._clean_affiliation_part(part)
            
            # Check for institution
            elif any(kw in part_lower for kw in self.INSTITUTION_KEYWORDS):
                if not result['institution']:
                    result['institution'] = self._clean_affiliation_part(part)
            
            # Check for country (common countries)
            elif self._is_country(part):
                result['country'] = self._clean_affiliation_part(part)
            
            # Likely city if short and not matched
            elif len(part) < 30 and not result['city']:
                result['city'] = self._clean_affiliation_part(part)
        
        return result
    
    def _clean_affiliation_part(self, part: str) -> str:
        """Clean an affiliation component"""
        # Remove job titles
        cleaned = re.sub(r'\b(Professor|Asst|Assoc|Senior|Junior|Head|Dean|Director)\b\.?\s*', 
                        '', part, flags=re.IGNORECASE)
        # Remove postal codes
        cleaned = self._remove_postal_code(cleaned)
        # Clean whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def _remove_postal_code(self, text: str) -> str:
        """Remove postal/zip codes from text"""
        # Common postal code patterns
        patterns = [
            r'\b\d{5,6}\b',  # 5-6 digit codes
            r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK postcodes
            r'\b\d{3}-\d{4}\b',  # Japanese postal
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned)
        return cleaned.strip()
    
    def _is_email(self, text: str) -> bool:
        """Check if text looks like an email"""
        return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', text.strip()))
    
    def _clean_email(self, email: str) -> str:
        """Clean and lowercase email"""
        return email.strip().lower()
    
    def _is_country(self, text: str) -> bool:
        """Check if text is a country name"""
        countries = [
            'india', 'usa', 'united states', 'uk', 'united kingdom', 'china',
            'japan', 'germany', 'france', 'canada', 'australia', 'brazil',
            'russia', 'south korea', 'italy', 'spain', 'netherlands', 'sweden',
            'switzerland', 'singapore', 'malaysia', 'indonesia', 'thailand',
            'vietnam', 'philippines', 'pakistan', 'bangladesh', 'sri lanka',
            'nepal', 'iran', 'iraq', 'saudi arabia', 'uae', 'egypt', 'nigeria',
            'south africa', 'kenya', 'mexico', 'argentina', 'chile', 'colombia',
            'peru', 'poland', 'czech republic', 'austria', 'belgium', 'denmark',
            'finland', 'norway', 'ireland', 'portugal', 'greece', 'turkey',
            'israel', 'new zealand', 'taiwan', 'hong kong'
        ]
        return text.strip().lower() in countries
    
    def _normalize_affiliation_key(self, aff: str) -> str:
        """Create a normalized key for affiliation matching"""
        return re.sub(r'\s+', ' ', aff.lower().strip())
    
    def format_latex_authors(self, author_block: AuthorBlock) -> str:
        """
        Generate IEEE-compliant LaTeX author block.
        
        Returns LaTeX code for the \author{} command.
        """
        if not author_block.authors:
            return ""
        
        lines = []
        
        # Check if we have multiple affiliations
        unique_affiliations = set(a.affiliation_number for a in author_block.authors if a.affiliation_number)
        use_superscripts = len(unique_affiliations) > 1
        
        # Format each author
        author_lines = []
        for author in author_block.authors:
            author_str = author.name
            if use_superscripts and author.affiliation_number:
                author_str += f"$^{{{author.affiliation_number}}}$"
            author_lines.append(author_str)
        
        # Join authors appropriately
        if len(author_lines) <= 3:
            lines.append(" \\and ".join(author_lines))
        else:
            # Multiple authors - use IEEEauthorblockN
            for author in author_block.authors:
                lines.append(f"\\IEEEauthorblockN{{{author.name}}}")
                if author.affiliation:
                    aff_str = self._format_affiliation_latex(author)
                    lines.append(f"\\IEEEauthorblockA{{{aff_str}}}")
        
        # Add affiliations section if using superscripts
        if use_superscripts and author_block.affiliations:
            lines.append("")
            for aff in author_block.affiliations:
                aff_str = self._build_affiliation_string(aff)
                lines.append(f"$^{{{aff['number']}}}${aff_str}")
        
        # Add emails
        emails = [a.email for a in author_block.authors if a.email]
        if emails:
            lines.append("")
            lines.append("\\texttt{" + ", ".join(emails) + "}")
        
        return "\n".join(lines)
    
    def _format_affiliation_latex(self, author: Author) -> str:
        """Format a single author's affiliation for LaTeX"""
        parts = []
        if author.department:
            parts.append(author.department)
        if author.institution:
            parts.append(author.institution)
        location = []
        if author.city:
            location.append(author.city)
        if author.country:
            location.append(author.country)
        if location:
            parts.append(", ".join(location))
        if author.email:
            parts.append(f"\\texttt{{{author.email}}}")
        return "\\\\ ".join(parts)
    
    def _build_affiliation_string(self, aff: Dict[str, str]) -> str:
        """Build affiliation string from components"""
        parts = []
        if aff.get('department'):
            parts.append(aff['department'])
        if aff.get('institution'):
            parts.append(aff['institution'])
        location = []
        if aff.get('city'):
            location.append(aff['city'])
        if aff.get('country'):
            location.append(aff['country'])
        if location:
            parts.append(", ".join(location))
        return ", ".join(parts) if parts else aff.get('full', '')
    
    def validate_author_block(self, author_block: AuthorBlock) -> List[str]:
        """
        Validate the author block against IEEE rules.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        for i, author in enumerate(author_block.authors, 1):
            # Check name is present
            if not author.name or len(author.name) < 2:
                errors.append(f"Author {i}: Missing or invalid name")
            
            # Check for remaining titles
            for pattern in self.TITLE_PATTERNS:
                if re.search(pattern, author.name, re.IGNORECASE):
                    errors.append(f"Author {i}: Name contains title/designation")
            
            # Check for remaining degrees
            for pattern in self.DEGREE_PATTERNS:
                if re.search(pattern, author.name, re.IGNORECASE):
                    errors.append(f"Author {i}: Name contains degree")
            
            # Check email format if present
            if author.email and not self._is_email(author.email):
                errors.append(f"Author {i}: Invalid email format")
            
            # Check email is lowercase
            if author.email and author.email != author.email.lower():
                errors.append(f"Author {i}: Email should be lowercase")
        
        return errors


# Convenience function for integration
def format_ieee_authors(raw_authors: List[str], raw_affiliations: List[str] = None) -> Tuple[str, List[str]]:
    """
    Format author data into IEEE-compliant LaTeX.
    
    Args:
        raw_authors: List of raw author strings
        raw_affiliations: Optional list of affiliation strings
        
    Returns:
        Tuple of (LaTeX string, list of validation errors)
    """
    parser = IEEEAuthorParser()
    author_block = parser.parse_authors(raw_authors, raw_affiliations)
    latex = parser.format_latex_authors(author_block)
    errors = parser.validate_author_block(author_block)
    return latex, errors
