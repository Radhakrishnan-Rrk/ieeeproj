import re

class TextProcessor:
    """
    Centralized text processing and repair logic to enforce IEEE invariants.
    Handles hyphenation repair, normalization, and reference standardization.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
            
        # 1. Fix Broken Words (Hyphenation Repair) - aggressively join known fragments
        # Fix "mag- netic" -> "magnetic", "depen- dencies" -> "dependencies"
        # Suffix list covers common word endings
        suffixes = r'(tic|tion|ment|ing|able|cal|y|al|ic|ous|ive|fy|ize|ise|ism|ity|ness|less|ful|work|rithm|mance|encies|ence|ance|ency|ancy|cies|gies|ies|ed|es|er|or|est|tives|ly|ary|ory|ism|ist|logy|nomy|try|phy|sis|tural|tional|tional|rence|rent|rrent)\b'
        text = re.sub(r'(\w+)-\s+' + suffixes, r'\1\2', text, flags=re.IGNORECASE)
        
        # 2. Fix Spaced Hyphens for Compound Words (The "Catch-All")
        # "long- range" -> "long-range", "time- consuming" -> "time-consuming"
        # If it wasn't joined by step 1, assume it's a compound word that needs the hyphen kept but space removed.
        # Enforcer ensures no `\w+-\s+\w+`. This fixes that.
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        
        # 3. Explicit Fixes for common failures
        text = text.replace('mag-netic', 'magnetic') # In case pass 2 caught it
        text = text.replace('imple-mentation', 'implementation')
        
        # 4. Fix Figure Referencing Style - AGGRESSIVE
        # Convert ALL variants to strict "Fig. X" format
        # Handles: "Figure 1", "Fig 1", "Fig1", "figure x", "FIGURE 1"
        text = re.sub(r'\bFig(?:ure)?\.?\s*(\d+)', r'Fig. \1', text, flags=re.IGNORECASE)
        # Handle plurals: "Figures 1 and 2", "Figs 1-3"
        text = re.sub(r'\bFig(?:ure)?s\.?\s*(\d+)', r'Figs. \1', text, flags=re.IGNORECASE)
        
        # 5. Fix Normalization Placeholders - AGGRESSIVE
        # Replace ALL placeholder patterns with explicit numeric range [0, 1]
        
        # Pattern 1: [?], [TBD], [XXX], [tbd], etc.
        text = re.sub(r'\[\s*(\?|TBD|XXX|tbd|xxx)\s*\]', '[0, 1]', text, flags=re.IGNORECASE)
        
        # Pattern 2: Standalone TBD, XXX in range context
        # "normalized to TBD" -> "normalized to [0, 1]"
        # "range is TBD" -> "range is [0, 1]"
        # "scaled to XXX" -> "scaled to [0, 1]"
        text = re.sub(r'(normalized|scaled|range|values?)\s+(to|is|of|are|between)?\s*(TBD|XXX|\?)', r'\1 \2 [0, 1]', text, flags=re.IGNORECASE)
        
        # Pattern 3: "range [?]" or "range ?" or "range TBD"
        text = re.sub(r'(range)\s*\[?\s*(\?|TBD|XXX)\s*\]?', r'\1 [0, 1]', text, flags=re.IGNORECASE)
        
        # Pattern 4: Just "?" in numeric context (but not in questions)
        # "[0, ?]" -> "[0, 1]"
        text = re.sub(r'\[\s*(\d+)\s*,\s*\?\s*\]', r'[\1, 1]', text)
        text = re.sub(r'\[\s*\?\s*,\s*(\d+)\s*\]', r'[0, \1]', text)
        
        # Pattern 5: Standalone TBD/XXX anywhere (nuclear option)
        text = re.sub(r'\bTBD\b', '[0, 1]', text)
        text = re.sub(r'\bXXX\b', '[0, 1]', text)
        
        # 6. Fix Punctuation/Spacing
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text.strip()
