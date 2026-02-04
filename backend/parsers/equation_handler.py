"""
Equation Handler Module
Detects, parses, and formats mathematical equations for IEEE LaTeX output

This module handles:
- LaTeX equation detection ($...$, $$...$$, \begin{equation}...)
- Plain text equation parsing (y = mx + b, E = mc²)
- Subscript/superscript notation (x_1, x^2, H₂O)
- Greek letters (alpha, beta, gamma → α, β, γ)
- Mathematical operators and symbols
- Fraction detection (a/b → \frac{a}{b})
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Equation:
    """Represents a detected mathematical equation"""
    original: str          # Original text
    latex: str             # Converted LaTeX
    is_display: bool       # True for display equations, False for inline
    confidence: float      # Detection confidence (0-1)
    equation_type: str     # Type: 'latex', 'assignment', 'formula', 'expression'


class EquationHandler:
    """
    Handles detection, parsing, and formatting of mathematical equations
    for IEEE-formatted LaTeX output
    """
    
    # Greek letter mappings (text → LaTeX)
    GREEK_LETTERS = {
        'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma', 'delta': r'\delta',
        'epsilon': r'\epsilon', 'zeta': r'\zeta', 'eta': r'\eta', 'theta': r'\theta',
        'iota': r'\iota', 'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
        'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi', 'rho': r'\rho',
        'sigma': r'\sigma', 'tau': r'\tau', 'upsilon': r'\upsilon', 'phi': r'\phi',
        'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
        # Capital Greek
        'Alpha': r'\Alpha', 'Beta': r'\Beta', 'Gamma': r'\Gamma', 'Delta': r'\Delta',
        'Theta': r'\Theta', 'Lambda': r'\Lambda', 'Pi': r'\Pi', 'Sigma': r'\Sigma',
        'Phi': r'\Phi', 'Psi': r'\Psi', 'Omega': r'\Omega',
        # Unicode Greek
        'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
        'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
        'λ': r'\lambda', 'μ': r'\mu', 'π': r'\pi', 'ρ': r'\rho',
        'σ': r'\sigma', 'τ': r'\tau', 'φ': r'\phi', 'ψ': r'\psi', 'ω': r'\omega',
        'Γ': r'\Gamma', 'Δ': r'\Delta', 'Θ': r'\Theta', 'Λ': r'\Lambda',
        'Σ': r'\Sigma', 'Φ': r'\Phi', 'Ψ': r'\Psi', 'Ω': r'\Omega',
    }
    
    # Unicode subscript/superscript mappings
    SUBSCRIPTS = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        'ₐ': 'a', 'ₑ': 'e', 'ₕ': 'h', 'ᵢ': 'i', 'ⱼ': 'j',
        'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₒ': 'o',
        'ₚ': 'p', 'ᵣ': 'r', 'ₛ': 's', 'ₜ': 't', 'ᵤ': 'u',
        'ᵥ': 'v', 'ₓ': 'x',
    }
    
    SUPERSCRIPTS = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        'ⁿ': 'n', 'ⁱ': 'i', '⁺': '+', '⁻': '-',
    }
    
    # Mathematical operators
    MATH_OPERATORS = {
        '×': r'\times', '÷': r'\div', '±': r'\pm', '∓': r'\mp',
        '≤': r'\leq', '≥': r'\geq', '≠': r'\neq', '≈': r'\approx',
        '∞': r'\infty', '∑': r'\sum', '∏': r'\prod', '∫': r'\int',
        '∂': r'\partial', '∇': r'\nabla', '√': r'\sqrt',
        '→': r'\rightarrow', '←': r'\leftarrow', '↔': r'\leftrightarrow',
        '⇒': r'\Rightarrow', '⇐': r'\Leftarrow', '⇔': r'\Leftrightarrow',
        '∈': r'\in', '∉': r'\notin', '⊂': r'\subset', '⊃': r'\supset',
        '∪': r'\cup', '∩': r'\cap', '∀': r'\forall', '∃': r'\exists',
        '·': r'\cdot', '°': r'^{\circ}',
    }
    
    # Common mathematical functions
    MATH_FUNCTIONS = [
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
        'log', 'ln', 'exp', 'sqrt', 'min', 'max',
        'lim', 'sup', 'inf', 'det', 'dim', 'ker',
        'arg', 'deg', 'gcd', 'lcm', 'mod',
        'softmax', 'sigmoid', 'relu', 'tanh',  # ML functions
    ]
    
    def __init__(self):
        self.equation_counter = 0
        self.equations_cache: List[Equation] = []
    
    def extract_equations(self, text: str) -> List[Equation]:
        """
        Extract all equations from text
        
        Args:
            text: Input text containing potential equations
            
        Returns:
            List of Equation objects
        """
        equations = []
        
        # 1. Extract existing LaTeX equations first
        equations.extend(self._extract_latex_equations(text))
        
        # 2. Extract assignment-style equations (x = y + z)
        equations.extend(self._extract_assignment_equations(text))
        
        # 3. Extract mathematical formulas
        equations.extend(self._extract_formulas(text))
        
        # Remove duplicates based on original text
        seen = set()
        unique_equations = []
        for eq in equations:
            if eq.original not in seen:
                seen.add(eq.original)
                unique_equations.append(eq)
        
        self.equations_cache = unique_equations
        return unique_equations
    
    def _extract_latex_equations(self, text: str) -> List[Equation]:
        """Extract existing LaTeX-formatted equations"""
        equations = []
        
        # Display math: $$...$$
        for match in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
            equations.append(Equation(
                original=match.group(0),
                latex=match.group(1).strip(),
                is_display=True,
                confidence=1.0,
                equation_type='latex'
            ))
        
        # Inline math: $...$
        for match in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text):
            eq_content = match.group(1).strip()
            if len(eq_content) > 1:  # Avoid single character matches
                equations.append(Equation(
                    original=match.group(0),
                    latex=eq_content,
                    is_display=False,
                    confidence=1.0,
                    equation_type='latex'
                ))
        
        # \begin{equation}...\end{equation}
        for match in re.finditer(r'\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}', text, re.DOTALL):
            equations.append(Equation(
                original=match.group(0),
                latex=match.group(1).strip(),
                is_display=True,
                confidence=1.0,
                equation_type='latex'
            ))
        
        # \[ ... \]
        for match in re.finditer(r'\\\[(.+?)\\\]', text, re.DOTALL):
            equations.append(Equation(
                original=match.group(0),
                latex=match.group(1).strip(),
                is_display=True,
                confidence=1.0,
                equation_type='latex'
            ))
        
        return equations
    
    def _extract_assignment_equations(self, text: str) -> List[Equation]:
        """Extract assignment-style equations like 'y = mx + b'"""
        equations = []
        
        # Much more restrictive pattern for equations
        # Only match short, clearly mathematical expressions
        patterns = [
            # Short variable assignment with math operators: L = a + b, E = mc²
            r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_+\-*/^()λαβγδεζηθικλμνξπρστυφχψω²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉\s]{1,50})(?=\s*[,.\n)]|$)',
            
            # Subscripted variable: L_{ce} = expression
            r'([A-Za-z]_\{[^}]+\})\s*=\s*([A-Za-z0-9_+\-*/^(){}\s]{1,50})(?=\s*[,.\n)]|$)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                lhs = match.group(1).strip()
                rhs = match.group(2).strip()
                
                # Skip if too short
                if len(rhs) < 2:
                    continue
                
                # Skip if RHS looks like prose (has too many long words)
                words = re.findall(r'\b[a-zA-Z]{5,}\b', rhs)
                if len(words) > 2:
                    continue
                
                # Skip if it looks like regular prose
                if self._is_prose(lhs + ' = ' + rhs):
                    continue
                
                # Must have some mathematical content
                math_chars = set('+-*/^()_{}λαβγδεζηθικλμνξπρστυφχψω0123456789²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉')
                if not any(c in math_chars for c in rhs):
                    # Check if it's a simple variable assignment like x = y
                    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', rhs):
                        continue
                
                full_eq = f"{lhs} = {rhs}"
                latex_eq = self._convert_to_latex(full_eq)
                
                # Determine if display or inline based on context
                is_display = self._should_be_display(match.group(0), text, match.start())
                
                equations.append(Equation(
                    original=full_eq,
                    latex=latex_eq,
                    is_display=is_display,
                    confidence=0.8,
                    equation_type='assignment'
                ))
        
        return equations
    
    def _extract_formulas(self, text: str) -> List[Equation]:
        """Extract mathematical formulas - disabled for now to prevent false positives"""
        # Return empty list - formula extraction was too aggressive
        # Only rely on explicit LaTeX and assignment equations
        return []
    
    def _is_prose(self, text: str) -> bool:
        """Check if text is likely regular prose rather than an equation"""
        # Count regular words vs mathematical symbols
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        math_symbols = re.findall(r'[+\-*/^=<>≤≥∑∏∫√]', text)
        
        # If many long words and few math symbols, probably prose
        if len(words) > 3 and len(math_symbols) < 2:
            return True
        
        # Common prose patterns
        prose_patterns = [
            r'\b(the|is|are|was|were|this|that|which|with|from|have|has)\b',
            r'\b(for|and|but|not|you|can|will|would|should)\b',
        ]
        
        for pattern in prose_patterns:
            if len(re.findall(pattern, text.lower())) > 2:
                return True
        
        return False
    
    def _should_be_display(self, eq_text: str, full_text: str, position: int) -> bool:
        """Determine if equation should be display or inline"""
        # Check if equation is on its own line
        before = full_text[max(0, position-50):position]
        after = full_text[position+len(eq_text):position+len(eq_text)+50]
        
        # If surrounded by newlines, make it display
        if before.rstrip().endswith('\n') or before.strip() == '':
            if after.lstrip().startswith('\n') or after.strip() == '':
                return True
        
        # Long equations should be display
        if len(eq_text) > 30:
            return True
        
        # Equations with fractions, sums, integrals should be display
        display_indicators = [r'\frac', r'\sum', r'\int', r'\prod', '∑', '∫', '∏']
        for indicator in display_indicators:
            if indicator in eq_text:
                return True
        
        return False
    
    def _convert_to_latex(self, text: str) -> str:
        """Convert plain text equation to proper LaTeX"""
        result = text
        
        # Convert Unicode subscripts to LaTeX
        for unicode_char, replacement in self.SUBSCRIPTS.items():
            if unicode_char in result:
                # Find the character before the subscript
                result = result.replace(unicode_char, f'_{{{replacement}}}')
        
        # Convert Unicode superscripts to LaTeX
        for unicode_char, replacement in self.SUPERSCRIPTS.items():
            if unicode_char in result:
                result = result.replace(unicode_char, f'^{{{replacement}}}')
        
        # Convert Greek letters
        for greek, latex in self.GREEK_LETTERS.items():
            # Word boundary for text Greek letters
            if len(greek) > 1:
                # Use lambda to avoid regex escape issues with backslashes
                result = re.sub(rf'\b{greek}\b', lambda m: latex, result)
            else:
                # Direct replacement for Unicode Greek
                result = result.replace(greek, latex)
        
        # Convert math operators
        for symbol, latex in self.MATH_OPERATORS.items():
            result = result.replace(symbol, latex)
        
        # Convert math functions to proper LaTeX
        for func in self.MATH_FUNCTIONS:
            result = re.sub(rf'\b{func}\b(?!\{{)', lambda m: f'\\{func}', result)
        
        # Convert simple fractions a/b to \frac{a}{b}
        def make_fraction(m):
            return f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}'
        result = re.sub(
            r'(?<![a-zA-Z0-9])([a-zA-Z0-9_{}\\]+)\s*/\s*([a-zA-Z0-9_{}\\]+)(?![a-zA-Z0-9])',
            make_fraction,
            result
        )
        
        # Convert x_n to x_{n} if not already
        result = re.sub(r'_([a-zA-Z0-9])(?![{}])', r'_{\1}', result)
        
        # Convert x^n to x^{n} if not already
        result = re.sub(r'\^([a-zA-Z0-9])(?![{}])', r'^{\1}', result)
        
        # Handle multiple subscripts/superscripts
        result = re.sub(r'_\{([^}]+)\}_\{([^}]+)\}', r'_{\1\2}', result)
        
        return result
    
    def format_for_latex(self, text: str, number_equations: bool = True) -> str:
        """
        Process text and format all equations for LaTeX output
        
        Args:
            text: Input text with equations
            number_equations: Whether to number display equations
            
        Returns:
            Text with equations formatted as LaTeX
        """
        result = text
        equations = self.extract_equations(text)
        
        # Sort by position (reverse) to replace from end to start
        # This prevents position shifts during replacement
        positioned_eqs = []
        for eq in equations:
            pos = result.find(eq.original)
            if pos >= 0:
                positioned_eqs.append((pos, eq))
        
        positioned_eqs.sort(key=lambda x: x[0], reverse=True)
        
        for pos, eq in positioned_eqs:
            if eq.is_display:
                if eq.equation_type == 'latex':
                    # Already LaTeX, just ensure proper environment
                    replacement = f'\\begin{{equation}}\n{eq.latex}\n\\end{{equation}}'
                else:
                    replacement = f'\\begin{{equation}}\n{eq.latex}\n\\end{{equation}}'
            else:
                # Inline equation
                replacement = f'${eq.latex}$'
            
            result = result[:pos] + replacement + result[pos + len(eq.original):]
        
        return result
    
    def detect_equation_lines(self, lines: List[str]) -> List[Tuple[int, str, bool]]:
        """
        Detect which lines contain equations
        
        Args:
            lines: List of text lines
            
        Returns:
            List of (line_index, equation_latex, is_display) tuples
        """
        equation_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like an equation
            if self._line_is_equation(line):
                latex = self._convert_to_latex(line)
                is_display = len(line) < 60 and '=' in line
                equation_lines.append((i, latex, is_display))
        
        return equation_lines
    
    def _line_is_equation(self, line: str) -> bool:
        """Check if a line is likely an equation"""
        # Must have some math content
        math_indicators = ['=', '+', '-', '*', '/', '^', '_', '(', ')', 
                          '∑', '∫', '∏', '√', '≤', '≥', '≠', '≈']
        
        has_math = any(ind in line for ind in math_indicators)
        if not has_math:
            return False
        
        # Should not be too prose-like
        if self._is_prose(line):
            return False
        
        # Should have variables or numbers
        has_vars = re.search(r'[a-zA-Z][_\^]?[a-zA-Z0-9]|[a-zA-Z]\s*=', line)
        has_numbers = re.search(r'\d', line)
        
        return has_vars or has_numbers
    
    def get_equation_summary(self) -> Dict:
        """Get summary of detected equations"""
        return {
            'total': len(self.equations_cache),
            'display': len([e for e in self.equations_cache if e.is_display]),
            'inline': len([e for e in self.equations_cache if not e.is_display]),
            'types': {
                'latex': len([e for e in self.equations_cache if e.equation_type == 'latex']),
                'assignment': len([e for e in self.equations_cache if e.equation_type == 'assignment']),
                'formula': len([e for e in self.equations_cache if e.equation_type == 'formula']),
                'expression': len([e for e in self.equations_cache if e.equation_type == 'expression']),
            }
        }
