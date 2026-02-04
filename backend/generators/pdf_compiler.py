"""
PDF Compiler Module
Compiles LaTeX documents to PDF using pdflatex with proper IEEE formatting
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class PDFCompiler:
    """
    Compiles LaTeX documents to PDF using pdflatex
    Includes IEEEtran class for proper IEEE conference formatting
    """
    
    def __init__(self, output_dir: str = "output", template_dir: str = "templates"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent.parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        self.pdflatex_available = self._check_pdflatex()
        
        # Ensure IEEEtran.cls exists
        self._ensure_ieee_template()
    
    def _check_pdflatex(self) -> bool:
        """Check if pdflatex is available on the system"""
        # Check standard paths for pdflatex
        pdflatex_paths = [
            'pdflatex',  # System PATH
            '/Library/TeX/texbin/pdflatex',  # MacTeX standard path
            '/usr/local/texlive/2025/bin/universal-darwin/pdflatex',  # TeX Live 2025
            '/usr/local/texlive/2024/bin/universal-darwin/pdflatex',  # TeX Live 2024
            '/usr/texbin/pdflatex',  # Alternative macOS path
        ]
        
        for pdflatex_path in pdflatex_paths:
            try:
                result = subprocess.run(
                    [pdflatex_path, '--version'],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self.pdflatex_path = pdflatex_path
                    print(f"Found pdflatex at: {pdflatex_path}")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        self.pdflatex_path = None
        print("pdflatex not found")
        return False
    
    def _ensure_ieee_template(self):
        """Ensure IEEEtran.cls is available"""
        ieee_cls = self.template_dir / "IEEEtran.cls"
        if not ieee_cls.exists():
            # Create a minimal IEEEtran placeholder or download
            # For now, we'll rely on system-installed IEEEtran or use article fallback
            pass
    
    def compile(self, latex_content: str, output_path: str, job_id: str) -> bool:
        """
        Compile LaTeX content to PDF
        
        Args:
            latex_content: Complete LaTeX document as string
            output_path: Path where the PDF should be saved
            job_id: Job ID for temporary files
            
        Returns:
            True if compilation succeeded, False otherwise
        """
        if self.pdflatex_available:
            success = self._compile_with_pdflatex(latex_content, output_path, job_id)
            if success:
                return True
            
            # If IEEE compilation fails, try standalone version
            print("IEEE compilation failed, trying standalone format...")
            from generators.latex_generator import LaTeXGenerator
            from extractors.content_analyzer import StructuredPaper
            
            # Try to compile with article class fallback
            standalone_content = self._convert_to_standalone(latex_content)
            return self._compile_with_pdflatex(standalone_content, output_path, job_id)
        else:
            # Fallback: generate PDF with reportlab
            return self._compile_with_reportlab(latex_content, output_path, job_id)
    
    def _convert_to_standalone(self, latex_content: str) -> str:
        """Convert IEEEtran document to standalone article format"""
        import re
        
        # Replace IEEEtran with article class
        content = latex_content.replace(
            r'\documentclass[conference]{IEEEtran}',
            r'\documentclass[10pt,twocolumn]{article}'
        )
        
        # Add geometry package for IEEE-like margins
        content = content.replace(
            r'\begin{document}',
            r'''\usepackage[top=0.75in, bottom=1in, left=0.625in, right=0.625in]{geometry}
\usepackage{times}
\setlength{\columnsep}{0.25in}
\begin{document}'''
        )
        
        # Remove IEEEtran-specific commands
        content = re.sub(r'\\IEEEauthorblockN\{([^}]+)\}', r'\1', content)
        content = re.sub(r'\\IEEEauthorblockA\{([^}]+)\}', r'\\\\\\textit{\1}', content)
        content = content.replace(r'\begin{IEEEkeywords}', r'\noindent\textbf{Keywords:} ')
        content = content.replace(r'\end{IEEEkeywords}', r'\\\vspace{1em}')
        
        return content
    
    def _compile_with_pdflatex(self, latex_content: str, output_path: str, job_id: str) -> bool:
        """Compile using pdflatex"""
        # Create temporary directory for compilation
        temp_dir = Path(tempfile.mkdtemp(prefix=f"ieee_{job_id}_"))
        tex_file = temp_dir / "paper.tex"
        
        try:
            # Write LaTeX content to file
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Also save a copy for debugging
            debug_tex = self.output_dir / f"{job_id}_debug.tex"
            with open(debug_tex, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Run pdflatex (twice for proper references and citations)
            pdflatex_cmd = getattr(self, 'pdflatex_path', 'pdflatex') or 'pdflatex'
            for run in range(2):
                result = subprocess.run(
                    [
                        pdflatex_cmd,
                        '-interaction=nonstopmode',
                        '-file-line-error',
                        '-output-directory', str(temp_dir),
                        str(tex_file)
                    ],
                    capture_output=True,
                    timeout=120,
                    cwd=str(temp_dir)
                )
                
                # Log output for debugging
                if result.returncode != 0:
                    log_file = temp_dir / "paper.log"
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                        # Save error log
                        error_log = self.output_dir / f"{job_id}_error.log"
                        with open(error_log, 'w') as f:
                            f.write(log_content)
                        print(f"LaTeX errors saved to: {error_log}")
            
            # Check if PDF was generated
            pdf_path = temp_dir / "paper.pdf"
            if pdf_path.exists():
                shutil.move(str(pdf_path), output_path)
                print(f"PDF generated successfully: {output_path}")
                return True
            else:
                print("PDF file not generated")
                return False
                
        except subprocess.TimeoutExpired:
            print("pdflatex timed out")
            return False
        except Exception as e:
            print(f"Compilation error: {e}")
            return False
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def _compile_with_reportlab(self, latex_content: str, output_path: str, job_id: str) -> bool:
        """
        Fallback PDF generation using reportlab
        Creates a proper two-column IEEE-like layout
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table as RLTable,
                TableStyle, Frame, PageTemplate, BaseDocTemplate, FrameBreak
            )
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
            from reportlab.lib import colors
            
            # Parse LaTeX content
            content = self._parse_latex_content(latex_content)
            
            # Create two-column document
            doc = self._create_two_column_doc(output_path)
            
            # Build story (content)
            story = self._build_story(content)
            
            # Build PDF
            doc.build(story)
            
            return True
            
        except ImportError as e:
            print(f"reportlab not available: {e}")
            return self._create_latex_only(latex_content, output_path)
        except Exception as e:
            print(f"Reportlab PDF generation error: {e}")
            return False
    
    def _create_two_column_doc(self, output_path: str):
        """Create a two-column document template"""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
        
        # Page dimensions
        page_width, page_height = letter
        margin_top = 0.75 * inch
        margin_bottom = 1 * inch
        margin_left = 0.625 * inch
        margin_right = 0.625 * inch
        column_gap = 0.25 * inch
        
        # Calculate column width
        usable_width = page_width - margin_left - margin_right - column_gap
        column_width = usable_width / 2
        
        # Create frames for two columns
        frame1 = Frame(
            margin_left,
            margin_bottom,
            column_width,
            page_height - margin_top - margin_bottom,
            id='col1'
        )
        
        frame2 = Frame(
            margin_left + column_width + column_gap,
            margin_bottom,
            column_width,
            page_height - margin_top - margin_bottom,
            id='col2'
        )
        
        # Create document with two-column template
        doc = BaseDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=margin_left,
            rightMargin=margin_right,
            topMargin=margin_top,
            bottomMargin=margin_bottom
        )
        
        # Add page templates
        doc.addPageTemplates([
            PageTemplate(id='TwoColumn', frames=[frame1, frame2])
        ])
        
        return doc
    
    def _build_story(self, content: dict):
        """Build the document story from parsed content"""
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import Paragraph, Spacer, Table as RLTable, TableStyle, FrameBreak
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.lib import colors
        
        styles = getSampleStyleSheet()
        
        # Custom styles for IEEE format
        title_style = ParagraphStyle(
            'IEEETitle',
            parent=styles['Title'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName='Times-Bold'
        )
        
        author_style = ParagraphStyle(
            'IEEEAuthor',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName='Times-Roman'
        )
        
        abstract_style = ParagraphStyle(
            'IEEEAbstract',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            fontName='Times-Italic',
            leftIndent=20,
            rightIndent=20
        )
        
        heading_style = ParagraphStyle(
            'IEEEHeading',
            parent=styles['Heading1'],
            fontSize=10,
            fontName='Times-Bold',
            spaceBefore=12,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            'IEEEBody',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            fontName='Times-Roman',
            spaceAfter=6,
            firstLineIndent=15
        )
        
        story = []
        
        # Title
        if content.get('title'):
            story.append(Paragraph(content['title'], title_style))
            story.append(Spacer(1, 12))
        
        # Authors
        for author in content.get('authors', []):
            story.append(Paragraph(author, author_style))
        story.append(Spacer(1, 12))
        
        # Abstract
        if content.get('abstract'):
            story.append(Paragraph('<b><i>Abstract</i></b>—' + content['abstract'], abstract_style))
            story.append(Spacer(1, 12))
        
        # Keywords
        if content.get('keywords'):
            story.append(Paragraph(f"<b><i>Index Terms</i></b>—{', '.join(content['keywords'])}", abstract_style))
            story.append(Spacer(1, 12))
        
        # Sections
        section_num = 0
        for section in content.get('sections', []):
            if section.get('level', 1) == 1:
                section_num += 1
                roman = self._int_to_roman(section_num)
                story.append(Paragraph(f'{roman}. {section["title"].upper()}', heading_style))
            else:
                story.append(Paragraph(f'<i>{section["title"]}</i>', heading_style))
            
            story.append(Paragraph(section.get('content', ''), body_style))
            story.append(Spacer(1, 6))
        
        # Tables
        for table in content.get('tables', []):
            story.append(self._create_reportlab_table(table))
            story.append(Spacer(1, 12))
        
        # References
        if content.get('references'):
            story.append(Paragraph('REFERENCES', heading_style))
            for i, ref in enumerate(content['references'], 1):
                story.append(Paragraph(f'[{i}] {ref}', body_style))
        
        return story
    
    def _create_reportlab_table(self, table_data: dict):
        """Create a reportlab table from table data"""
        from reportlab.platypus import Table as RLTable, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        rows = table_data.get('rows', [])
        if not rows:
            return Spacer(1, 0)
        
        # Create table
        t = RLTable(rows)
        
        # Style the table
        style = TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ])
        
        t.setStyle(style)
        
        return t
    
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
    
    def _parse_latex_content(self, latex_content: str) -> dict:
        """Parse LaTeX content for reportlab rendering"""
        import re
        
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'keywords': [],
            'sections': [],
            'tables': [],
            'references': []
        }
        
        # Extract title
        title_match = re.search(r'\\title\{([^}]+)\}', latex_content)
        if title_match:
            content['title'] = self._clean_latex(title_match.group(1))
        
        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_content, re.DOTALL)
        if abstract_match:
            content['abstract'] = self._clean_latex(abstract_match.group(1))
        
        # Extract keywords
        keywords_match = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', latex_content, re.DOTALL)
        if keywords_match:
            kw_text = self._clean_latex(keywords_match.group(1))
            content['keywords'] = [kw.strip() for kw in kw_text.split(',')]
        
        # Extract sections
        section_pattern = r'\\section\{([^}]+)\}(.*?)(?=\\section|\\begin\{thebibliography\}|\\end\{document\})'
        for match in re.finditer(section_pattern, latex_content, re.DOTALL):
            section_content = match.group(2)
            
            # Process subsections within
            subsections = []
            subsec_pattern = r'\\subsection\{([^}]+)\}(.*?)(?=\\subsection|$)'
            for sub_match in re.finditer(subsec_pattern, section_content, re.DOTALL):
                subsections.append({
                    'title': self._clean_latex(sub_match.group(1)),
                    'content': self._clean_latex(sub_match.group(2)),
                    'level': 2
                })
            
            # Main section content (before any subsections)
            main_content = re.split(r'\\subsection', section_content)[0]
            
            content['sections'].append({
                'title': self._clean_latex(match.group(1)),
                'content': self._clean_latex(main_content),
                'level': 1
            })
            
            content['sections'].extend(subsections)
        
        # Extract references
        bibitem_pattern = r'\\bibitem\{[^}]+\}\s*(.+?)(?=\\bibitem|\\end\{thebibliography\})'
        for match in re.finditer(bibitem_pattern, latex_content, re.DOTALL):
            content['references'].append(self._clean_latex(match.group(1)))
        
        return content
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands and clean text"""
        import re
        
        if not text:
            return ""
        
        # Handle formatting commands
        text = re.sub(r'\\textbf\{([^}]+)\}', r'<b>\1</b>', text)
        text = re.sub(r'\\textit\{([^}]+)\}', r'<i>\1</i>', text)
        text = re.sub(r'\\emph\{([^}]+)\}', r'<i>\1</i>', text)
        text = re.sub(r'\\texttt\{([^}]+)\}', r'<font face="Courier">\1</font>', text)
        
        # Remove cite commands but keep reference number
        text = re.sub(r'\\cite\{ref(\d+)\}', r'[\1]', text)
        text = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', text)
        
        # Remove other LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _create_latex_only(self, latex_content: str, output_path: str) -> bool:
        """Save LaTeX source when no PDF compiler is available"""
        try:
            tex_path = output_path.replace('.pdf', '.tex')
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Create notice
            notice_path = output_path.replace('.pdf', '_INSTRUCTIONS.txt')
            with open(notice_path, 'w') as f:
                f.write("IEEE Conference Paper - Compilation Instructions\n")
                f.write("=" * 50 + "\n\n")
                f.write("To generate the PDF, you need one of the following:\n\n")
                f.write("Option 1: Install TeX Live (recommended)\n")
                f.write("  - macOS: brew install --cask mactex\n")
                f.write("  - Ubuntu: sudo apt-get install texlive-full\n")
                f.write("  - Windows: Download from https://tug.org/texlive/\n\n")
                f.write("Option 2: Use Overleaf\n")
                f.write("  - Upload the .tex file to https://overleaf.com\n")
                f.write("  - Compile online\n\n")
                f.write(f"LaTeX source saved to: {tex_path}\n")
            
            print(f"LaTeX source saved to: {tex_path}")
            return False
        except Exception as e:
            print(f"Error saving LaTeX: {e}")
            return False
