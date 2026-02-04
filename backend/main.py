"""
IEEE Conference Paper Converter - FastAPI Backend
A production-ready API for converting academic papers to IEEE format
"""

import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from extractors.content_analyzer import ContentAnalyzer
from generators.latex_generator import LaTeXGenerator
from generators.pdf_compiler import PDFCompiler
from validators.ieee_invariant_enforcer import IEEEInvariantEnforcer

# Initialize FastAPI app
app = FastAPI(
    title="IEEE Conference Paper Converter",
    description="Convert PDF/Word documents to IEEE-formatted conference papers",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

# In-memory job storage (use Redis/DB in production)
jobs = {}

# Current active template (default: None uses built-in IEEEtran)
active_template = {"path": None, "name": "IEEEtran (built-in)"}

# Pydantic models
class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    output_file: Optional[str] = None
    source_file: Optional[str] = None
    error: Optional[str] = None

class ConversionResponse(BaseModel):
    job_id: str
    message: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "IEEE Paper Converter"}


@app.post("/api/upload", response_model=ConversionResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF or DOCX file for conversion to IEEE format
    """
    # Validate file type
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only PDF and DOCX files are accepted."
        )
    
    # Validate file size (50MB max)
    MAX_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 50MB."
        )
    
    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    file_ext = ".pdf" if file.content_type == "application/pdf" else ".docx"
    input_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    with open(input_path, "wb") as f:
        f.write(content)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "File uploaded, starting conversion...",
        "created_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_file": None,
        "error": None
    }
    
    # Start background conversion
    background_tasks.add_task(process_conversion, job_id, str(input_path), file_ext)
    
    return ConversionResponse(
        job_id=job_id,
        message="File uploaded successfully. Conversion started."
    )


def create_source_bundle(job_id: str, latex_content: str, structured_paper) -> str:
    """Create a ZIP bundle with LaTeX source and figures for Overleaf"""
    bundle_dir = OUTPUT_DIR / f"{job_id}_source"
    figures_dir = bundle_dir / "figures"
    
    # Create directories
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    figures_dir.mkdir()
    
    # Copy images and update LaTeX content
    new_latex = latex_content
    
    if hasattr(structured_paper, 'figures'):
        for fig in structured_paper.figures:
            if fig.image_path and os.path.exists(fig.image_path):
                # Copy image
                filename = os.path.basename(fig.image_path)
                # Ensure unique filename if needed? Assuming unique for now.
                shutil.copy2(fig.image_path, figures_dir / filename)
                
                # Replace path in LaTeX
                # Look for absolute path in latex
                # We used os.path.abspath(fig.image_path).replace('\\', '/') in Generator
                abs_path = os.path.abspath(fig.image_path).replace('\\', '/')
                rel_path = f"figures/{filename}"
                
                # Simple string replacement (safe enough if path is unique string)
                new_latex = new_latex.replace(abs_path, rel_path)
    
    # Write main.tex
    with open(bundle_dir / "main.tex", "w") as f:
        f.write(new_latex)
        
    # Create ZIP
    zip_path = OUTPUT_DIR / f"{job_id}_source" # make_archive adds .zip
    shutil.make_archive(str(zip_path), 'zip', bundle_dir)
    
    # Cleanup directory
    shutil.rmtree(bundle_dir)
    
    return str(zip_path.with_suffix('.zip'))


async def process_conversion(job_id: str, input_path: str, file_ext: str):
    """
    Background task to process document conversion
    """
    try:
        # Step 1: Parse document
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Parsing document..."
        
        if file_ext == ".pdf":
            parser = PDFParser()
            raw_content = parser.parse(input_path)
        else:
            parser = DOCXParser()
            raw_content = parser.parse(input_path)
        
        # Step 2: Extract and analyze content
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Analyzing content structure..."
        
        analyzer = ContentAnalyzer()
        structured_content = analyzer.analyze(raw_content)
        
        # Step 2.5: Enforce IEEE Invariants
        jobs[job_id]["message"] = "Verifying IEEE compliance..."
        enforcer = IEEEInvariantEnforcer()
        validation = enforcer.validate(structured_content)
        
        if not validation["passed"]:
            error_msg = "IEEE Compliance Check Failed:\n" + "\n".join(validation["errors"])
            raise Exception(error_msg)
        
        # Step 3: Generate LaTeX
        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Generating IEEE LaTeX document..."
        
        generator = LaTeXGenerator()
        latex_content = generator.generate(structured_content, job_id)
        
        # Step 3.5: Final Hard Gate (Output Validation)
        # Verify the GENERATED LaTeX before compiling to ensure no errors slipped through
        output_validation = enforcer.validate_latex(latex_content)
        if not output_validation["passed"]:
            error_msg = "Final Hard Gate Failed: Output Verification Errors:\n" + "\n".join(output_validation["errors"])
            raise Exception(error_msg)
        
        # Step 4: Compile PDF
        jobs[job_id]["progress"] = 70
        jobs[job_id]["message"] = "Compiling PDF..."
        
        compiler = PDFCompiler()
        output_path = OUTPUT_DIR / f"{job_id}.pdf"
        success = compiler.compile(latex_content, str(output_path), job_id)
        
        if success:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Conversion completed successfully!"
            jobs[job_id]["output_file"] = str(output_path)
            
            # Create Source Bundle (ZIP for Overleaf)
            source_zip = create_source_bundle(job_id, latex_content, structured_content)
            jobs[job_id]["source_file"] = str(source_zip)
        else:
            raise Exception("PDF compilation failed")
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = "Conversion failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """
    Check the status of a conversion job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        output_file=job.get("output_file"),
        error=job.get("error")
    )


@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """
    Get the converted PDF for preview
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    output_path = job.get("output_file")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )


@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """
    Download the converted IEEE-formatted PDF
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    output_path = job.get("output_file")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"ieee_paper_{job_id[:8]}.pdf",
        headers={"Content-Disposition": "attachment"}
    )


@app.get("/api/download-source/{job_id}")
async def download_source(job_id: str):
    """
    Download the LaTeX source code (ZIP) for Overleaf
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    source_path = job.get("source_file")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Source bundle not found")
    
    return FileResponse(
        source_path,
        media_type="application/zip",
        filename=f"ieee_source_{job_id[:8]}.zip",
        headers={"Content-Disposition": "attachment"}
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """
    Clean up uploaded and generated files for a job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Remove input file
    input_path = job.get("input_path")
    if input_path and Path(input_path).exists():
        Path(input_path).unlink()
    
    # Remove output file
    output_path = job.get("output_file")
    if output_path and Path(output_path).exists():
        Path(output_path).unlink()
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"message": "Job cleaned up successfully"}


@app.post("/api/template/upload")
async def upload_template(file: UploadFile = File(...)):
    """
    Upload a custom IEEE template file (LaTeX .tex or .cls)
    This template will be used for subsequent conversions.
    """
    global active_template
    
    # Validate file type
    allowed_extensions = [".tex", ".cls", ".pdf"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid template file type: {file_ext}. Allowed: .tex, .cls, .pdf"
        )
    
    # Save template file
    content = await file.read()
    template_path = TEMPLATE_DIR / file.filename
    
    with open(template_path, "wb") as f:
        f.write(content)
    
    # Set as active template
    active_template = {
        "path": str(template_path),
        "name": file.filename
    }
    
    return {
        "message": f"Template '{file.filename}' uploaded successfully",
        "template_name": file.filename,
        "template_path": str(template_path)
    }


@app.get("/api/template/list")
async def list_templates():
    """
    List all available templates in the templates directory
    """
    templates = []
    
    # Add built-in template
    templates.append({
        "name": "IEEEtran (built-in)",
        "path": None,
        "is_active": active_template["path"] is None
    })
    
    # Add custom templates
    for file in TEMPLATE_DIR.iterdir():
        if file.suffix.lower() in [".tex", ".cls", ".pdf"]:
            templates.append({
                "name": file.name,
                "path": str(file),
                "is_active": str(file) == active_template.get("path")
            })
    
    return {"templates": templates, "active": active_template["name"]}


@app.post("/api/template/select/{template_name}")
async def select_template(template_name: str):
    """
    Select a template to use for conversions
    """
    global active_template
    
    if template_name == "built-in" or template_name == "IEEEtran (built-in)":
        active_template = {"path": None, "name": "IEEEtran (built-in)"}
        return {"message": "Switched to built-in IEEEtran template"}
    
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    active_template = {
        "path": str(template_path),
        "name": template_name
    }
    
    return {"message": f"Switched to template '{template_name}'"}


@app.delete("/api/template/{template_name}")
async def delete_template(template_name: str):
    """
    Delete a custom template
    """
    global active_template
    
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    template_path.unlink()
    
    # If this was the active template, reset to built-in
    if active_template.get("path") == str(template_path):
        active_template = {"path": None, "name": "IEEEtran (built-in)"}
    
    return {"message": f"Template '{template_name}' deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

