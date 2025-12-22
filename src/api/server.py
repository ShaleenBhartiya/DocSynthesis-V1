"""
FastAPI REST API Server for DocSynthesis-V1
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import uuid
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import Settings
from main import DocSynthesisV1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DocSynthesis-V1 API",
    description="Intelligent Document Processing for IndiaAI Challenge",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Initialize settings and processing system
settings = Settings()
processor = None

# Job storage (in production, use Redis or database)
jobs = {}


class ProcessingOptions(BaseModel):
    """Processing options for document."""
    language: Optional[str] = None
    translate: bool = False
    extract_fields: bool = True
    generate_summary: bool = True
    explain: bool = False


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: int
    stages: Optional[Dict[str, str]] = None
    estimated_time: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup."""
    global processor
    logger.info("Initializing DocSynthesis-V1 processor...")
    try:
        processor = DocSynthesisV1()
        logger.info("Processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "DocSynthesis-V1",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "process": "/api/v1/process",
            "status": "/api/v1/status/{job_id}",
            "results": "/api/v1/results/{job_id}",
            "health": "/api/v1/health"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "processor_ready": processor is not None
    }


@app.post("/api/v1/process")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: Optional[str] = None
):
    """
    Process a document through the DocSynthesis-V1 pipeline.
    
    Args:
        file: Document file (PDF, JPG, PNG)
        options: JSON string with processing options
        
    Returns:
        Job ID and status
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Parse options
    import json
    proc_options = ProcessingOptions()
    if options:
        try:
            opts_dict = json.loads(options)
            proc_options = ProcessingOptions(**opts_dict)
        except Exception as e:
            logger.error(f"Failed to parse options: {e}")
    
    # Save uploaded file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "file_path": str(file_path),
        "options": proc_options.dict(),
        "result": None
    }
    
    # Add processing task to background
    background_tasks.add_task(process_job, job_id, str(file_path), proc_options)
    
    logger.info(f"Job {job_id} queued for processing")
    
    return {
        "job_id": job_id,
        "status": "processing",
        "estimated_time": 30
    }


async def process_job(job_id: str, file_path: str, options: ProcessingOptions):
    """Background task to process document."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        
        # Create output directory
        output_dir = Path("output") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process document
        result = processor.process(
            input_path=file_path,
            output_dir=str(output_dir),
            language=options.language,
            translate=options.translate,
            extract_fields=options.extract_fields,
            generate_summary=options.generate_summary,
            explain=options.explain
        )
        
        # Update job
        jobs[job_id]["status"] = "completed" if result["status"] == "completed" else "failed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = result
        
        # Clean up temp file
        Path(file_path).unlink(missing_ok=True)
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/v1/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing status for a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        Job status information
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stages": {
            "preprocessing": "completed" if job["progress"] > 20 else "pending",
            "ocr": "completed" if job["progress"] > 50 else "pending",
            "extraction": "completed" if job["progress"] > 80 else "pending"
        }
    }


@app.get("/api/v1/results/{job_id}")
async def get_job_results(job_id: str):
    """
    Get processing results for a completed job.
    
    Args:
        job_id: Job ID
        
    Returns:
        Processing results
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "processing" or job["status"] == "queued":
        raise HTTPException(status_code=202, detail="Job still processing")
    
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Processing failed"))
    
    return job["result"]


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its results.
    
    Args:
        job_id: Job ID
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up files
    output_dir = Path("output") / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.api.enable_cors else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_server():
    """Start the API server."""
    uvicorn.run(
        "src.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=False
    )


if __name__ == "__main__":
    start_server()

