"""
BMI Prediction Backend Server

FastAPI backend for BMI prediction from images.
Loads trained PyTorch models and provides a REST API for predictions.

Run with:
    uvicorn backend.main:app --reload
    
Or for production:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

from predictor import predict_from_image, calculate_bmi, get_bmi_category
from gemini_service import GeminiService


# ============================================================================
# Initialize FastAPI app
# ============================================================================

app = FastAPI(
    title="BMI Prediction API",
    description="Predicts height, weight, and BMI from person's image",
    version="1.0.0"
)

# Enable CORS for mobile app to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log all incoming requests
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"🔵 [REQUEST] {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        print(f"🟢 [RESPONSE] {response.status_code}")
        return response
    except Exception as e:
        print(f"🔴 [ERROR] {str(e)}")
        raise


# ============================================================================
# Response Models
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for BMI prediction endpoint."""
    height_cm: float
    weight_kg: float
    bmi: float
    category: str
    prior_bmi: float
    prior_category: str
    recommendation: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Server startup initialization.
    Preloads Keypoint R-CNN and configures Gemini API.
    """
    print("\n" + "="*60)
    print("Starting BMI Prediction Server (Pose-Based Estimation)")
    print("="*60)
    try:
        # Preload Keypoint R-CNN model weights (one-time, ~170MB download)
        print("Initializing pose estimator…")
        from pose_estimator import load_pose_model
        load_pose_model()
        print("✓ Pose estimator ready")
        
        # Configure Gemini API for health recommendations
        GeminiService.configure()
        print("✓ Health recommendation service ready")
        
        print("✓ Server ready for predictions")
    except Exception as e:
        print(f"✗ Failed to initialize server: {str(e)}")
        raise
    print("="*60 + "\n")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    print("✓ GET / called")
    return {
        "service": "BMI Prediction API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/test")
async def test():
    """Simple test endpoint to verify connectivity."""
    print("✓ GET /test called - connection working!")
    return {"status": "ok", "message": "Phone can reach backend"}


@app.post("/test-post")
async def test_post():
    """Simple POST test endpoint."""
    print("✓ POST /test-post called - connection working!")
    return {"status": "ok", "message": "POST connection working"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict height, weight, and BMI from an uploaded image.
    
    Process:
    1. Validate image upload
    2. Detect body pose using Keypoint R-CNN
    3. Estimate height and weight from pose landmarks
    4. Calculate BMI
    5. Determine BMI category
    6. Generate health recommendation
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        
    Returns:
        PredictionResponse with height, weight, BMI, category, and recommendation
        
    Raises:
        HTTPException: If image cannot be processed or pose cannot be detected
    """
    try:
        # Validate file is an image
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        print(f"\n📷 Processing image: {file.filename}")
        
        # Pose detection and dimensional estimation
        height, weight, prior_bmi, error_msg = predict_from_image(image_bytes)
        
        if error_msg or height is None or weight is None or prior_bmi is None:
            error_detail = error_msg or "Failed to detect pose in image"
            print(f"✗ Pose estimation failed: {error_detail}\n")
            raise HTTPException(
                status_code=400,
                detail=error_detail
            )
        
        # Type narrowing: height, weight, prior_bmi are guaranteed non-None at this point
        assert height is not None and weight is not None and prior_bmi is not None
        
        print(f"✓ Predictions: Height={height:.2f} cm, Weight={weight:.2f} kg, Prior BMI={prior_bmi:.2f}")
        
        # Calculate final BMI (based on clipped weight)
        bmi = calculate_bmi(height, weight)
        category = get_bmi_category(bmi)
        
        # Calculate prior category (based on prior BMI)
        prior_category = get_bmi_category(prior_bmi)
        
        print(f"✓ BMI calculated: {bmi:.2f} ({category}) | Prior: {prior_bmi:.2f} ({prior_category})")
        
        # Generate health recommendation with weight for personalized macros
        recommendation = GeminiService.generate_recommendation(bmi, category, weight)
        if recommendation is None:
            recommendation = f"Maintain a healthy lifestyle for your {category} BMI category."
        print(f"✓ Recommendation generated")
        
        # Return response
        response = PredictionResponse(
            height_cm=float(round(height, 2)),
            weight_kg=float(round(weight, 2)),
            bmi=float(round(bmi, 2)),
            category=category,
            prior_bmi=float(round(prior_bmi, 2)),
            prior_category=prior_category,
            recommendation=recommendation
        )
        
        print(f"✓ Response sent\n")
        return response
        
    except HTTPException as e:
        print(f"✗ HTTP Error: {e.detail}\n")
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


# ============================================================================
# Run Information
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=True,  # Enable access logging to see all HTTP requests
        log_level="debug"  # Show all logs including debug level
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
