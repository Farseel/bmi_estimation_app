# Backend API Server

FastAPI-based backend for BMI prediction from images using PyTorch CNN models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIzaSyDHhSDafHDEPhA7ofYtZM5pNUtGLCIKKW8"

# Or create .env file
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

For Gemini API key, visit: https://makersuite.google.com/app/apikey

### 3. Run Server

```bash
uvicorn main:app --reload
```

Server starts at: **http://localhost:8000**

- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Architecture

### Model Loader (`model_loader.py`)
Manages PyTorch model lifecycle:
- Loads models once at startup
- Handles GPU/CPU device selection
- Provides model access methods

**Models:**
- `model_ep_48.pth` → Height prediction (centimeters)
- `model_ep_37.pth` → Weight prediction (kilograms)

### Image Preprocessing (`preprocess.py`)
Transforms uploaded images for model inference:

```
Input Image (any size/format)
    ↓
Load PIL Image
    ↓
Convert to RGB
    ↓
Resize to 224×224
    ↓
Convert to Tensor
    ↓
Normalize (ImageNet stats)
    ↓
Add Batch Dimension
    ↓
Output: (1, 3, 224, 224) Tensor
```

### Prediction (`predictor.py`)
Functions for inference:
- `predict_height(tensor)` → float (cm)
- `predict_weight(tensor)` → float (kg)
- `calculate_bmi(height, weight)` → float
- `get_bmi_category(bmi)` → str

BMI Categories:
- < 18.5 → "Underweight"
- 18.5 - 24.9 → "Normal"
- 25 - 29.9 → "Overweight"
- ≥ 30 → "Obese"

### Gemini Service (`gemini_service.py`)
Generates health recommendations:

```python
recommendation = GeminiService.generate_recommendation(bmi=24.5, category="Normal")
```

Returns AI-generated health advice including diet and exercise suggestions.

---

## API Endpoints

### GET /
Health check endpoint.

**Response:**
```json
{
  "service": "BMI Prediction API",
  "status": "running",
  "version": "1.0.0"
}
```

---

### POST /predict
Main prediction endpoint. Upload image and get results.

**Request:**
```
POST /predict
Content-Type: multipart/form-data

file: <image_file> (PNG, JPG, or JPEG)
```

**Response (200 OK):**
```json
{
  "height_cm": 175.32,
  "weight_kg": 72.48,
  "bmi": 23.6,
  "category": "Normal",
  "recommendation": "Your BMI is in the normal range. Continue maintaining a balanced diet with regular exercise..."
}
```

**Errors:**
- `400 Bad Request`: Invalid image format or empty file
- `413 Payload Too Large`: Image file too large
- `500 Internal Server Error`: Model error

---

## Request/Response Examples

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

### Using Python Requests
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

### Using Axios (JavaScript/React Native)
```javascript
const formData = new FormData();
formData.append('file', {
  uri: 'file://path/to/image.jpg',
  type: 'image/jpeg',
  name: 'image.jpg'
});

const response = await axios.post('http://localhost:8000/predict', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});
```

---

## Performance

| Metric | Value |
|--------|-------|
| **First Request** | 2-5 seconds (models load/compile) |
| **Subsequent Requests** | 1-2 seconds |
| **With GPU** | 0.5-1 second |
| **Max Image Size** | ~50 MB |
| **Supported Formats** | PNG, JPG, JPEG |

---

## Configuration

### Device Selection
Automatically uses GPU if available:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Image Size
Configured in `preprocess.py`:
```python
IMAGE_SIZE = 224  # ResNet standard
```

### Model Paths
```python
HEIGHT_MODEL_PATH = "../models/model_ep_48.pth"
WEIGHT_MODEL_PATH = "../models/model_ep_37.pth"
```

---

## CORS Configuration

Currently allows all origins (for development). In production, update in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

---

## Logging

The server prints detailed logs:

```
============================================================
Starting BMI Prediction Server
============================================================
Loading height model from: ../models/model_ep_48.pth
✓ Height model loaded successfully
Loading weight model from: ../models/model_ep_37.pth
✓ Weight model loaded successfully
✓ Gemini API configured successfully
✓ Server ready for predictions
============================================================

📷 Processing image: photo.jpg
✓ Image preprocessed
✓ Predictions: Height=175.3 cm, Weight=72.5 kg
✓ BMI calculated: 23.6 (Normal)
✓ Recommendation generated
✓ Response sent
```

---

## Error Handling

### Invalid Image
```json
{
  "detail": "File must be an image (PNG, JPG, JPEG)"
}
```

### Models Not Loaded
```json
{
  "detail": "Models not loaded. Server not ready."
}
```

### Processing Error
```json
{
  "detail": "Image processing failed: <error details>"
}
```

---

## Development

### Start with Auto-Reload
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Debug Mode
```python
# In main.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Endpoint
```bash
# Health check
curl http://localhost:8000/

# With Swagger UI
open http://localhost:8000/docs
```

---

## Production Deployment

### Using Gunicorn + Uvicorn
```bash
pip install gunicorn
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t bmi-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key bmi-api
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.104.1 | Web framework |
| `uvicorn` | 0.24.0 | ASGI server |
| `torch` | 2.1.1 | Neural networks |
| `torchvision` | 0.16.1 | Image processing |
| `pillow` | 10.1.0 | Image I/O |
| `numpy` | 1.24.3 | Numerical computing |
| `google-generativeai` | 0.8.3 | Gemini API |

---

## Troubleshooting

### torch not found
```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Models not found
Ensure `models/` directory exists at project root with both `.pth` files.

### Port already in use
```bash
# Use different port
uvicorn main:app --port 8001

# Or kill process using port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000   # Windows
```

### GEMINI_API_KEY errors
```bash
# Set environment variable
export GEMINI_API_KEY="your_key"  # Linux/macOS
set GEMINI_API_KEY=your_key       # Windows CMD
$env:GEMINI_API_KEY="your_key"    # PowerShell
```

---

## Next Steps

1. Place model files in `models/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Set GEMINI_API_KEY (optional)
4. Run server: `uvicorn main:app --reload`
5. Test with mobile app or cURL
