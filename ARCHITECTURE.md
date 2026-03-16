# Architecture & Design

Complete architectural overview of the BMI Prediction Application.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MOBILE APPLICATION                       │
│                   (React Native + Expo)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   App Component                         │ │
│  │  - State: selectedImage, results, loading, error       │ │
│  │  - Image Picker Integration                            │ │
│  │  - Form Data Construction                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓ HTTP POST                        │
│                 (multipart/form-data)                         │
└─────────────────────────────────────────────────────────────┘
                             ↓↑
        ┌────────────────────────────────────────────────┐
        │           BACKEND SERVER (FastAPI)             │
        │         http://localhost:8000                  │
        ├────────────────────────────────────────────────┤
        │                                                │
        │  ┌──────────────────────────────────────────┐ │
        │  │          main.py (FastAPI app)           │ │
        │  │  - Receives image upload                 │ │
        │  │  - Orchestrates prediction pipeline      │ │
        │  │  - Returns JSON response                 │ │
        │  └──────────────────────────────────────────┘ │
        │   ↓                                            │
        │  ┌──────────────────────────────────────────┐ │
        │  │   preprocess.py (Image Processing)       │ │
        │  │  - Load PIL Image from bytes             │ │
        │  │  - Convert to RGB                        │ │
        │  │  - Resize to 224×224                     │ │
        │  │  - Normalize with ImageNet stats         │ │
        │  │  - Return torch.Tensor                   │ │
        │  └──────────────────────────────────────────┘ │
        │   ↓                                            │
        │  ┌──────────────────────────────────────────┐ │
        │  │ model_loader.py (Model Management)       │ │
        │  │  - Loads models at startup               │ │
        │  │  - Returns cached model instances        │ │
        │  │  - Selects device (GPU/CPU)              │ │
        │  └──────────────────────────────────────────┘ │
        │   ↓                         ↓                  │
        │  ┌──────────────────────────────────────────┐ │
        │  │    predictor.py (Inference & Math)       │ │
        │  │  ┌────────────────────────────────────┐ │ │
        │  │  │ predict_height()                   │ │ │
        │  │  │ Model: model_ep_48.pth             │ │ │
        │  │  │ Output: Height (cm)                │ │ │
        │  │  └────────────────────────────────────┘ │ │
        │  │  ┌────────────────────────────────────┐ │ │
        │  │  │ predict_weight()                   │ │ │
        │  │  │ Model: model_ep_37.pth             │ │ │
        │  │  │ Output: Weight (kg)                │ │ │
        │  │  └────────────────────────────────────┘ │ │
        │  │  ┌────────────────────────────────────┐ │ │
        │  │  │ calculate_bmi()                    │ │ │
        │  │  │ Formula: weight / (height/100)²    │ │ │
        │  │  │ Output: BMI value                  │ │ │
        │  │  └────────────────────────────────────┘ │ │
        │  │  ┌────────────────────────────────────┐ │ │
        │  │  │ get_bmi_category()                 │ │ │
        │  │  │ Output: Category string            │ │ │
        │  │  └────────────────────────────────────┘ │ │
        │  └──────────────────────────────────────────┘ │
        │   ↓                                            │
        │  ┌──────────────────────────────────────────┐ │
        │  │ gemini_service.py (AI Recommendations)  │ │
        │  │  - Calls Gemini API                     │ │
        │  │  - Generates health advice              │ │
        │  │  - Returns recommendation text          │ │
        │  └──────────────────────────────────────────┘ │
        │   ↓                                            │
        │  ┌──────────────────────────────────────────┐ │
        │  │     main.py (Response Assembly)          │ │
        │  │  - Combine all predictions               │ │
        │  │  - Return JSON to mobile app             │ │
        │  └──────────────────────────────────────────┘ │
        │                                                │
        └────────────────────────────────────────────────┘
                             ↓↑
                   (JSON Response)
                             ↓
        ┌─────────────────────────────────────────┐
        │        MOBILE APP (Display)              │
        │  - Show height, weight, BMI             │
        │  - Display category with color          │
        │  - Show health recommendation           │
        └─────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
User Interface
    │
    ├─→ Image Picker (Expo)
    │       └─→ File URI
    │
    ├─→ FormData Creation
    │       ├─ File blob
    │       ├─ Content-Type: multipart/form-data
    │       └─→ Axios HTTP POST
    │
    ├─→ API Communication
    │       ├─ URL: http://localhost:8000/predict
    │       ├─ Method: POST
    │       └─→ Backend Processing
    │
    └─→ Loading State
            ├─ Show spinner
            ├─ Disable buttons
            └─→ Wait for response
                    │
                    ├─→ Success: Display results
                    ├─→ Error: Show error message
                    └─→ Timeout: Retry option
```

---

## Data Flow - Request

```
1. User Action
   User taps "Select Image"
        ↓
2. Image Selection
   Gallery opens → User selects photo
        ↓
3. Image Processing (Client)
   Image URI → FormData
        ↓
4. HTTP Request
   POST /predict with FormData
        ↓
5. Server Reception
   FastAPI receives multipart form-data
        ↓
6. Validation
   - Check file is image (MIME type)
   - Check file not empty
   - Check models loaded
        ↓
7. Image Preprocessing
   Bytes → PIL Image
        → RGB conversion
        → Resize 224×224
        → Normalize
        → Tensor
        ↓
8. Model Inference
   Tensor → Height Model → height_cm
   Tensor → Weight Model → weight_kg
        ↓
9. Calculations
   height_cm + weight_kg → BMI
   BMI → Category
        ↓
10. AI Generation
    BMI + Category → Gemini API → Recommendation
        ↓
11. Response Assembly
    {height_cm, weight_kg, bmi, category, recommendation}
        ↓
12. HTTP Response
    200 OK + JSON payload
        ↓
13. Client Processing
    Parse JSON → Update state
        ↓
14. UI Rendering
    Display results with formatting
```

---

## Model Pipeline

```
Input Image (any size/format)
        ↓
    PIL Image
        ↓
    RGB Conversion
   ┌─────────────┐
   │ JPG → RGB   │
   │ PNG → RGB   │
   │ etc → RGB   │
   └─────────────┘
        ↓
    Resize
   ┌─────────────┐
   │ 224 × 224   │
   │ (standard)  │
   └─────────────┘
        ↓
    to_tensor()
   ┌──────────────┐
   │ Normalize    │
   │ ImageNet     │
   │ Mean: [0.485,│
   │       0.456, │
   │       0.406] │
   │ Std:  [0.229,│
   │       0.224, │
   │       0.225] │
   └──────────────┘
        ↓
    Add Batch Dim
   ┌──────────────┐
   │ (1, 3, 224,  │
   │      224)    │
   └──────────────┘
        ↓
    Model 1 (Height)
   ┌──────────────┐
   │ model_ep_48  │
   │ Output:      │
   │ height_cm    │
   └──────────────┘
        ↓
    Model 2 (Weight)
   ┌──────────────┐
   │ model_ep_37  │
   │ Output:      │
   │ weight_kg    │
   └──────────────┘
```

---

## BMI Calculation Logic

```
Height (cm), Weight (kg) → BMI

Step 1: Convert height to meters
  height_m = height_cm / 100.0

Step 2: Square height
  height_squared = height_m ^ 2

Step 3: Calculate BMI
  BMI = weight_kg / height_squared

Step 4: Categorize
  IF BMI < 18.5:
    Category = "Underweight"
  ELSE IF 18.5 ≤ BMI < 25:
    Category = "Normal"
  ELSE IF 25 ≤ BMI < 30:
    Category = "Overweight"
  ELSE (BMI ≥ 30):
    Category = "Obese"

Example:
  Height: 175 cm → 1.75 m
  Weight: 72 kg
  BMI = 72 / (1.75²) = 72 / 3.0625 ≈ 23.5
  Category: Normal ✓
```

---

## State Management (Mobile App)

```
App Component State

┌─────────────────────────────────┐
│      selectedImage              │
│  ┌─────────────────────────────┐│
│  │ null                         ││
│  │ OR                           ││
│  │ {                            ││
│  │   uri: "file://...",         ││
│  │   type: "image/jpeg",        ││
│  │   width: 1080,               ││
│  │   height: 1920               ││
│  │ }                            ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
         ↓ When user selects
         
┌─────────────────────────────────┐
│        results                  │
│  ┌─────────────────────────────┐│
│  │ null                         ││
│  │ OR                           ││
│  │ {                            ││
│  │   height_cm: 175.32,         ││
│  │   weight_kg: 72.48,          ││
│  │   bmi: 23.6,                 ││
│  │   category: "Normal",        ││
│  │   recommendation: "..."      ││
│  │ }                            ││
│  └─────────────────────────────┘│
│                                  │
│  Set when prediction succeeds    │
│  Cleared when new image selected │
└─────────────────────────────────┘

         ↓ During prediction
         
┌─────────────────────────────────┐
│        loading                  │
│  ┌─────────────────────────────┐│
│  │ false (default)              ││
│  │ true (during API call)       ││
│  │                              ││
│  │ Used to show spinner         ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘

         ↓ If error occurs
         
┌─────────────────────────────────┐
│        error                    │
│  ┌─────────────────────────────┐│
│  │ null (no error)              ││
│  │ OR                           ││
│  │ "Error message string"       ││
│  │                              ││
│  │ Used to show alert/banner    ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

---

## API Response Structure

```
HTTP Status: 200 OK

{
  "height_cm": float (2 decimals),
    └─ Range: typically 140-220 cm
    └─ Predicted by: model_ep_48.pth
    
  "weight_kg": float (2 decimals),
    └─ Range: typically 30-200 kg
    └─ Predicted by: model_ep_37.pth
    
  "bmi": float (2 decimals),
    └─ Calculated: weight_kg / (height_m²)
    └─ Max precision: 2 decimal places
    
  "category": string,
    └─ Possible values:
       - "Underweight" (BMI < 18.5)
       - "Normal" (18.5 ≤ BMI < 25)
       - "Overweight" (25 ≤ BMI < 30)
       - "Obese" (BMI ≥ 30)
    
  "recommendation": string,
    └─ Generated by: Google Gemini API
    └─ Contains: Diet, exercise, lifestyle tips
    └─ Length: Typically 2-3 sentences
    └─ If API fails: Error message or placeholder
}
```

---

## Error Handling Flow

```
User Action
    ├─→ Validation Errors
    │   ├─ No image selected
    │   │  └─→ Alert: "Please select an image"
    │   │
    │   └─ Invalid file type
    │      └─→ Alert: "File must be..."
    │
    ├─→ Network Errors
    │   ├─ No connection
    │   │  └─→ Alert: "Cannot connect to backend"
    │   │
    │   └─ Timeout (30s)
    │      └─→ Alert: "Request timed out"
    │
    ├─→ Server Errors
    │   ├─ 400 Bad Request
    │   │  └─→ Alert: Server error message
    │   │
    │   ├─ 500 Internal Error
    │   │  └─→ Alert: "Server error"
    │   │
    │   └─ 503 Service Unavailable
    │      └─→ Alert: "Server not ready"
    │
    └─→ Processing Errors
        ├─ Image corruption
        │  └─→ Alert: "Image processing failed"
        │
        └─ Model error
           └─→ Alert: "Prediction failed"
```

---

## Deployment Architecture

```
PRODUCTION SETUP

┌─────────────────────────────────────────────────────────────┐
│                    User's Device                             │
│                  (Mobile Phone/Web)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            React Native Mobile App                   │   │
│  │          or Web Browser (Expo Web)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTPS
    ┌────────────────────────────────────────────────────┐
    │          Application Load Balancer                 │
    │              (Optional, multi-server)              │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │         Reverse Proxy (Nginx)                      │
    │  - SSL termination                                 │
    │  - Rate limiting                                   │
    │  - Request routing                                 │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │    FastAPI Application Server (Gunicorn)           │
    │  - Multiple worker processes                       │
    │  - Auto-reload disabled                            │
    │  - Production configuration                        │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │         Application Logic                          │
    │  - Model inference                                 │
    │  - Image processing                                │
    │  - API integrations                                │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │     Model Files (GPU-cached)                       │
    │  - model_ep_48.pth (Height)                        │
    │  - model_ep_37.pth (Weight)                        │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │    External APIs                                   │
    │  - Google Gemini API                               │
    │  - (for health recommendations)                    │
    └────────────────────────────────────────────────────┘
```

---

## Performance Optimization

### Caching Strategy
```
Level 1: Model Caching
├─ Load models once at startup
├─ Keep in GPU memory
└─ Reuse for all predictions

Level 2: Image Processing
├─ Optimize PIL operations
├─ Batch processing (future)
└─ Resize once, use for both models

Level 3: Recommendation Caching
├─ Cache Gemini API responses
├─ Key: (BMI, Category) tuple
└─ TTL: 1 hour
```

### Memory Management
```
Total Memory Usage
├─ Model 1: ~500 MB
├─ Model 2: ~500 MB
├─ Buffer: ~500 MB
└─ Runtime: ~500 MB
Total: ~2 GB
```

---

## Security Architecture

```
Request Flow with Security

1. Input Validation
   ├─ MIME type check
   ├─ File size limit (50 MB)
   ├─ File content scan
   └─ ✓ Passed → Continue

2. Processing
   ├─ Sandboxed image processing
   ├─ Model inference isolation
   └─ Error handling (no stack traces)

3. Output
   ├─ Sanitized response
   ├─ No sensitive info leaked
   └─ CORS validated

4. Network
   ├─ HTTPS only (production)
   ├─ TLS 1.3+
   └─ Secure headers
```

---

## Scalability Plan

```
Current: Single Server
├─ 1 backend instance
├─ 2-3 GB memory
└─ Handle ~100 concurrent users

Phase 1: Load Balancing
├─ Multiple backend instances
├─ Shared model cache (Redis)
└─ Load balancer distribution

Phase 2: Microservices
├─ Image processing service
├─ Model inference service
├─ API service
└─ Message queue (Celery)

Phase 3: Cloud Native
├─ Kubernetes orchestration
├─ Auto-scaling policies
├─ Model serving (TensorFlow Serving)
└─ Distributed tracing
```

---

This architecture provides a scalable, maintainable, and production-ready BMI prediction system.
