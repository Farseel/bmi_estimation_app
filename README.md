# BMI Prediction Application

A comprehensive full-stack application for predicting Body Mass Index (BMI), height, and weight from person images using advanced machine learning. Features a FastAPI backend with PyTorch CNN models and a React Native mobile app with Expo.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [Backend Setup](#backend-setup)
  - [Mobile App Setup](#mobile-app-setup)
- [Running the Application](#running-the-application)
  - [Starting Backend](#starting-backend)
  - [Starting Mobile App](#starting-mobile-app)
- [API Endpoints](#api-endpoints)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**BMI Prediction Application** is an intelligent solution that analyzes images of individuals and predicts their:
- **Height** (in centimeters)
- **Weight** (in kilograms)
- **BMI Index** (Body Mass Index)
- **BMI Category** (Underweight, Normal, Overweight, Obese)

The application leverages deep learning models trained on extensive datasets and provides AI-generated health recommendations using Google's Gemini API.

---

## ✨ Features

- 📸 **Image-based Predictions**: Upload a person's photo to get BMI predictions
- 🧠 **Machine Learning**: PyTorch CNN models for accurate predictions
- 📱 **Cross-Platform**: iOS and Android support via React Native
- 🎨 **User-Friendly Interface**: Intuitive mobile app built with Expo
- 🤖 **AI Health Recommendations**: Google Gemini-powered personalized health advice
- 📊 **BMI Categorization**: Automatic classification (Underweight, Normal, Overweight, Obese)
- 🔄 **Real-time Processing**: Fast inference with GPU/CPU support
- 📡 **RESTful API**: Complete API documentation with Swagger UI

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.8+)
- **ML Framework**: PyTorch
- **Image Processing**: Pillow, NumPy
- **API Server**: Uvicorn
- **AI Integration**: Google Generative AI (Gemini)
- **Environment**: Python virtual environment

### Mobile App
- **Framework**: React Native
- **Development Platform**: Expo
- **UI Framework**: React Native Web
- **HTTP Client**: Axios
- **Build Tool**: EAS (Expo Application Services)

### Tools & Services
- **Version Control**: Git
- **API Documentation**: Swagger/OpenAPI
- **AI Models**: PyTorch CNN models
- **Health API**: Google Gemini API

---

## 📂 Project Structure

```
bmi-app-clone/
├── backend/                           # FastAPI Backend Server
│   ├── main.py                       # FastAPI application entry point
│   ├── requirements.txt               # Python dependencies
│   ├── predictor.py                  # Prediction logic and BMI calculations
│   ├── preprocess.py                 # Image preprocessing pipeline
│   ├── model_loader.py               # Model loading and management
│   ├── pose_estimator.py             # Pose estimation module
│   ├── gemini_service.py             # Google Gemini API integration
│   ├── .env.example                  # Example environment variables
│   ├── .env                          # Environment variables (local)
│   └── README.md                     # Backend documentation
│
├── mobile-app/                        # React Native + Expo App
│   ├── App.js                        # Main app component
│   ├── package.json                  # NPM dependencies
│   ├── app.json                      # Expo configuration
│   ├── eas.json                      # EAS build configuration
│   ├── android/                      # Android native code (generated)
│   ├── assets/                       # Images and static files
│   └── README.md                     # Mobile app documentation
│
├── models/                            # Machine Learning Models Directory
│   ├── model_ep_48.pth               # Height prediction model (PyTorch)
│   └── model_ep_37.pth               # Weight prediction model (PyTorch)
│
├── testing_images/                    # Sample test images
├── ARCHITECTURE.md                    # Detailed system architecture
└── README.md                          # This file

```

---

## 📋 Prerequisites

Before getting started, ensure you have the following installed:

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for dependencies and models

### Required Software

1. **Python 3.8 or higher**
   ```bash
   # Check version
   python --version
   ```
   
2. **Node.js 14+ and npm**
   ```bash
   # Check version
   node --version
   npm --version
   ```

3. **Git**
   ```bash
   git --version
   ```

4. **Expo CLI** (for mobile development)
   ```bash
   npm install -g expo-cli
   ```

### Optional
- **Android Studio** (for Android emulator)
- **Xcode** (for iOS development on macOS)
- **pip-tools** (for dependency management)

---

## 🔧 Installation & Setup

### Backend Setup

#### Step 1: Navigate to Backend Directory
```bash
cd backend
```

#### Step 2: Create Python Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- FastAPI, Uvicorn (web framework)
- PyTorch, TorchVision (ML models)
- Pillow, NumPy (image processing)
- python-dotenv (environment management)
- google-generativeai (Gemini API)

#### Step 4: Configure Environment Variables

Copy the example file and add your Gemini API key:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your-api-key-here
```

**Getting a Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create new API key"
3. Copy the key and paste it in `.env`

#### Step 5: Verify Backend Installation
```bash
# Test imports
python -c "import fastapi; import torch; import cv2; print('✓ All dependencies installed')"
```

---

### Mobile App Setup

#### Step 1: Navigate to Mobile App Directory
```bash
cd mobile-app
```

#### Step 2: Install Dependencies
```bash
npm install
```

This installs all required packages including:
- React Native, React
- Expo and Expo modules
- Axios (HTTP client)
- TypeScript (optional)

#### Step 3: Configure Backend URL

Edit `App.js` and update the API URL to match your backend server:

```javascript
// For local development (same machine)
const API_BASE_URL = 'http://localhost:8000';

// For physical devices on same network
const API_BASE_URL = 'http://192.168.1.100:8000'; // Your computer's IP
```

#### Step 4: Verify Mobile Setup
```bash
# Check Expo CLI
expo --version

# Verify npm packages
npm list react react-native
```

---

## ▶️ Running the Application

### Starting Backend

#### Option 1: Development Mode (Recommended)
```bash
# From backend directory (with virtual environment activated)
uvicorn main:app --reload

# Output:
# Uvicorn running on http://127.0.0.1:8000
# - API Docs (Swagger UI): http://127.0.0.1:8000/docs
# - ReDoc: http://127.0.0.1:8000/redoc
```

#### Option 2: Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Option 3: With Custom Port
```bash
uvicorn main:app --reload --port 5000
```

**Verify Backend is Running:**
- Open `http://localhost:8000/docs` in your browser
- You should see the Swagger UI with all API endpoints

---

### Starting Mobile App

#### Option 1: Expo Development Server (Recommended)
```bash
# From mobile-app directory
npm start
# or
expo start
```

This starts the Expo Metro bundler. You'll see:
```
Metro Server is running on port 19000.
Connect a device by typing scan the QR code above.
- iOS           press 'i'.
- Android       press 'a'.
- Web Browser   press 'w'.
```

#### Option 2: Android Emulator
```bash
# Make sure Android emulator is running first
npm run android
# or
expo run:android
```

#### Option 3: iOS Simulator (macOS only)
```bash
npm run ios
# or
expo run:ios
```

#### Option 4: Web Browser
```bash
npm run web
# or
expo start --web
```

#### Option 5: Physical Device
1. Install **Expo Go** app from App Store or Google Play
2. Run `expo start` or `npm start`
3. Scan the QR code with your device's camera
4. App will open in Expo Go

---

## 📡 API Endpoints

### Health Check
```
GET /
Response: {"message": "BMI Prediction API running"}
```

### Predict BMI from Image
```
POST /predict/
Content-Type: multipart/form-data

Request:
- image: <image_file> (JPEG, PNG, etc.)

Response:
{
  "height_cm": 175.5,
  "weight_kg": 72.3,
  "bmi": 23.5,
  "bmi_category": "Normal",
  "recommendation": "Your BMI is in the normal range...",
  "processing_time_ms": 245
}
```

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────┐
│    React Native + Expo Mobile App   │
│  (User selects image)               │
└──────────────┬──────────────────────┘
               │ HTTP POST (multipart/form-data)
               ↓
┌─────────────────────────────────────┐
│      FastAPI Backend Server         │
│  (http://localhost:8000)            │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ 1. Receive & Parse Image    │  │
│  └──────────────┬──────────────┘  │
│                 ↓                   │
│  ┌─────────────────────────────┐  │
│  │ 2. Preprocess Image         │  │
│  │    - Resize to 224x224      │  │
│  │    - Normalize              │  │
│  └──────────────┬──────────────┘  │
│                 ↓                   │
│  ┌─────────────────────────────┐  │
│  │ 3. Load ML Models           │  │
│  │    - Height Model           │  │
│  │    - Weight Model           │  │
│  └──────────────┬──────────────┘  │
│                 ↓                   │
│  ┌─────────────────────────────┐  │
│  │ 4. Run Inference            │  │
│  │    - Get Height (cm)        │  │
│  │    - Get Weight (kg)        │  │
│  └──────────────┬──────────────┘  │
│                 ↓                   │
│  ┌─────────────────────────────┐  │
│  │ 5. Calculate BMI            │  │
│  │ 6. Generate Recommendation  │  │
│  │    (Google Gemini API)      │  │
│  └──────────────┬──────────────┘  │
│                 ↓                   │
│  ┌─────────────────────────────┐  │
│  │ 7. Return JSON Response     │  │
│  └─────────────────────────────┘  │
└─────────────────────────────────────┘
               ↑ HTTP Response
               │
┌──────────────┴──────────────────────┐
│    Mobile App displays results      │
└─────────────────────────────────────┘
```

### Core Components

1. **Image Preprocessing** (`preprocess.py`)
   - Loads image from bytes
   - Converts to RGB
   - Resizes to 224×224 pixels
   - Normalizes using ImageNet statistics

2. **Model Inference** (`predictor.py`)
   - Loads PyTorch models (height & weight)
   - Runs inference on preprocessed image
   - Returns predictions in cm and kg

3. **BMI Calculation**
   - Formula: BMI = weight(kg) / height(m)²
   - Categorizes result (Underweight, Normal, Overweight, Obese)

4. **Health Recommendations** (`gemini_service.py`)
   - Integrates with Google Gemini API
   - Generates personalized health advice
   - Provides diet and exercise suggestions

---

## 🐛 Troubleshooting

### Backend Issues

#### Error: "ModuleNotFoundError: No module named 'fastapi'"
```bash
# Ensure virtual environment is activated
# Then reinstall dependencies
pip install -r requirements.txt
```

#### Error: "Could not find CUDA device"
```bash
# The system will automatically fallback to CPU
# Models will work, but inference will be slower
# No action needed
```

#### Error: Port 8000 already in use
```bash
# Use a different port
uvicorn main:app --reload --port 5000
# Update API_BASE_URL in App.js accordingly
```

#### Error: "GEMINI_API_KEY not found"
```bash
# Make sure .env file exists in backend directory with:
GEMINI_API_KEY=your-actual-key
# Or set environment variable
$env:GEMINI_API_KEY = "your-key"
```

### Mobile App Issues

#### Error: "Cannot find module 'axios'"
```bash
cd mobile-app
npm install axios
```

#### Error: "Network request failed" / "Cannot reach backend"
```bash
# 1. Verify backend is running: http://localhost:8000/docs
# 2. Check API_BASE_URL in App.js (use correct IP if on different machine)
# 3. Ensure both on same network
# 4. Check firewall settings
```

#### Error: "expo: command not found"
```bash
# Install Expo CLI globally
npm install -g expo-cli
```

#### App freezes when uploading image
```bash
# This is normal for large images (>5MB)
# The model is processing - wait for completion
# Consider optimizing image size before upload
```

### General Issues

#### "Image quality/predictions seem off"
1. Ensure images are clear and well-lit
2. Full body visible in frame
3. Good contrast with background
4. Image resolution at least 480×640

#### Backend and Mobile on Different Machines
```javascript
// In App.js, use your machine's IP instead of localhost
const API_BASE_URL = 'http://192.168.1.X:8000';

// Find your IP:
# Windows PowerShell
ipconfig | findstr "IPv4"

# macOS/Linux
ifconfig | grep inet
```

---

## 📝 Contributing

We welcome contributions! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Write clean, documented code
- Test thoroughly before submitting PR
- Follow Python PEP 8 for backend code
- Use meaningful commit messages
- Add comments for complex logic

### Areas for Contribution
- Model improvements (better accuracy)
- UI/UX enhancements
- Performance optimization
- Bug fixes
- Documentation improvements
- Translation support

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review API documentation at `http://localhost:8000/docs`
3. Open an issue on GitHub
4. Contact the development team

---

## 🙏 Acknowledgments

- PyTorch community for ML framework
- FastAPI for amazing web framework
- React Native & Expo for mobile platform
- Google Gemini for AI recommendations
- All contributors and testers

---

**Happy coding! 🚀 Start building amazing BMI predictions!**
