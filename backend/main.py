"""
BinBot FastAPI Backend Server
=================================
RESTful API + WebSocket streaming for React Native mobile app

Features:
- JWT Authentication (Super Admin & Staff roles)
- Real-time YOLO detection streaming
- Analytics & Statistics endpoints
- Hardware control (Camera, ESP32, Sensors)
- CORS enabled for mobile access
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import jwt
import cv2
import numpy as np
import asyncio
import json
import logging
from pathlib import Path
import sys
import io
import base64
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent / "02_Dashboard_App"))

# Lazy imports to avoid Python 3.13 + torch issues on startup
# from modules.ai_logic import WasteDetector, ProductionVideoThread
# from modules.serial_logic import get_esp32_bridge, get_ultrasonic_thread
# from modules.database_logic import get_data_logger

# Configuration
SECRET_KEY = "binbot_secret_key_2024_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Credentials (same as Streamlit version)
SUPER_ADMIN_USERNAME = "admin"
SUPER_ADMIN_PASSWORD = "superbinbot2024"
STAFF_USERNAME = "staff"
STAFF_PASSWORD = "viewer2024"

# Initialize FastAPI
app = FastAPI(
    title="BinBot API - UIGM",
    description="Smart Waste Classification System API - Universitas Indo Global Mandiri",
    version="1.0.0"
)

# Mount static files for assets (UIGM Logo)
assets_path = Path(__file__).parent.parent.parent / "02_Dashboard_App" / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
    logger.info(f"✅ Static assets mounted from: {assets_path}")
else:
    logger.warning(f"⚠️ Assets directory not found: {assets_path}")

# CORS Configuration - Allow mobile devices on local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile access
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# OAuth2 Configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Logging - Suppress asyncio ConnectionResetError warnings on Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific asyncio errors that are not critical
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Global instances
video_thread = None
esp32_bridge = None
ultrasonic_thread = None
data_logger = None

# ============================================
# Pydantic Models
# ============================================

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class User(BaseModel):
    username: str
    role: str  # "super_admin" or "staff"

class CameraControl(BaseModel):
    action: str  # "start" or "stop"

class StatsResponse(BaseModel):
    anorganic: int
    metal: int
    organic: int
    total: int
    timestamp: str

class SensorData(BaseModel):
    anorganic_level: float
    metal_level: float
    organic_level: float
    timestamp: str

# ============================================
# Authentication Functions
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> User:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return User(username=username, role=role)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from token"""
    return verify_token(token)

async def get_super_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require Super Admin role"""
    if current_user.role != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super Admin access required"
        )
    return current_user

# ============================================
# Startup & Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global data_logger, esp32_bridge, ultrasonic_thread
    
    # Lazy import modules here to avoid Python 3.13 + torch initialization issues
    from modules.database_logic import get_data_logger
    from modules.serial_logic import get_esp32_bridge
    
    logger.info("🚀 Starting BinBot API Server...")
    
    # Initialize data logger
    data_logger = get_data_logger()
    logger.info("✅ Data logger initialized")
    
    # Initialize ESP32 bridge (optional - may not be connected)
    try:
        esp32_bridge = get_esp32_bridge()
        if esp32_bridge and esp32_bridge.is_connected:
            logger.info("✅ ESP32 bridge connected")
        else:
            logger.warning("⚠️ ESP32 bridge not connected (optional)")
    except Exception as e:
        logger.warning(f"⚠️ ESP32 bridge initialization failed: {e}")
    
    # Initialize ultrasonic sensors (optional) - Skip for now to prevent crashes
    try:
        # Disabled ultrasonic thread to prevent serial port issues
        # ultrasonic_thread = get_ultrasonic_thread()
        # if ultrasonic_thread:
        #     logger.info("✅ Ultrasonic sensors initialized")
        logger.info("⚠️ Ultrasonic sensors disabled (optional hardware)")
    except Exception as e:
        logger.warning(f"⚠️ Ultrasonic sensors initialization failed: {e}")
    
    logger.info("🎉 BinBot API Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_thread, esp32_bridge, ultrasonic_thread
    
    logger.info("🛑 Shutting down BinBot API Server...")
    
    # Stop camera
    if video_thread:
        try:
            if hasattr(video_thread, 'stop'):
                video_thread.stop()
            logger.info("✅ Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    # Close ESP32 connection only if connected
    if esp32_bridge:
        try:
            if hasattr(esp32_bridge, 'is_connected') and esp32_bridge.is_connected:
                if hasattr(esp32_bridge, 'disconnect'):
                    esp32_bridge.disconnect()
                elif hasattr(esp32_bridge, 'close'):
                    esp32_bridge.close()
                logger.info("✅ ESP32 connection closed")
            else:
                logger.info("ℹ️ ESP32 was not connected")
        except Exception as e:
            logger.error(f"Error closing ESP32: {e}")
    
    # Stop ultrasonic thread
    if ultrasonic_thread:
        try:
            if hasattr(ultrasonic_thread, 'stop'):
                ultrasonic_thread.stop()
            logger.info("✅ Ultrasonic sensors stopped")
        except Exception as e:
            logger.error(f"Error stopping ultrasonic: {e}")
    
    logger.info("👋 BinBot API Server stopped")

# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "BinBot API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "auth": "/login (POST)",
            "stream": "/stream (WebSocket)",
            "stats": "/stats (GET)",
            "sensors": "/sensors (GET)",
            "control": "/control (POST - Super Admin only)",
            "history": "/history (GET)",
            "health": "/health (GET)"
        }
    }

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token
    
    Credentials:
    - Super Admin: admin / superbinbot2024
    - Staff: staff / viewer2024
    """
    # Validate credentials
    if form_data.username == SUPER_ADMIN_USERNAME and form_data.password == SUPER_ADMIN_PASSWORD:
        role = "super_admin"
    elif form_data.username == STAFF_USERNAME and form_data.password == STAFF_PASSWORD:
        role = "staff"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "role": role},
        expires_delta=access_token_expires
    )
    
    logger.info(f"✅ Login successful: {form_data.username} ({role})")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": role,
        "username": form_data.username
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - Returns server status for mobile connectivity monitoring"""
    global video_thread, esp32_bridge, ultrasonic_thread
    
    try:
        import socket
        
        # Auto-detect server's local IP address
        hostname = socket.gethostname()
        try:
            # Get local IP by connecting to external address (doesn't actually send data)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = socket.gethostbyname(hostname)
        
        camera_running = False
        if video_thread is not None:
            if hasattr(video_thread, 'running'):
                camera_running = video_thread.running
        
        esp32_connected = False
        if esp32_bridge is not None:
            if hasattr(esp32_bridge, 'is_connected'):
                try:
                    esp32_connected = esp32_bridge.is_connected
                except:
                    esp32_connected = False
        
        sensors_running = False
        if ultrasonic_thread is not None:
            if hasattr(ultrasonic_thread, 'running'):
                try:
                    sensors_running = ultrasonic_thread.running
                except:
                    sensors_running = False
        
        return {
            "status": "healthy",
            "server": "BinBot API v1.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime": "OK",
            "server_ip": local_ip,
            "server_url": f"http://{local_ip}:8000",
            "hostname": hostname,
            "services": {
                "camera": camera_running,
                "esp32": esp32_connected,
                "sensors": sensors_running,
                "database": data_logger is not None
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",
            "server": "BinBot API v1.0.0",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/camera/frame")
async def get_camera_frame(current_user: User = Depends(get_current_user)):
    """
    Get current camera frame as base64 JPEG
    Simple polling endpoint for mobile app
    """
    global video_thread
    
    if not video_thread or not hasattr(video_thread, 'running') or not video_thread.running:
        return {
            "status": "camera_not_running",
            "message": "Camera is not active. Start camera from Settings.",
            "image": None
        }
    
    try:
        # Get latest frame from video thread
        if hasattr(video_thread, 'get_latest_frame'):
            frame = video_thread.get_latest_frame()
            if frame is not None:
                # Apply AI detection if detector is available
                detection_info = None
                processed_frame = frame
                
                if hasattr(video_thread, 'detector') and video_thread.detector:
                    try:
                        # Run detection on the frame
                        processed_frame, det_info = video_thread.detector.detect(frame, conf_threshold=0.5)
                        
                        # Log to database if new object detected
                        if det_info.get("classified") and data_logger:
                            # Check if this is a new detection (not just processing)
                            if "New" in det_info.get("status", ""):
                                try:
                                    data_logger.log_classification(
                                        waste_type=det_info["classified"],
                                        confidence=det_info.get("confidence", 0.0),
                                        waste_level=0.0,
                                        device_status="Online",
                                        processing_time_ms=det_info.get("inference_time_ms", 0.0)
                                    )
                                    logger.info(f"📝 Logged {det_info['classified']} to database")
                                except Exception as log_err:
                                    logger.warning(f"Failed to log classification: {log_err}")
                        
                        detection_info = {
                            "current_object": video_thread.detector.current_detected_object,
                            "status": video_thread.detector.detection_status,
                            "counts": video_thread.detector.waste_counts.copy(),
                            "classified": det_info.get("classified"),
                            "confidence": det_info.get("confidence", 0.0)
                        }
                    except Exception as e:
                        logger.warning(f"Detection error: {e}")
                        # If detection fails, use original frame
                        processed_frame = frame
                
                # Encode frame to JPEG with higher quality
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpg_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "status": "ok",
                    "image": jpg_base64,
                    "timestamp": datetime.now().isoformat(),
                    "detection": detection_info
                }
        
        return {
            "status": "no_frame",
            "message": "No frame available",
            "image": None
        }
    except Exception as e:
        logger.error(f"❌ Failed to get camera frame: {e}")
        return {
            "status": "error",
            "message": str(e),
            "image": None
        }

@app.post("/control")
async def control_camera(
    control: CameraControl,
    current_user: User = Depends(get_super_admin)
):
    """
    Control camera (Start/Stop) - Super Admin only
    
    Actions:
    - start: Start camera with YOLO detection
    - stop: Stop camera
    """
    global video_thread
    
    if control.action == "start":
        if video_thread and hasattr(video_thread, 'running') and video_thread.running:
            return {"message": "Camera already running", "status": "running"}
        
        try:
            # Lazy import here to avoid initialization issues
            from modules.ai_logic import ProductionVideoThread, WasteDetector
            
            logger.info("🔄 Initializing YOLO detector...")
            # Initialize detector first
            detector = WasteDetector()
            if not detector.is_loaded:
                logger.error("❌ YOLO model failed to load")
                raise Exception("Failed to load YOLO model. Check if best.pt exists in the correct location.")
            
            logger.info("✅ YOLO detector loaded successfully")
            logger.info("🎥 Initializing camera...")
            
            # Initialize camera with external camera (index 1) or fallback to 0
            video_thread = ProductionVideoThread(camera_index=1, resolution=(640, 480))
            
            # Attach detector to video thread
            video_thread.detector = detector
            
            if video_thread.initialize_camera(timeout=5.0):
                video_thread.start()
                logger.info(f"✅ Camera started with AI detection by {current_user.username}")
                return {
                    "message": "Camera started successfully with AI detection", 
                    "status": "running",
                    "camera_index": video_thread.camera_index
                }
            else:
                # Try with default camera
                logger.warning("⚠️ External camera not found, trying default camera...")
                video_thread = ProductionVideoThread(camera_index=0, resolution=(640, 480))
                video_thread.detector = detector
                if video_thread.initialize_camera(timeout=5.0):
                    video_thread.start()
                    logger.info(f"✅ Camera started (fallback) with AI detection by {current_user.username}")
                    return {
                        "message": "Camera started successfully (default camera) with AI detection", 
                        "status": "running",
                        "camera_index": video_thread.camera_index
                    }
                else:
                    logger.error("❌ No camera found on any index")
                    raise Exception("Camera not found. Please check if camera is connected and not used by another application.")
        except Exception as e:
            video_thread = None
            logger.error(f"❌ Camera start failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")
    
    elif control.action == "stop":
        if video_thread and hasattr(video_thread, 'running') and video_thread.running:
            video_thread.stop()
            video_thread = None
            logger.info(f"✅ Camera stopped by {current_user.username}")
            return {"message": "Camera stopped successfully", "status": "stopped"}
        else:
            return {"message": "Camera not running", "status": "stopped"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'start' or 'stop'")

@app.get("/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    """
    Get waste classification statistics
    
    Returns today's detection counts by category
    """
    global data_logger
    
    if not data_logger:
        raise HTTPException(status_code=500, detail="Data logger not initialized")
    
    try:
        # Try to get stats, fallback to safe defaults
        if hasattr(data_logger, 'get_today_stats'):
            stats = data_logger.get_today_stats()
            return {
                "anorganic": stats.get("anorganic_count", 0),
                "metal": stats.get("metal_count", 0),
                "organic": stats.get("organic_count", 0),
                "total": stats.get("total_classifications", 0),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Return safe defaults if method not available
            return {
                "anorganic": 0,
                "metal": 0,
                "organic": 0,
                "total": 0,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        # Return safe defaults instead of error
        return {
            "anorganic": 0,
            "metal": 0,
            "organic": 0,
            "total": 0,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/sensors")
async def get_sensor_data(current_user: User = Depends(get_current_user)):
    """
    Get ultrasonic sensor data (waste bin levels)
    
    Returns fill levels for each waste category (0-100%)
    """
    global ultrasonic_thread
    
    if not ultrasonic_thread:
        # Return mock data if sensors not available
        return {
            "anorganic_level": 0.0,
            "metal_level": 0.0,
            "organic_level": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "sensors_not_connected"
        }
    
    try:
        # Try to get latest data, fallback to safe defaults
        if hasattr(ultrasonic_thread, 'get_latest_data'):
            data = ultrasonic_thread.get_latest_data()
            return {
                "anorganic_level": data.get("anorganic", 0.0),
                "metal_level": data.get("metal", 0.0),
                "organic_level": data.get("organic", 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": "ok"
            }
        else:
            # Return mock data if method not available
            return {
                "anorganic_level": 0.0,
                "metal_level": 0.0,
                "organic_level": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "method_not_available"
            }
    except Exception as e:
        logger.error(f"❌ Failed to get sensor data: {e}")
        # Return mock data instead of error
        return {
            "anorganic_level": 0.0,
            "metal_level": 0.0,
            "organic_level": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

@app.get("/history")
async def get_history(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Get detection history for the last N days
    
    Returns daily statistics for charts
    """
    global data_logger
    
    if not data_logger:
        raise HTTPException(status_code=500, detail="Data logger not initialized")
    
    try:
        # Try to get history data, fallback to safe defaults
        if hasattr(data_logger, 'df'):
            # Get analytics data
            df = data_logger.df
            
            # Filter by date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Group by date and category
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df_filtered = df[pd.to_datetime(df['timestamp']) >= start_date]
            
            # Aggregate data
            daily_stats = df_filtered.groupby(['date', 'waste_type']).size().unstack(fill_value=0).reset_index()
            daily_stats['date'] = daily_stats['date'].astype(str)
            
            return {
                "data": daily_stats.to_dict(orient='records'),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        else:
            # Return empty data if df not available
            return {
                "data": [],
                "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
                "end_date": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"❌ Failed to get history: {e}")
        # Return empty data instead of error
        return {
            "data": [],
            "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
            "end_date": datetime.now().isoformat()
        }

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream with YOLO detection
    
    Client must send authentication token as first message
    """
    global video_thread
    
    await websocket.accept()
    
    try:
        # Wait for authentication token
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)
        token = auth_data.get("token")
        
        if not token:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return
        
        # Verify token
        try:
            user = verify_token(token)
            logger.info(f"✅ WebSocket connected: {user.username}")
        except HTTPException:
            await websocket.send_json({"error": "Invalid token"})
            await websocket.close()
            return
        
        # Check if camera is running
        if not video_thread or not hasattr(video_thread, 'running') or not video_thread.running:
            await websocket.send_json({
                "status": "camera_not_running",
                "message": "Camera not started. Super Admin needs to start it."
            })
            await websocket.close()
            return
        
        # Stream video frames
        while True:
            try:
                # Get latest frame
                frame_data = video_thread.get_latest_frame()
                
                if frame_data and frame_data.get('frame') is not None:
                    frame = frame_data['frame']
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpg_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame with detection info
                    await websocket.send_json({
                        "type": "frame",
                        "image": jpg_base64,
                        "detection": frame_data.get('detection'),
                        "fps": frame_data.get('fps', 0),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # Send status update
                    await websocket.send_json({
                        "type": "status",
                        "message": "Waiting for camera...",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Control frame rate (15 FPS)
                await asyncio.sleep(1/15)
                
            except WebSocketDisconnect:
                logger.info(f"✅ WebSocket disconnected: {user.username}")
                break
            except Exception as e:
                logger.error(f"❌ Stream error: {e}")
                await websocket.send_json({"error": str(e)})
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await websocket.close()

# ============================================
# PDF Export Endpoint (Super Admin Only)
# ============================================

@app.get("/export-pdf")
async def export_pdf(
    lang: str = "en",
    current_user: User = Depends(get_super_admin)
):
    """
    Generate PDF report with UIGM branding - Super Admin only
    
    Parameters:
    - lang: Language code ("en" or "id")
    """
    try:
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E7D32'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Translations
        translations = {
            "en": {
                "title": "BinBot Waste Classification Report",
                "institution": "Universitas Indo Global Mandiri",
                "developer": "Developed by: Satria",
                "generated": "Report Generated",
                "summary": "Waste Classification Summary",
                "organic": "Organic Waste",
                "anorganic": "Anorganic Waste",
                "metal": "Metal Waste",
                "total": "Total Waste Classified",
                "activity": "Recent Activity Log",
                "timestamp": "Timestamp",
                "category": "Category",
                "count": "Count",
                "footer": "This report is generated automatically by BinBot AI System"
            },
            "id": {
                "title": "Laporan Klasifikasi Sampah BinBot",
                "institution": "Universitas Indo Global Mandiri",
                "developer": "Dikembangkan oleh: Satria",
                "generated": "Laporan Dibuat",
                "summary": "Ringkasan Klasifikasi Sampah",
                "organic": "Sampah Organik",
                "anorganic": "Sampah Anorganik",
                "metal": "Sampah Logam",
                "total": "Total Sampah Terklasifikasi",
                "activity": "Log Aktivitas Terkini",
                "timestamp": "Waktu",
                "category": "Kategori",
                "count": "Jumlah",
                "footer": "Laporan ini dibuat otomatis oleh Sistem AI BinBot"
            }
        }
        
        text = translations.get(lang, translations["en"])
        
        # Add UIGM Logo if exists
        logo_path = assets_path / "wordmark_dan_tampilan_halaman_login.png"
        if logo_path.exists():
            try:
                logo = RLImage(str(logo_path), width=3*inch, height=1*inch)
                elements.append(logo)
                elements.append(Spacer(1, 20))
            except:
                pass
        
        # Title
        elements.append(Paragraph(text["title"], title_style))
        elements.append(Spacer(1, 12))
        
        # Institution & Developer info
        elements.append(Paragraph(f"<b>{text['institution']}</b>", styles['Normal']))
        elements.append(Paragraph(text["developer"], styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Generation timestamp
        elements.append(Paragraph(
            f"{text['generated']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        # Get statistics
        stats_data = {}
        if data_logger:
            try:
                stats_data = data_logger.get_totals()
                logger.info(f"PDF Export - Stats data: {stats_data}")
            except Exception as e:
                logger.error(f"PDF Export - Error getting totals: {e}")
                stats_data = {"anorganic": 0, "metal": 0, "organic": 0}
        else:
            stats_data = {"anorganic": 0, "metal": 0, "organic": 0}
        
        # Calculate total with explicit type conversion
        # Robust total calculation with error handling
        def safe_int(val):
            try:
                return int(val)
            except Exception:
                return 0

        total_waste = (
            safe_int(stats_data.get("anorganic", 0)) +
            safe_int(stats_data.get("metal", 0)) +
            safe_int(stats_data.get("organic", 0))
        )
        logger.info(f"PDF Export - Total waste calculated: {total_waste} (anorganic={stats_data.get('anorganic', 0)}, metal={stats_data.get('metal', 0)}, organic={stats_data.get('organic', 0)})")
        
        # Summary section
        elements.append(Paragraph(text["summary"], heading_style))
        
        # Statistics table
        stats_table_data = [
            [text["category"], text["count"]],
            [text["organic"], str(stats_data.get("organic", 0))],
            [text["anorganic"], str(stats_data.get("anorganic", 0))],
            [text["metal"], str(stats_data.get("metal", 0))],
            [f"<b>{text['total']}</b>", f"<b>{str(total_waste)}</b>"]
        ]
        
        stats_table = Table(stats_table_data, colWidths=[4*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8F5E9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 20))
        
        # Activity log section
        elements.append(Paragraph(text["activity"], heading_style))
        
        activity_data = [[text["timestamp"], text["category"]]]
        
        # Get recent activity (last 10 entries)
        if data_logger:
            try:
                recent_logs = data_logger.get_recent_logs(limit=10)
                for log in recent_logs:
                    activity_data.append([
                        log.get("timestamp", "N/A"),
                        log.get("category", "Unknown")
                    ])
            except:
                activity_data.append(["No activity", "N/A"])
        else:
            activity_data.append(["No data available", "N/A"])
        
        activity_table = Table(activity_data, colWidths=[3*inch, 3*inch])
        activity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E3F2FD')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(activity_table)
        elements.append(Spacer(1, 30))
        
        # Footer
        elements.append(Paragraph(text["footer"], styles['Italic']))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Generate filename with timestamp
        filename = f"BinBot_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        logger.info(f"✅ PDF generated by {current_user.username}: {filename}")
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
