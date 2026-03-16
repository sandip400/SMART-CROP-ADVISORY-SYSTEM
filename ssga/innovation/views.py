from __future__ import annotations
import markdown
import os
import json
import base64
from pathlib import Path
from typing import Optional

import requests
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from dotenv import load_dotenv

# Load .env file from project root
# BASE_DIR is Path(__file__).resolve().parent.parent.parent (d:\projects\soil)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# PyTorch image model deps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import google.generativeai as genai

# -------------------------
# Path resolution utilities
# -------------------------

APP_DIR = Path(__file__).resolve().parent

def resolve_path(p: str | os.PathLike) -> Path:
    """
    Resolve a weights file path.
    - Absolute path: returned as-is
    - Relative path: resolved relative to this app directory (where your .pth sits)
    """
    p = Path(p)
    return p if p.is_absolute() else (APP_DIR / p)


# -------------------------
# Configurable file paths
# -------------------------

TORCH_WEIGHTS_PATH = resolve_path(os.environ.get("TORCH_WEIGHTS_PATH", "soil_model.pth"))
CLASS_NAMES = ["Alluvial", "Black", "Red", "Laterite"]  # must match your training order

# Gemini configuration (server-side only)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
chatbot_model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# -------------------------
# Lazy singletons
# -------------------------

torch_model: Optional[torch.nn.Module] = None
torch_transform: Optional[object] = None  # flexible typing for transforms.Compose


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


def _load_torch_pipeline() -> None:
    """Load MobileNetV2 classification model and preprocessing once per process."""
    global torch_model, torch_transform

    if torch_model is None:
        _assert_exists(TORCH_WEIGHTS_PATH, "Image model weights (soil_model.pth)")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

        # Load custom weights while allowing final classifier mismatch
        state_dict = torch.load(TORCH_WEIGHTS_PATH, map_location=torch.device("cpu"))
        from collections import OrderedDict
        new_state = OrderedDict((k, v) for k, v in state_dict.items() if "classifier.1" not in k)
        model.load_state_dict(new_state, strict=False)
        model.eval()
        torch_model = model

    if torch_transform is None:
        torch_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


# -------------------------
# Gemini helper
# -------------------------

def generate_soil_description_with_gemini(soil_type: str) -> str:
    """
    Call Gemini to generate a comprehensive HTML description for the soil type.
    Returns HTML (no outer <html> wrapper) ready to be injected into the page.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    prompt = (
        f"Give me a comprehensive, structured, and detailed description of the soil type: {soil_type}. "
        "Include the following sections:\n"
        "1. Texture and physical characteristics\n"
        "2. Water retention and drainage properties\n"
        "3. Typical pH range\n"
        "4. Nutrient holding capacity (CEC) and fertility level\n"
        "5. Common advantages and disadvantages\n"
        "6. Crops best suited for this soil\n"
        "7. Step-by-step methods to improve this soil's quality for farming\n"
        "8. Extra tips for sustainable soil management\n"
        "9. Provide **statistical crop insights** (numerical facts such as average yield in quintals/hectare, % area coverage in India, major states producing these crops, contribution to national output). "
        "Make sure statistics are presented as clear bullet points with numbers.\n\n"
        "Write the response in a clear, organized format with headings and bullet points. "
        "Output STRICTLY as HTML with only these tags: h3,h4,ul,ol,li,p,strong,em,br. "
        "Do NOT include <html>, <head>, or <body> wrappers and do not use code fences."
    )


    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    resp = requests.post(GEMINI_ENDPOINT, headers=headers, params=params, json=body, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    # Gemini response parsing (v1beta)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # Fallback: provide the raw response for debugging
        raise RuntimeError(f"Gemini response parsing failed: {data}")

    # Light cleanup if the model returned Markdown/code fences
    text = text.replace("\`\`\`html", "").replace("\`\`\`", "").strip()
    return text

@require_http_methods(["POST"])
@csrf_protect
def chatbot_query(request: HttpRequest):
    """
    Handle chatbot queries with text, image, and navigation support.
    """
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        message_type = data.get('type', 'text')  # text, image, navigation
        image_data = data.get('image', None)
        
        if not user_message and not image_data:
            return JsonResponse({"ok": False, "error": "Message or image is required."}, status=400)

        # Define available pages for navigation
        available_pages = {
            'soil prediction': '/soil_predictor/',
            'soil predictor': '/soil_predictor/',
            'crop recommendation': '/croprecom3/',
            'crop suggest': '/croprecom3/',
            'weather advisory': '/advisory5/',
            'weather': '/advisory5/',
            'routine tracker': '/routinetrack4/',
            'calendar': '/routinetrack4/',
            'fertilizer guidance': '/fertilizer/',
            'fertilizer': '/fertilizer/',
            'pest detection': '/pltdis/',
            'disease detection': '/pltdis/',
            'yield prediction': '/aiyield/',
            'yield': '/aiyield/',
            'government schemes': '/gov/',
            'gov schemes': '/gov/',
            'plant growth': '/growth/',
            'growth analysis': '/growth/',
            'soil education': '/soiledu/',
            'education': '/soiledu/',
            'home': '/',
        }

        # Check if user is asking for navigation
        navigation_request = None
        user_lower = user_message.lower()
        for page_key, page_url in available_pages.items():
            if page_key in user_lower:
                navigation_request = {
                    'page': page_key.title(),
                    'url': page_url
                }
                break

        # Prepare system prompt for agriculture and soil focus
        system_prompt = """You are an expert agricultural assistant specializing in soil science, farming, and crop management. 
        Your knowledge covers:
        - Soil types, composition, and health assessment
        - Crop selection and rotation strategies
        - Fertilizer and irrigation management
        - Pest and disease identification and treatment
        - Weather-based farming decisions
        - Sustainable agriculture practices
        - Government agricultural schemes and support
        
        Always provide practical, actionable advice. Keep responses concise but informative.
        If asked about navigation, mention the available features but focus on agricultural guidance.
        """

        # Handle image input
        if image_data and message_type == 'image':
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                
                # Create prompt for image analysis
                image_prompt = f"{system_prompt}\n\nUser uploaded an image and asks: {user_message}\n\nPlease analyze this agricultural/soil image and provide relevant insights about soil condition, plant health, or farming practices visible in the image."
                
                # Use Gemini for image analysis
                response = chatbot_model.generate_content([
                    image_prompt,
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                
                bot_response = response.text
                
            except Exception as e:
                return JsonResponse({"ok": False, "error": f"Image analysis failed: {str(e)}"}, status=500)
        
        else:
            # Handle text input
            import markdown  # Add this import at the top if not already

# Handle text input
            full_prompt = f"""{system_prompt}

            User: {user_message}

            Respond in **HTML** using <h3>,<h4>,<ul>,<ol>,<li>,<p>,<strong>,<em>,<br>. 
            Do not return Markdown or code fences. Do not include <html>, <head>, or <body>.
            """
            try:
                response = chatbot_model.generate_content(full_prompt)
                bot_response = response.text.strip()

                # Fallback: if Gemini still gives Markdown, convert it
                bot_response = markdown.markdown(bot_response, extensions=['extra', 'tables', 'sane_lists'])
            except Exception as e:  
                return JsonResponse({"ok": False, "error": f"Chatbot error: {str(e)}"}, status=500)


        # Add navigation suggestion if relevant
        if navigation_request:
            bot_response += f"\n\n🔗 I can help you navigate to the {navigation_request['page']} page. Would you like me to take you there?"

        return JsonResponse({
            "ok": True,
            "response": bot_response,
            "navigation": navigation_request,
            "available_pages": list(available_pages.keys())
        })

    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON data."}, status=400)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Server error: {str(e)}"}, status=500)

# ---------------
# Page rendering
# ---------------
@require_http_methods(["GET"])
@ensure_csrf_cookie  # ensure CSRF cookie exists for subsequent fetch POSTs
def index(request: HttpRequest):
    # Preload image model in background; ignore errors here to keep page loading
    try:
        _load_torch_pipeline()
    except Exception:
        pass
    return render(request, "soil_predictor.html")


# -------------------------
# Image prediction view
# -------------------------
@require_http_methods(["POST"])
@csrf_protect
def soil_image_predict(request: HttpRequest):
    # Load model
    try:
        _load_torch_pipeline()
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Failed to load image model: {e}"}, status=500)

    # Validate file
    file = request.FILES.get("image")
    if not file:
        return JsonResponse({"ok": False, "error": "image file is required."}, status=400)

    # Open image
    try:
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Invalid image: {e}"}, status=400)

    # Predict soil type
    try:
        input_tensor = torch_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = torch_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Inference failed: {e}"}, status=500)

    # Ask Gemini for detailed description
    try:
        description_html = generate_soil_description_with_gemini(predicted_class)

        soil_metrics = {
        "ph_min": 6.0 if predicted_class == "Alluvial" else 5.5,
        "ph_max": 7.5 if predicted_class == "Alluvial" else 8.0,
        "cec_cmol_per_kg": 15 if predicted_class == "Alluvial" else 25,
        "water_holding_score": 7,
        "drainage_score": 6,
        "fertility_score": 8,
        "erosion_risk_score": 4,
        }

        mineral_composition = [
            {"label": "Nitrogen", "value": 25},
            {"label": "Phosphorus", "value": 15},
            {"label": "Potassium", "value": 30},
            {"label": "Organic Matter", "value": 20},
        ]

        prevalent_regions = ["Punjab", "Haryana", "Uttar Pradesh"]
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Gemini error: {e}"}, status=500)

    return JsonResponse(
        {
            "ok": True,
            "prediction": {
                "soil_type": predicted_class,
                "description_html": description_html,
                "metrics": soil_metrics,                # ✅ Added here
                "mineral_composition": mineral_composition,
                "prevalent_regions": prevalent_regions,
            },
        }
    )

def advisory5(request):
  return render(request, 'advisory5.html')


def croprecom3(request):
    return render(request,'croprecom3.html')


def routinetrack4(request):
  return render(request, 'routinetrack4.html')

def soil_im_predict(request):
  return render(request, 'soil_im_predict.html')

def soil_predictor(request):
  return render(request, 'soil_predictor.html')

def fertilizer(request):
  return render(request, 'fertilizer.html')

def aiyield(request):
  return render(request, 'aiyield.html')

def gov(request):
  return render(request, 'gov.html')

def growth(request):
  return render(request, 'growth.html')     

def pltdis(request):
  return render(request, 'pltdis.html')

def wastedis(request):
  return render(request, 'wastedis.html')

def home(request):
    if request.method == 'GET':
        return render(request, 'home.html')
    
    elif request.method == 'POST':
        # Handle chatbot queries
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            message_type = data.get('type', 'text')  # text, image, navigation, voice
            image_data = data.get('image', None)
            
            if not user_message and not image_data:
                return JsonResponse({"ok": False, "error": "Message or image is required."}, status=400)

            # Define available pages for navigation
            available_pages = {
                'soil prediction': '/soil_predictor/',
                'soil predictor': '/soil_predictor/',
                'crop recommendation': '/croprecom3/',
                'crop suggest': '/croprecom3/',
                'weather advisory': '/advisory5/',
                'weather': '/advisory5/',
                'routine tracker': '/routinetrack4/',
                'calendar': '/routinetrack4/',
                'fertilizer guidance': '/fertilizer/',
                'fertilizer': '/fertilizer/',
                'pest detection': '/pltdis/',
                'disease detection': '/pltdis/',
                'yield prediction': '/aiyield/',
                'yield': '/aiyield/',
                'government schemes': '/gov/',
                'gov schemes': '/gov/',
                'plant growth': '/growth/',
                'growth analysis': '/growth/',
                'soil education': '/soiledu/',
                'education': '/soiledu/',
                'home': '/',
            }

            # Check if user is asking for navigation
            navigation_request = None
            user_lower = user_message.lower()
            for page_key, page_url in available_pages.items():
                if page_key in user_lower:
                    navigation_request = {
                        'page': page_key.title(),
                        'url': page_url
                    }
                    break

            # Prepare system prompt
            system_prompt = """You are an expert agricultural assistant specializing in soil science, farming, and crop management. 
            Your knowledge covers:
            - Soil types, composition, and health assessment
            - Crop selection and rotation strategies
            - Fertilizer and irrigation management
            - Pest and disease identification and treatment
            - Weather-based farming decisions
            - Sustainable agriculture practices
            - Government agricultural schemes and support
            
            Always provide practical, actionable advice. Keep responses concise but informative.
            If asked about navigation, mention the available features but focus on agricultural guidance.
            """

            # Handle image input
            if image_data and message_type == 'image':
                try:
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)

                    image_prompt = (
                        f"{system_prompt}\n\n"
                        f"User uploaded an image and asks: {user_message}\n\n"
                        "Please analyze this agricultural/soil image and provide relevant insights about "
                        "soil condition, plant health, or farming practices visible in the image."
                    )

                    response = chatbot_model.generate_content([
                        image_prompt,
                        {"mime_type": "image/jpeg", "data": image_bytes}
                    ])
                    bot_response = response.text

                except Exception as e:
                    return JsonResponse({"ok": False, "error": f"Image analysis failed: {str(e)}"}, status=500)

            else:
                # Handle text input
                full_prompt = f"""{system_prompt}

                User: {user_message}

                Respond in well-structured **HTML** using <h3>,<h4>,<ul>,<ol>,<li>,<p>,<strong>,<em>,<br>.
                Do not include <html>, <head>, or <body> tags.
                Do not return Markdown or code fences.
                """
                try:
                    response = chatbot_model.generate_content(full_prompt)
                    bot_response = response.text.strip()
                    bot_response = markdown.markdown(bot_response, extensions=['extra', 'tables', 'sane_lists'])
                except Exception as e:
                    return JsonResponse({"ok": False, "error": f"Chatbot error: {str(e)}"}, status=500)

            # Add navigation suggestion if relevant
            if navigation_request:
                bot_response += (
                    f"\n\n🔗 I can help you navigate to the {navigation_request['page']} page. "
                    "Would you like me to take you there?"
                )

            # ✅ Build final response payload
            response_payload = {
                "ok": True,
                "response": bot_response,
                "navigation": navigation_request,
                "available_pages": list(available_pages.keys())
            }

            # ✅ Mark if this was triggered by voice so frontend can auto-redirect
            if message_type == "voice":
                response_payload["voice_triggered"] = True

            return JsonResponse(response_payload)

        except json.JSONDecodeError:
            return JsonResponse({"ok": False, "error": "Invalid JSON data."}, status=400)
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"Server error: {str(e)}"}, status=500)


def soiledu(request):
  return render(request, 'soiledu.html')
