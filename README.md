# SMART-CROP-ADVISORY-SYSTEM
This project, likely titled SSGA (Soil Science & Growth Assistant), is a comprehensive AI-Powered Agricultural Management System designed to help farmers optimize their crops and soil health.

Based on my review of the code in d:\projects\soil, here are the core features:

1. 🌍 Smart Soil Analysis
AI Soil Predictor: Uses a PyTorch (MobileNetV2) model to classify soil types (like Alluvial, Black, Red, or Laterite) from uploaded photos.
Gemini Integration: Connected to Google Gemini AI to generate detailed, structured reports on soil texture, nutrient capacity, and step-by-step improvement methods for farmers.
Data Processing: Includes scripts to process sensor data (soil moisture and type) to provide automated irrigation and fertilization recommendations.

2. 🌿 Plant Health & Disease Detection
Disease Identifier: Uses TensorFlow/Keras to analyze crop leaves and identify specific diseases (e.g., Apple Scab, Tomato Blight) with confidence scores.
Growth Tracking: Features a module to monitor plant growth stages and provide care advice.

3. 🤖 Intelligent Farming Assistant
Agri-Chatbot: An AI chatbot integrated into the home page that answers farming questions, analyzes images of crops, and helps users navigate the various tools.
Routine Tracking: Connects to the Perenual API to fetch watering frequencies, sunlight requirements, and specific care instructions for different plant species.

4. 📊 Specialized Farm Tools
Yield Prediction: Models to estimate potential crop output.
Fertilizer Guidance: Recommends the right nutrients based on soil and crop data.
Weather Advisory: Provides farming tips based on local weather conditions.
Gov Schemes: A dedicated section to help farmers find relevant government support and subsidies.
Tech Stack Summary
Web: Django (Python) with a modern HTML/CSS frontend.
AI/ML: PyTorch, TensorFlow, Scikit-learn, and Google Generative AI (Gemini).
Data: Pandas, NumPy, and Matplotlib.
It’s essentially an all-in-one digital toolkit for modern, data-driven agriculture!
