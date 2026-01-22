import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import base64
from utils import (
    get_recipe_from_spoonacular, 
    get_nutrition_from_usda, 
    get_fun_facts,
    combine_nutrition_data,
    get_cache_stats
)

# Page configuration
st.set_page_config(
    page_title="AI Food Recognition Studio",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to show party popper only once
def show_party_popper():
    """Show confetti animation only when food is detected"""
    st.components.v1.html("""
        <style>
        @keyframes confetti-fall {
            0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
            100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
        }
        .confetti {
            position: fixed;
            width: 10px;
            height: 10px;
            z-index: 9999;
            animation: confetti-fall 3s linear;
            pointer-events: none;
        }
        </style>
        <div id="confetti-container"></div>
        <script>
        const colors = ['#14B8A6', '#0D9488', '#5EEAD4', '#2DD4BF', '#99F6E4'];
        const confettiCount = 50;
        const container = document.getElementById('confetti-container');
        
        for(let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + 'vw';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.animationDelay = Math.random() * 3 + 's';
            confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
            container.appendChild(confetti);
            
            setTimeout(() => confetti.remove(), 5000);
        }
        </script>
    """, height=0)

# Enhanced CSS with better colors and background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@700&family=Poppins:wght@600;700&display=swap');
    
    /* Main Background with image */
    .stApp {
        background-image: 
            linear-gradient(rgba(255, 248, 240, 0.92), rgba(255, 245, 235, 0.92)),
            url('https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=1920');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle Animations */
    @keyframes glow-subtle {
        0%, 100% { box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2); }
        50% { box-shadow: 0 6px 16px rgba(20, 184, 166, 0.3); }
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Header Styling - Clean and Modern */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 50%, #0F766E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Recipe Card - Subtle */
    .recipe-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(20, 184, 166, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .recipe-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(20, 184, 166, 0.15);
        border-color: rgba(20, 184, 166, 0.3);
    }
    
    /* Ingredient Cards - Teal Theme */
    .ingredient-card {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        border-left: 4px solid #14B8A6;
        padding: 0.9rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #115E59;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(20, 184, 166, 0.1);
    }
    
    .ingredient-card:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 8px rgba(20, 184, 166, 0.2);
    }
    
    /* Step Card - Grey/Teal */
    .step-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        border: 2px solid #CBD5E1;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        color: #334155;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(71, 85, 105, 0.1);
    }
    
    .step-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
        border-color: #14B8A6;
    }
    
    /* Metric Cards - No floating */
    .metric-card {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid #5EEAD4;
        color: #115E59;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.15);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.25);
    }
    
    /* Fun Fact Card */
    .fun-fact-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        border: 2px solid #94A3B8;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        color: #334155;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(71, 85, 105, 0.15);
    }
    
    .fun-fact-card::after {
        content: '✨';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 2.5rem;
        opacity: 0.2;
    }
    
    .fun-fact-card:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.25);
    }
    
    /* Nutrition Badge - Teal */
    .nutrition-badge {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        color: #0F766E;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        transition: all 0.2s ease;
        border: 2px solid #5EEAD4;
        box-shadow: 0 2px 6px rgba(20, 184, 166, 0.1);
    }
    
    .nutrition-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.25);
    }
    
    /* Button Styling - Teal */
    .stButton>button {
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.35);
        background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
    }
    
    /* Prediction Card - Subtle Shine */
    .prediction-item {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.3rem;
        margin: 0.8rem 0;
        border: 2px solid rgba(20, 184, 166, 0.2);
        transition: all 0.2s ease;
        color: #334155;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: shine 4s infinite;
    }
    
    .prediction-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
        border-color: #14B8A6;
    }
    
    /* Sidebar Styling - Grey/Teal */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%);
        border-right: 2px solid rgba(20, 184, 166, 0.15);
    }
    
    /* Tab Styling - Teal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.08), rgba(13, 148, 136, 0.08));
        border-radius: 12px;
        padding: 0.6rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        color: #64748B;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.7rem 1.3rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.9);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #14B8A6, #0D9488);
        color: white;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.3);
    }
    
    /* Upload Area - Teal */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.8);
        border: 2px dashed #14B8A6;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0D9488;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Text Color */
    .stMarkdown, p, label {
        color: #334155 !important;
    }
    
    /* Progress Bar - Teal */
    .stProgress > div > div {
        background: linear-gradient(90deg, #14B8A6, #0D9488);
        box-shadow: 0 0 8px rgba(20, 184, 166, 0.3);
    }
    
    /* Step Number Badge - No Float */
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #14B8A6, #0D9488);
        color: white;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        text-align: center;
        line-height: 45px;
        font-weight: bold;
        margin-right: 15px;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.3);
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #334155 !important;
    }
    
    /* Welcome Badge - Teal */
    .welcome-badge {
        display: inline-block;
        background: linear-gradient(135deg, #14B8A6, #0D9488);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.25);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    if os.path.exists('food_recognition_model.keras'):
        model = tf.keras.models.load_model('food_recognition_model.keras')
        return model
    elif os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
        return model
    else:
        return None

@st.cache_data
def load_class_labels():
    """Load class labels"""
    if os.path.exists('class_labels.json'):
        with open('class_labels.json', 'r') as f:
            return json.load(f)
    return None

def preprocess_image(image, img_size=128):
    """Preprocess image for model prediction"""
    img = image.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_dish(model, image, class_labels, top_k=3):
    """Predict dish from image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        class_name = class_labels[str(idx)]
        confidence = predictions[0][idx] * 100
        results.append({
            'name': class_name,
            'confidence': confidence
        })
    
    return results

def format_nutrient_value(nutrient_data):
    """Safely format nutrient data regardless of structure"""
    try:
        if isinstance(nutrient_data, dict):
            if 'amount' in nutrient_data and 'unit' in nutrient_data:
                return f"{nutrient_data['amount']} {nutrient_data['unit']}"
            elif 'value' in nutrient_data and 'unit' in nutrient_data:
                return f"{nutrient_data['value']} {nutrient_data['unit']}"
            else:
                return str(nutrient_data)
        else:
            return str(nutrient_data)
    except Exception:
        return "N/A"

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ AI Food Recognition Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover recipes, nutrition, and culinary secrets with cutting-edge AI</p>', unsafe_allow_html=True)
    
    # Load model and labels
    model = load_model()
    class_labels = load_class_labels()
    
    if model is None or class_labels is None:
        st.error("⚠️ **Model not found!** Please train the model first.")
        st.info("📝 Run: `python train_model.py` to train the model (1-3 hours)")
        st.code("python train_model.py", language="bash")
        st.stop()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'current_fact' not in st.session_state:
        st.session_state.current_fact = 0
    if 'food_detected' not in st.session_state:
        st.session_state.food_detected = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Control Panel")
        st.markdown("---")
        
        st.markdown("#### 📊 Model Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Classes", len(class_labels))
        with col2:
            st.metric("📸 Size", "128×128")
        
        st.markdown("---")
        
        st.markdown("#### ⚙️ Prediction Settings")
        show_top_predictions = st.slider("🔝 Top predictions", 1, 5, 3)
        confidence_threshold = st.slider("📊 Confidence threshold", 0, 100, 20)
        
        st.markdown("---")
        
        st.markdown("#### 🚀 Quick Stats")
        stats = get_cache_stats()
        st.metric("💾 Cached Recipes", stats['recipes_cached'])
        st.metric("🥗 Cached Nutrition", stats['nutrition_cached'])
        
        st.markdown("---")
        
        st.markdown("#### 💡 Pro Tips")
        st.info("📷 Use well-lit, centered images")
        st.success("🎯 Bird's eye view works best")
        st.warning("✨ Clear background for accuracy")
    
    # File uploader - NEW LAYOUT
    st.markdown('<div class="welcome-badge">📤 Upload Your Food Image Below</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of your dish",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # NEW LAYOUT - Image on top, predictions below
        st.markdown("---")
        
        # Full width image display - SMALLER
        st.markdown("### 📸 Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        
        # Smaller centered image
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, use_container_width=True, caption="🔍 Analyzing...")
        
        st.markdown("---")
        
        # AI Analysis Section
        st.markdown("### 🎯 AI Prediction Results")
        
        with st.spinner("🔮 AI is thinking..."):
            predictions = predict_dish(model, image, class_labels, top_k=show_top_predictions)
        
        # Display predictions in a grid layout
        pred_cols = st.columns(min(len(predictions), 3))
        
        for i, pred in enumerate(predictions):
            if pred['confidence'] >= confidence_threshold:
                with pred_cols[i % 3]:
                    rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.markdown(f"""
                        <div class="prediction-item">
                            <h2 style="text-align: center;">{rank_emoji}</h2>
                            <h3 style="text-align: center;">{pred['name']}</h3>
                            <p style="font-size: 1.3rem; color: #FF6B6B; text-align: center; font-weight: bold;">
                                {pred['confidence']:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(pred['confidence'] / 100)
        
        # Get top prediction
        top_prediction = predictions[0]
        dish_name = top_prediction['name']
        
        if top_prediction['confidence'] >= confidence_threshold:
            # Show party popper only once when food is detected
            if not st.session_state.food_detected:
                show_party_popper()
                st.session_state.food_detected = True
            
            st.markdown("---")
            
            st.markdown(f"""
                <div class="recipe-card">
                    <h1 style="text-align: center; color: #FF6B6B; font-size: 2.5rem;">✨ {dish_name}</h1>
                    <p style="text-align: center; color: #5D6D7E; font-size: 1.3rem; margin-top: 1rem;">
                        Confidence Score: <span style="color: #FF6B6B; font-weight: bold;">{top_prediction['confidence']:.2f}%</span> 🎯
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create tabs - MORE VISUAL
            tab1, tab2, tab3, tab4 = st.tabs([
                "📖 Recipe", 
                "🥗 Nutrition", 
                "💡 Fun Facts",
                "📊 Analysis"
            ])
            
            # TAB 1: Recipe
            with tab1:
                st.markdown("### 👨‍🍳 Step-by-Step Recipe Guide")
                
                with st.spinner("📥 Fetching recipe..."):
                    recipe_data = get_recipe_from_spoonacular(dish_name)
                
                if recipe_data:
                    # Recipe metrics in floating cards
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h1 style="font-size: 3rem;">🍽️</h1>
                                <h2 style="margin: 0.5rem 0;">{recipe_data['servings']}</h2>
                                <p style="font-size: 1.1rem; font-weight: 600;">Servings</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h1 style="font-size: 3rem;">⏱️</h1>
                                <h2 style="margin: 0.5rem 0;">{recipe_data['ready_in_minutes']}</h2>
                                <p style="font-size: 1.1rem; font-weight: 600;">Minutes</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h1 style="font-size: 3rem;">⭐</h1>
                                <h2 style="margin: 0.5rem 0;">Premium</h2>
                                <p style="font-size: 1.1rem; font-weight: 600;">Recipe</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Ingredients
                    with st.expander("🛒 **Ingredients Shopping List**", expanded=True):
                        if recipe_data['ingredients']:
                            for idx, ingredient in enumerate(recipe_data['ingredients'], 1):
                                st.markdown(f"""
                                    <div class="ingredient-card">
                                        <strong style="font-size: 1.1rem;">{idx}.</strong> 
                                        <span style="font-size: 1.1rem;">{ingredient}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("📝 Ingredients list not available")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Cooking steps
                    st.markdown("### 📋 Cooking Instructions")
                    
                    if recipe_data['instructions']:
                        instructions = recipe_data['instructions']
                        current_step = st.session_state.current_step
                        
                        if 0 <= current_step < len(instructions):
                            st.markdown(f"""
                                <div class="step-card">
                                    <span class="step-number">{current_step + 1}</span>
                                    <span style="font-size: 1.1rem; line-height: 1.6;">{instructions[current_step]}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Navigation
                        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
                        
                        with col_nav1:
                            if st.button("⬅️ Previous", disabled=current_step == 0):
                                st.session_state.current_step -= 1
                                st.rerun()
                        
                        with col_nav2:
                            st.markdown(f"<p style='text-align: center; font-size: 1rem; font-weight: 600;'>Step {current_step + 1} of {len(instructions)}</p>", unsafe_allow_html=True)
                        
                        with col_nav3:
                            if st.button("Next ➡️", disabled=current_step >= len(instructions) - 1):
                                st.session_state.current_step += 1
                                st.rerun()
                        
                        progress = (current_step + 1) / len(instructions)
                        st.progress(progress)
                        
                        if st.button("🔄 Reset to Start"):
                            st.session_state.current_step = 0
                            st.rerun()
                    else:
                        st.info("📝 Detailed instructions not available")
                    
                    if recipe_data['source_url']:
                        st.markdown("---")
                        st.markdown(f"🔗 [View Complete Recipe at Source]({recipe_data['source_url']})")
                else:
                    st.warning("⚠️ Recipe not found in database")
            
            # TAB 2: Nutrition
            with tab2:
                st.markdown("### 🥗 Complete Nutritional Breakdown")
                
                with st.spinner("📊 Calculating nutrition..."):
                    spoon_data = get_recipe_from_spoonacular(dish_name)
                    usda_data = get_nutrition_from_usda(dish_name)
                
                nutrition = combine_nutrition_data(spoon_data, usda_data)
                
                if nutrition:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 🎯 Key Nutrients")
                    
                    key_nutrients = ['Calories', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Sugar']
                    
                    cols = st.columns(3)
                    col_idx = 0
                    for nutrient in key_nutrients:
                        if nutrient in nutrition:
                            with cols[col_idx % 3]:
                                nutrient_value = format_nutrient_value(nutrition[nutrient])
                                st.markdown(f"""
                                    <div class="nutrition-badge">
                                        <h3 style="margin-bottom: 0.5rem;">{nutrient}</h3>
                                        <h1 style="margin: 0;">{nutrient_value}</h1>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.markdown("<br>", unsafe_allow_html=True)
                            col_idx += 1
                    
                    st.markdown("---")
                    
                    with st.expander("📊 **Complete Nutritional Profile**"):
                        for nutrient, data in nutrition.items():
                            nutrient_value = format_nutrient_value(data)
                            st.markdown(f"""
                                <div class="ingredient-card">
                                    <strong style="font-size: 1.1rem;">{nutrient}:</strong> 
                                    <span style="font-size: 1.1rem;">{nutrient_value}</span>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Nutrition data not available")
            
            # TAB 3: Fun Facts
            with tab3:
                st.markdown("### 💡 Fascinating Food Trivia")
                
                facts = get_fun_facts(dish_name)
                
                if facts:
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    current_fact = st.session_state.current_fact
                    
                    if 0 <= current_fact < len(facts):
                        st.markdown(f"""
                            <div class="fun-fact-card">
                                <h3 style="color: #0F766E; margin-bottom: 1rem;">✨ Did You Know? #{current_fact + 1}</h3>
                                <p style="font-size: 1.1rem; line-height: 1.8;">{facts[current_fact]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
                    
                    with col_f1:
                        if st.button("⬅️ Previous", key="prev_fact", disabled=current_fact == 0):
                            st.session_state.current_fact -= 1
                            st.rerun()
                    
                    with col_f2:
                        st.markdown(f"<p style='text-align: center; font-size: 1rem; font-weight: 600;'>Fact {current_fact + 1} of {len(facts)}</p>", unsafe_allow_html=True)
                    
                    with col_f3:
                        if st.button("Next ➡️", key="next_fact", disabled=current_fact >= len(facts) - 1):
                            st.session_state.current_fact += 1
                            st.rerun()
                    
                    if st.button("🔄 Reset"):
                        st.session_state.current_fact = 0
                        st.rerun()
            
            # TAB 4: Analysis
            with tab4:
                st.markdown("### 📊 Detailed AI Analysis")
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("#### 🎯 All Model Predictions")
                for i, pred in enumerate(predictions, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                    st.markdown(f"""
                        <div class="prediction-item">
                            <h3>{emoji} <strong>{pred['name']}</strong></h3>
                            <h4 style="color: #FF6B6B; margin-top: 0.5rem;">Confidence: {pred['confidence']:.2f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### 📈 Confidence Level Assessment")
                conf = top_prediction['confidence']
                if conf >= 90:
                    st.success("🟢 **Excellent Match** - Very high confidence prediction!")
                elif conf >= 70:
                    st.info("🔵 **Good Match** - High confidence prediction")
                elif conf >= 50:
                    st.warning("🟡 **Fair Match** - Medium confidence")
                else:
                    st.error("🔴 **Low Confidence** - Consider uploading a clearer image")

    else:
        # Welcome screen with eye-catching elements
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="recipe-card">
                <h2 style="text-align: center; color: #14B8A6; font-size: 2rem;">👋 Welcome!</h2>
                <p style="text-align: center; color: #64748B; font-size: 1.1rem; margin-top: 1rem; line-height: 1.6;">
                    Upload a food image above to unlock <strong>recipes</strong>, <strong>nutrition facts</strong>, 
                    and <strong>culinary wisdom</strong> powered by AI! 🚀
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### 🌟 Discover Amazing Features")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h1 style="font-size: 4rem; margin-bottom: 1rem;">📖</h1>
                    <h3 style="margin-bottom: 1rem;">Interactive Recipes</h3>
                    <p style="font-size: 1rem;">Step-by-step cooking guide with beautiful navigation</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h1 style="font-size: 4rem; margin-bottom: 1rem;">🥗</h1>
                    <h3 style="margin-bottom: 1rem;">Nutrition Insights</h3>
                    <p style="font-size: 1rem;">Complete nutritional breakdown with visual badges</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h1 style="font-size: 4rem; margin-bottom: 1rem;">💡</h1>
                    <h3 style="margin-bottom: 1rem;">Fun Trivia</h3>
                    <p style="font-size: 1rem;">Fascinating facts about your favorite dishes</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()