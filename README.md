# 🍽️ Dish Recognition and Nutrition Analysis Pipeline

A complete deep learning application that recognizes food dishes from images and provides detailed recipe, nutrition, and fun facts.

## 🎯 Features

- **Deep Learning Model**: EfficientNetB0-based CNN with transfer learning
- **121 Food Classes**: Trained on MAFood-121 dataset
- **Recipe Extraction**: Detailed ingredients and cooking instructions
- **Nutrition Analysis**: Comprehensive nutritional breakdown
- **Fun Facts**: Interesting information about each dish
- **Interactive Web Interface**: Beautiful Streamlit UI

## 🏗️ Architecture

### Model Details
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Transfer Learning**: Two-phase training (frozen → fine-tuning)
- **Custom Layers**: 
  - GlobalAveragePooling2D
  - Dense(512) + Dropout(0.5)
  - Dense(256) + Dropout(0.3)
  - Output Dense(121, softmax)
- **Input Size**: 224x224x3
- **Data Augmentation**: Rotation, shifts, zoom, flip

### APIs Used
1. **Spoonacular API**: Recipe and nutrition data
2. **USDA FoodData Central**: Additional nutrition information

## 📦 Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify dataset**:
```bash
python dataset_verify.py
```

## 🚀 Usage

### Step 1: Train the Model
```bash
python train_model.py
```

This will:
- Load and augment the MAFood-121 dataset
- Train EfficientNetB0 model with transfer learning
- Save best model as `food_recognition_model.keras`
- Create `class_labels.json` for predictions

**Training takes approximately**: 3-6 hours on GPU, 12-24 hours on CPU

### Step 2: Run the Web Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Step 3: Upload and Analyze
1. Upload a food image
2. View top predictions with confidence scores
3. Explore recipe, nutrition, and fun facts

## 📁 Project Structure

```
dish-recognition/
│
├── food_images/              # MAFood-121 dataset
│   ├── Biryani/
│   ├── Pizza/
│   └── ... (121 classes)
│
├── train_model.py           # Deep learning training script
├── app.py                   # Streamlit web application
├── utils.py                 # API helpers and utilities
├── requirements.txt         # Python dependencies
│
├── api_test.py             # Spoonacular API testing
├── api_test_USDA.py        # USDA API testing
├── dataset_verify.py       # Dataset verification
├── show_image.py          # Image display utility
│
├── food_recognition_model.keras  # Trained model (generated)
├── class_labels.json             # Class mappings (generated)
└── README.md
```

## 🎓 Model Training Details

### Phase 1: Transfer Learning (Epochs 1-15)
- Base model frozen
- Train only custom layers
- Learning rate: 0.001
- Goal: Learn dataset-specific features

### Phase 2: Fine-Tuning (Epochs 16-50)
- Unfreeze last 20 layers of base model
- Lower learning rate: 0.0001
- Goal: Refine feature extraction

### Callbacks
- **EarlyStopping**: Patience=7, restore best weights
- **ReduceLROnPlateau**: Reduce LR on plateau
- **ModelCheckpoint**: Save best model by validation accuracy

### Expected Performance
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 75-85%
- **Top-5 Accuracy**: 90-95%

## 🔧 Configuration

### API Keys
Update in `utils.py`:
```python
SPOONACULAR_API_KEY = "your_key_here"
USDA_API_KEY = "your_key_here"
```

### Model Hyperparameters
Edit in `train_model.py`:
```python
IMG_SIZE = 224          # Image size
BATCH_SIZE = 32         # Batch size
EPOCHS = 50            # Training epochs
```

## 📊 Performance Metrics

The model tracks:
- **Categorical Accuracy**: Overall classification accuracy
- **Top-5 Accuracy**: Correct class in top 5 predictions
- **Loss**: Categorical cross-entropy

## 🎨 Web Interface Features

### Main Dashboard
- Image upload with drag & drop
- Real-time predictions with confidence scores
- Adjustable confidence threshold
- Multiple prediction display

### Recipe Tab
- Ingredient list
- Step-by-step instructions
- Servings and cooking time
- Source link for full recipe

### Nutrition Tab
- Key nutrients (calories, protein, fat, carbs)
- Complete nutritional breakdown
- Per serving information

### Fun Facts Tab
- Interesting trivia about the dish
- Historical information
- Cultural significance

## 🚀 Improvements & Suggestions

### Model Improvements
1. **Increase Dataset Size**: 
   - Add more images per class (currently ~175)
   - Use data augmentation more aggressively

2. **Try Different Architectures**:
   - EfficientNetB3/B4 for better accuracy
   - Vision Transformer (ViT) for state-of-the-art performance
   - Ensemble multiple models

3. **Advanced Techniques**:
   - Mixup/CutMix augmentation
   - Test-time augmentation (TTA)
   - Knowledge distillation

4. **Class Balancing**:
   - Handle imbalanced classes
   - Use focal loss or class weights

### Application Improvements
1. **Features**:
   - Batch processing of images
   - Save favorite recipes
   - Shopping list generator
   - Meal planning

2. **Performance**:
   - Model quantization for faster inference
   - Caching predictions
   - Progressive image loading

3. **User Experience**:
   - Dark mode
   - Multi-language support
   - Voice instructions
   - Social sharing

### Deployment
1. **Cloud Hosting**:
   - Deploy on Streamlit Cloud (free)
   - Use AWS/GCP for scalability
   - Dockerize the application

2. **Mobile App**:
   - Convert to TensorFlow Lite
   - Build React Native app
   - Add camera integration

3. **API Service**:
   - Create REST API with FastAPI
   - Add authentication
   - Rate limiting

## 🐛 Troubleshooting

### Common Issues

**1. Model not found**
```bash
# Train the model first
python train_model.py
```

**2. Out of memory**
```python
# Reduce batch size in train_model.py
BATCH_SIZE = 16  # or 8
```

**3. API rate limit**
- Spoonacular: 150 requests/day on free tier
- USDA: 1000 requests/hour
- Cache results to reduce API calls

**4. Low accuracy**
- Train for more epochs
- Adjust learning rate
- Increase data augmentation
- Verify dataset quality

## 📚 Resources

- **MAFood-121 Dataset**: [Link](https://www.kaggle.com/datasets/kmader/food41)
- **Spoonacular API**: [Documentation](https://spoonacular.com/food-api)
- **USDA API**: [Documentation](https://fdc.nal.usda.gov/api-guide.html)
- **EfficientNet Paper**: [arXiv](https://arxiv.org/abs/1905.11946)

## 📝 License

This project is for educational purposes. Please ensure you have proper rights to use the dataset and APIs.

## 🙏 Acknowledgments

- MAFood-121 dataset creators
- Spoonacular and USDA for API access
- TensorFlow and Streamlit teams

---

**Made with ❤️ for food lovers and ML enthusiasts**
