import tensorflow as tf
from PIL import Image
import numpy as np
import json
import sys
import os

def load_model_and_labels():
    """Load trained model and class labels"""
    print("Loading model...")
    if os.path.exists('food_recognition_model.keras'):
        model = tf.keras.models.load_model('food_recognition_model.keras')
        print("✓ Loaded: food_recognition_model.keras")
    elif os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
        print("✓ Loaded: best_model.keras")
    else:
        print("✗ Error: Model not found!")
        print("  Train the model first: python train_model.py")
        return None, None
    
    if os.path.exists('class_labels.json'):
        with open('class_labels.json', 'r') as f:
            class_labels = json.load(f)
        print(f"✓ Loaded {len(class_labels)} class labels")
    else:
        print("✗ Error: class_labels.json not found!")
        return None, None
    
    return model, class_labels

def preprocess_image(image_path, img_size=128):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_top_k(model, img_array, class_labels, k=5):
    """Get top k predictions"""
    predictions = model.predict(img_array, verbose=0)
    top_indices = np.argsort(predictions[0])[-k:][::-1]
    
    results = []
    for idx in top_indices:
        class_name = class_labels[str(idx)]
        confidence = predictions[0][idx] * 100
        results.append((class_name, confidence))
    
    return results

def main():
    print("=" * 60)
    print("Food Recognition - Prediction Test")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python test_prediction.py <image_path>")
        print("\nExamples:")
        print("  python test_prediction.py food_images/Biryani/1_2.jpg")
        print("  python test_prediction.py food_images/Pizza/2_1.jpg")
        print("  python test_prediction.py my_food_photo.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"✗ Error: Image not found at '{image_path}'")
        print("\nMake sure the file exists and path is correct.")
        return
    
    # Load model
    model, class_labels = load_model_and_labels()
    if model is None or class_labels is None:
        return
    
    print(f"\nProcessing: {image_path}")
    
    # Load and preprocess image
    try:
        img, img_array = preprocess_image(image_path)
        print("✓ Image preprocessed successfully")
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    results = predict_top_k(model, img_array, class_labels, k=5)
    
    # Display results
    print("\n" + "=" * 60)
    print("TOP 5 PREDICTIONS")
    print("=" * 60)
    
    for i, (class_name, confidence) in enumerate(results, 1):
        # Add confidence indicator
        if confidence >= 80:
            indicator = "🟢"
        elif confidence >= 50:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{indicator} {i}. {class_name:30s} {confidence:6.2f}%")
    
    print("=" * 60)
    
    # Show top prediction interpretation
    top_dish, top_conf = results[0]
    print(f"\nBest Prediction: {top_dish}")
    if top_conf >= 80:
        print("Confidence: Very High ✓")
    elif top_conf >= 50:
        print("Confidence: Medium ⚠")
    else:
        print("Confidence: Low - Try a clearer image ✗")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  • Run the web app: streamlit run app.py")
    print("  • Try another image: python test_prediction.py <path>")
    print("=" * 60)
    
    # Try to display image
    try:
        print("\nOpening image...")
        img.show()
    except:
        print("Could not display image automatically.")
        print(f"View it manually: {image_path}")

if __name__ == "__main__":
    main()