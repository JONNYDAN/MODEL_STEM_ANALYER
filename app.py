# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import easyocr
import json
import os
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

# Khởi tạo EasyOCR
print("Initializing EasyOCR...")
reader = easyocr.Reader(['vi', 'en'])

# Dictionary dịch thuật Việt-Anh cho các đối tượng STEM phổ biến
TRANSLATION_DICT = {
    # Hành tinh
    'sao thủy': 'mercury',
    'thủy tinh': 'mercury',
    'sao kim': 'venus',
    'trái đất': 'earth',
    'địa cầu': 'earth',
    'sao hỏa': 'mars',
    'sao mộc': 'jupiter',
    'mộc tinh': 'jupiter',
    'sao thổ': 'saturn',
    'thổ tinh': 'saturn',
    'sao thiên vương': 'uranus',
    'thiên vương tinh': 'uranus',
    'sao hải vương': 'neptune',
    'hải vương tinh': 'neptune',
    'mặt trời': 'sun',
    'mặt trăng': 'moon',
    
    # Vòng đời động vật
    'trứng': 'egg',
    'nòng nọc': 'tadpole',
    'ếch con': 'froglet',
    'ếch trưởng thành': 'adult frog',
    'sâu bướm': 'caterpillar',
    'nhộng': 'pupa',
    'bướm': 'butterfly',
    
    # Thực vật
    'hạt': 'seed',
    'mầm': 'sprout',
    'rễ': 'root',
    'thân': 'stem',
    'lá': 'leaf',
    'hoa': 'flower',
    'quả': 'fruit',
    
    # Cơ thể người
    'tim': 'heart',
    'phổi': 'lung',
    'gan': 'liver',
    'thận': 'kidney',
    'dạ dày': 'stomach',
    'ruột': 'intestine',
    'não': 'brain',
    
    # Khác
    'nước': 'water',
    'khí': 'air',
    'đất': 'soil',
    'lửa': 'fire',
}

def normalize_text(text):
    """Chuẩn hóa text: lowercase, loại bỏ dấu câu không cần thiết"""
    text = text.lower().strip()
    # Loại bỏ dấu câu ở đầu/cuối
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
    return text

def translate_to_english(text):
    """Dịch text sang tiếng Anh"""
    normalized = normalize_text(text)
    
    # Kiểm tra xem đã là tiếng Anh chưa
    if re.match(r'^[a-zA-Z\s]+$', normalized):
        return normalized
    
    # Tìm trong dictionary
    if normalized in TRANSLATION_DICT:
        return TRANSLATION_DICT[normalized]
    
    # Tìm partial match
    for viet_term, eng_term in TRANSLATION_DICT.items():
        if viet_term in normalized or normalized in viet_term:
            return eng_term
    
    # Không tìm thấy, trả về original
    return normalized

def detect_objects_with_ocr(image_np):
    """Phát hiện và nhận diện đối tượng STEM bằng OCR"""
    try:
        results = reader.readtext(image_np)
        detected_objects = []
        
        for i, (bbox, text, prob) in enumerate(results):
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x1, y1 = map(int, top_left)
            x2, y2 = map(int, bottom_right)
            
            # Dịch sang tiếng Anh
            original_text = text.strip()
            translated_text = translate_to_english(original_text)
            
            detected_objects.append({
                'id': i + 1,
                'original_text': original_text,
                'translated_text': translated_text,
                'confidence': float(prob) * 100,  # Convert to percentage
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                },
                'center': {
                    'x': int((x1 + x2) / 2),
                    'y': int((y1 + y2) / 2)
                }
            })
        
        return detected_objects
    except Exception as e:
        print(f"OCR Error: {e}")
        raise e

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'STEM Object Recognition API with OCR',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint để phân tích ảnh STEM và nhận diện các đối tượng"""
    try:
        # Kiểm tra file ảnh
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'message': 'Please provide an image file with key "image"'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400
        
        # Đọc và xử lý ảnh
        print(f"Processing image: {file.filename}")
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Xoay ảnh nếu cần (dựa trên EXIF orientation)
        try:
            exif = image._getexif()
            if exif:
                orientation = exif.get(274, 1)
                if orientation == 3:
                    image_np = cv2.rotate(image_np, cv2.ROTATE_180)
                elif orientation == 6:
                    image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except:
            pass
        
        height, width = image_np.shape[:2]
        
        # Phát hiện đối tượng bằng OCR
        print("Running OCR detection...")
        detected_objects = detect_objects_with_ocr(image_np)
        
        # Tạo ảnh với bounding boxes
        image_with_boxes = image_np.copy()
        for obj in detected_objects:
            bbox = obj['bbox']
            # Vẽ bounding box
            cv2.rectangle(image_with_boxes, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         (0, 255, 0), 2)
            
            # Vẽ label
            label = f"{obj['translated_text']} ({obj['confidence']:.1f}%)"
            cv2.putText(image_with_boxes, label,
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Chuyển ảnh sang base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Tạo response
        response_data = {
            'success': True,
            'metadata': {
                'filename': file.filename,
                'image_size': {
                    'width': width,
                    'height': height
                },
                'total_objects': len(detected_objects),
                'timestamp': datetime.now().isoformat()
            },
            'objects': detected_objects,
            'annotated_image': image_base64
        }
        
        # In kết quả ra console
        print("\n" + "="*60)
        print("DETECTION RESULTS:")
        print("="*60)
        for obj in detected_objects:
            print(f"  {obj['id']}. '{obj['original_text']}' -> {obj['translated_text']} (confidence: {obj['confidence']:.2f}%)")
        print("="*60 + "\n")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing the image'
        }), 500

@app.route('/api/analyze_url', methods=['POST'])
def analyze_from_url():
    """API endpoint để phân tích từ URL"""
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({
                'success': False,
                'error': 'No image URL provided'
            }), 400
        
        import requests as req
        
        url = data['image_url']
        print(f"Downloading image from: {url}")
        
        response = req.get(url, timeout=30)
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': f'Failed to download image. Status code: {response.status_code}'
            }), 400
        
        # Đọc ảnh từ response
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)
        
        height, width = image_np.shape[:2]
        
        # Phát hiện đối tượng bằng OCR
        print("Running OCR detection...")
        detected_objects = detect_objects_with_ocr(image_np)
        
        # Tạo ảnh với bounding boxes
        image_with_boxes = image_np.copy()
        for obj in detected_objects:
            bbox = obj['bbox']
            cv2.rectangle(image_with_boxes, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         (0, 255, 0), 2)
            
            label = f"{obj['translated_text']} ({obj['confidence']:.1f}%)"
            cv2.putText(image_with_boxes, label,
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Chuyển ảnh sang base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Tạo response
        response_data = {
            'success': True,
            'metadata': {
                'source_url': url,
                'image_size': {
                    'width': width,
                    'height': height
                },
                'total_objects': len(detected_objects),
                'timestamp': datetime.now().isoformat()
            },
            'objects': detected_objects,
            'annotated_image': image_base64
        }
        
        # In kết quả ra console
        print("\n" + "="*60)
        print("DETECTION RESULTS:")
        print("="*60)
        for obj in detected_objects:
            print(f"  {obj['id']}. '{obj['original_text']}' -> {obj['translated_text']} (confidence: {obj['confidence']:.2f}%)")
        print("="*60 + "\n")
        
        return jsonify(response_data), 200
            
    except Exception as e:
        print(f"Error in analyze_from_url: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing the URL'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"Starting STEM Object Recognition API on port {port}")
    print(f"Service: OCR-based object detection and translation")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=True)