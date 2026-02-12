# client_example.py
import requests
import json
from PIL import Image
import io
import base64

def analyze_local_image(image_path, api_url="http://localhost:5000/api/analyze"):
    """Gửi ảnh local đến API"""
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(api_url, files=files)
    
    return response.json()

def analyze_image_url(image_url, api_url="http://localhost:5000/api/analyze_url"):
    """Gửi URL ảnh đến API"""
    data = {'image_url': image_url}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(api_url, json=data, headers=headers)
    
    return response.json()

def print_results(results):
    """In kết quả ra console"""
    if results.get('success'):
        print("=" * 60)
        print("STEM OBJECT RECOGNITION RESULTS")
        print("=" * 60)
        
        metadata = results.get('metadata', {})
        print(f"\nImage Size: {metadata.get('image_size', {}).get('width')} x {metadata.get('image_size', {}).get('height')}")
        print(f"Total Objects Detected: {metadata.get('total_objects', 0)}")
        
        print("\nDETECTED OBJECTS:")
        print("-" * 60)
        
        objects = results.get('objects', [])
        for obj in objects:
            print(f"  {obj['id']}. '{obj['original_text']}' -> {obj['translated_text']} (confidence: {obj['confidence']:.2f}%)")
            print(f"      Position: ({obj['bbox']['x1']}, {obj['bbox']['y1']}) to ({obj['bbox']['x2']}, {obj['bbox']['y2']})")
            print(f"      Center: ({obj['center']['x']}, {obj['center']['y']})")
        
        # Lưu ảnh đã annotate
        if 'annotated_image' in results:
            image_data = base64.b64decode(results['annotated_image'])
            with open('annotated_result.png', 'wb') as f:
                f.write(image_data)
            print(f"\nAnnotated image saved as: annotated_result.png")
        
        print("=" * 60)
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"Message: {results.get('message', '')}")

def health_check(api_url="http://localhost:5000/api/health"):
    """Kiểm tra API health"""
    try:
        response = requests.get(api_url)
        result = response.json()
        
        print("=" * 60)
        print("API HEALTH CHECK")
        print("=" * 60)
        print(f"Status: {result.get('status')}")
        print(f"Service: {result.get('service')}")
        print(f"Version: {result.get('version', 'N/A')}")
        print(f"Timestamp: {result.get('timestamp')}")
        print("=" * 60)
        
        return result.get('status') == 'healthy'
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

if __name__ == "__main__":
    # Kiểm tra API trước
    print("Checking API health...")
    if not health_check():
        print("\nAPI is not available. Please start the server first.")
        exit(1)
    
    print("\n" + "=" * 60 + "\n")
    
    # Ví dụ 1: Phân tích ảnh local (nếu có)
    # Uncomment khi có file ảnh
    # print("Example 1: Analyzing local image")
    # results = analyze_local_image("solar_system.jpg")
    # print_results(results)
    # print("\n" + "=" * 60 + "\n")
    
    # Ví dụ 2: Phân tích từ URL
    print("Example: Analyzing from URL")
    # URL ví dụ - thay thế bằng URL ảnh STEM của bạn
    url = "https://video.vietjack.com/upload2/images/1673339885/1673340220-image2.png"
    
    try:
        results = analyze_image_url(url)
        print_results(results)
    except Exception as e:
        print(f"Error: {e}")
