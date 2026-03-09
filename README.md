# STEM Object Recognition API with OCR

API nhận diện đối tượng trong ảnh STEM sử dụng OCR (Optical Character Recognition) và dịch thuật sang tiếng Anh.

## Tính năng

- ✅ Nhận diện văn bản trong ảnh STEM bằng EasyOCR
- ✅ Hỗ trợ tiếng Việt và tiếng Anh
- ✅ Tự động dịch thuật sang tiếng Anh
- ✅ Chuẩn hóa tiếng Anh + sửa lỗi chính tả theo taxonomy từ PostgreSQL
- ✅ Phân tích ngữ nghĩa đa phương thức bằng Gemini (text/image)
- ✅ Hỗ trợ key rotation cho Gemini (`GEMINI_API_KEY_1..n`)
- ✅ Trả về confidence score cho mỗi detection
- ✅ Vẽ bounding box và label lên ảnh
- ✅ Hỗ trợ cả file upload và URL

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động server

```bash
python app.py
```

Server sẽ chạy tại `http://localhost:5000`

### 3. Cấu hình Gemini (khuyến nghị)

Thiết lập biến môi trường trước khi chạy API:

```bash
# Một key
set GEMINI_API_KEY=your_key

# Hoặc nhiều key để rotation
set GEMINI_API_KEY_1=your_key_1
set GEMINI_API_KEY_2=your_key_2
set GEMINI_API_KEY_3=your_key_3

# Tùy chọn
set GEMINI_MODEL=gemini-2.5-flash
set GEMINI_TIMEOUT_SECONDS=30
set GEMINI_ONLY_MODE=true
```

API sẽ tự động thử lần lượt key theo thứ tự khi gọi Gemini.
Để luân phiên key theo mỗi request, có thể khai báo nhiều key:

```bash
set GEMINI_API_KEY_1=your_key_1
set GEMINI_API_KEY_2=your_key_2
set GEMINI_API_KEY_3=your_key_3
set GEMINI_API_KEY_4=your_key_4
set GEMINI_API_KEY_5=your_key_5
```

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Kiểm tra trạng thái của API.

**Response:**
```json
{
  "status": "healthy",
  "service": "STEM Object Recognition API with OCR",
  "version": "1.0.0",
  "timestamp": "2026-02-02T10:30:00"
}
```

### 2. Analyze Image (Upload)

**POST** `/api/analyze`

Phân tích ảnh được upload.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File ảnh với key `image`

**Response:**
```json
{
  "success": true,
  "metadata": {
    "filename": "solar_system.jpg",
    "image_size": {
      "width": 1920,
      "height": 1080
    },
    "total_objects": 8,
    "timestamp": "2026-02-02T10:30:00"
  },
  "objects": [
    {
      "id": 1,
      "original_text": "Uranus",
      "translated_text": "uranus",
      "confidence": 99.99,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 250
      },
      "center": {
        "x": 200,
        "y": 225
      }
    },
    {
      "id": 2,
      "original_text": "Jupiter",
      "translated_text": "jupiter",
      "confidence": 100.00,
      "bbox": {
        "x1": 400,
        "y1": 200,
        "x2": 600,
        "y2": 250
      },
      "center": {
        "x": 500,
        "y": 225
      }
    }
  ],
  "annotated_image": "base64_encoded_image..."
}
```

### 3. Analyze Image (URL)

**POST** `/api/analyze_url`

Phân tích ảnh từ URL.

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response:** Giống như `/api/analyze`

### 4. Analyze Intent (Text / Image / Multimodal)

**POST** `/api/analyze_intent`

Endpoint dùng cho API tích hợp (pha 1/2/3), nhận `query_text`, `image` hoặc cả hai.

Các trường chính trong response:
- `phase`: `text` | `image` | `multimodal`
- `analysis_case`: `case_1_category_keyword` | `case_2_subject_or_sro` | `case_3_pending_learning`
- `category_candidates`, `subject_candidates`, `sro_candidates`
- `creative_recommendation.description`, `creative_recommendation.youtube_queries`

## Sử dụng với Python Client

```python
import requests

# Upload file
def analyze_local_image(image_path):
    url = "http://localhost:5000/api/analyze"
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    return response.json()

# Từ URL
def analyze_image_url(image_url):
    url = "http://localhost:5000/api/analyze_url"
    data = {'image_url': image_url}
    response = requests.post(url, json=data)
    return response.json()

# Sử dụng
results = analyze_local_image("my_stem_image.jpg")
print(results)
```

## Sử dụng với cURL

### Upload file

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@solar_system.jpg"
```

### Từ URL

```bash
curl -X POST http://localhost:5000/api/analyze_url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```

## Dictionary dịch thuật

API hỗ trợ dịch các đối tượng STEM phổ biến:

### Hành tinh (Planets)
- Sao Thủy / Thủy Tinh → mercury
- Sao Kim → venus
- Trái Đất / Địa Cầu → earth
- Sao Hỏa → mars
- Sao Mộc / Mộc Tinh → jupiter
- Sao Thổ / Thổ Tinh → saturn
- Sao Thiên Vương → uranus
- Sao Hải Vương → neptune
- Mặt Trời → sun
- Mặt Trăng → moon

### Vòng đời động vật (Life Cycles)
- Trứng → egg
- Nòng Nọc → tadpole
- Ếch Con → froglet
- Ếch Trưởng Thành → adult frog
- Sâu Bướm → caterpillar
- Nhộng → pupa
- Bướm → butterfly

### Thực vật (Plants)
- Hạt → seed
- Mầm → sprout
- Rễ → root
- Thân → stem
- Lá → leaf
- Hoa → flower
- Quả → fruit

### Cơ thể người (Human Body)
- Tim → heart
- Phổi → lung
- Gan → liver
- Thận → kidney
- Dạ Dày → stomach
- Ruột → intestine
- Não → brain

## Ví dụ kết quả

```
============================================================
DETECTION RESULTS:
============================================================
  1. 'Uranus' -> uranus (confidence: 99.99%)
  2. 'Jupiter' -> jupiter (confidence: 100.00%)
  3. 'Venus' -> venus (confidence: 66.24%)
  4. 'Neptune' -> neptune (confidence: 99.99%)
  5. 'Earth' -> earth (confidence: 98.08%)
  6. 'Mercury' -> mercury (confidence: 99.99%)
  7. 'Mars' -> mars (confidence: 100.00%)
  8. 'Saturn' -> saturn (confidence: 100.00%)
============================================================
```

## Mở rộng Dictionary

Để thêm từ mới vào dictionary dịch thuật, chỉnh sửa `TRANSLATION_DICT` trong file `app.py`:

```python
TRANSLATION_DICT = {
    # Thêm từ mới
    'từ tiếng việt': 'english translation',
    # ...
}
```

## Lưu ý

- EasyOCR cần tải models lần đầu chạy (có thể mất vài phút)
- API tự động xử lý cả text tiếng Việt và tiếng Anh
- Confidence score được trả về dưới dạng phần trăm (0-100%)
- Annotated image được trả về dưới dạng base64

## Deployment

### Sử dụng Gunicorn (Production)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Coming soon)

```bash
docker build -t stem-ocr-api .
docker run -p 5000:5000 stem-ocr-api
```

## License

MIT License
