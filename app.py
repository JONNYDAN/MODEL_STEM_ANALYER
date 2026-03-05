from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import difflib
import easyocr
import io
import json
import numpy as np
import os
import re
import requests as req
import unicodedata
from datetime import datetime, timedelta
from PIL import Image

try:
    import psycopg2
except Exception:
    psycopg2 = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from langdetect import detect as detect_language
except Exception:
    detect_language = None

try:
    from rapidfuzz import process as rapidfuzz_process
except Exception:
    rapidfuzz_process = None

app = Flask(__name__)
CORS(app)

print("Initializing EasyOCR...")
reader = easyocr.Reader(['vi', 'en'])

VI_EN_PHRASE_MAP = {
    "hay cho toi biet": "tell me",
    "cho toi biet": "tell me",
    "vong doi": "life cycle",
    "vong doi cua nuoc": "water cycle",
    "vong doi cua ladybug": "ladybug life cycle",
    "vong doi cua laydybug": "ladybug life cycle",
    "vong doi cua bo rua": "ladybug life cycle",
    "la gi": "what is",
    "nam o dau": "where located",
    "mat troi": "sun",
    "trai dat": "earth",
    "dia cau": "earth",
    "sao hoa": "mars",
    "sao moc": "jupiter",
    "sao tho": "saturn",
    "sao thuy": "mercury",
    "sao kim": "venus",
    "trung": "egg",
    "nong noc": "tadpole",
    "ech con": "froglet",
    "ech truong thanh": "adult frog",
    "chuon chuon": "dragonfly",
    "nuoc": "water",
    "lady bug": "ladybug",
    "laydybug": "ladybug",
    "bo rua": "ladybug",
}

TRANSLATION_DICT = {
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
    'trứng': 'egg',
    'nòng nọc': 'tadpole',
    'ếch con': 'froglet',
    'ếch': 'frog',
    'ếch trưởng thành': 'adult frog',
    'sâu bướm': 'caterpillar',
    'nhộng': 'pupa',
    'bướm': 'butterfly',
    'hạt': 'seed',
    'mầm': 'sprout',
    'rễ': 'root',
    'thân': 'stem',
    'lá': 'leaf',
    'hoa': 'flower',
    'quả': 'fruit',
    'tim': 'heart',
    'phổi': 'lung',
    'gan': 'liver',
    'thận': 'kidney',
    'dạ dày': 'stomach',
    'ruột': 'intestine',
    'bo nao': 'brain',
    'chuồn chuồn': 'dragonfly',
    'nước': 'water',
    'nuoc': 'water',
    'bọ rùa': 'ladybug',
    'bo rua': 'ladybug',
    'lady bug': 'ladybug',
    'laydybug': 'ladybug',
}

_taxonomy_cache = {"data": None, "expires_at": datetime.min}

VI_FILLER_TOKENS = {
    "cua", "la", "nhung", "cac", "cho", "toi", "hay", "xin", "vui", "long", "ve", "duoc",
    "khong", "o", "dau", "mot", "va", "hoac", "the", "nao", "gi", "co", "trong",
    "con", "nhu",
}

GENERIC_SUBJECT_TOKENS = {
    "life", "cycle", "diagram", "process", "stage", "stages", "system", "overview",
}

GENERIC_CATEGORY_TOKENS = {
    "life", "cycle", "cycles", "diagram", "process", "system", "stage", "stages",
}


def _normalize_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or ""))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    return value.lower()


def _clean_text(text: str) -> str:
    source = text or ""
    source = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", source)
    source = source.replace("_", " ")
    value = _normalize_ascii(source)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _translate_to_english(text: str) -> str:
    raw_input = (text or "").strip()
    value = _clean_text(raw_input)
    if not value:
        return ""

    translated_by_model = ""
    detected_lang = None
    if detect_language:
        try:
            detected_lang = detect_language(raw_input)
        except Exception:
            detected_lang = None

    should_translate = bool(detected_lang and detected_lang != "en") or bool(re.search(r"[^\x00-\x7F]", raw_input))
    if should_translate and GoogleTranslator:
        try:
            translated_by_model = GoogleTranslator(source="auto", target="en").translate(raw_input)
        except Exception:
            translated_by_model = ""

    if translated_by_model:
        value = _clean_text(translated_by_model)

    for vi, en in sorted(VI_EN_PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(rf"\b{re.escape(vi)}\b", en, value)

    for vi, en in TRANSLATION_DICT.items():
        vi_clean = _clean_text(vi)
        value = re.sub(rf"\b{re.escape(vi_clean)}\b", en, value)

    tokens = [tok for tok in value.split() if tok and tok not in VI_FILLER_TOKENS]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _extract_tokens(text: str) -> list:
    return [tok for tok in _clean_text(text).split() if tok]


def _build_query_ngrams(tokens: list, max_size: int = 3) -> set:
    ngrams = set()
    if not tokens:
        return ngrams
    upper = min(max_size, len(tokens))
    for size in range(2, upper + 1):
        for i in range(len(tokens) - size + 1):
            ngrams.add(" ".join(tokens[i:i + size]))
    return ngrams


def _get_pg_connection():
    if psycopg2 is None:
        return None
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "stem_kg"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def _fetch_taxonomy_from_postgres() -> dict:
    now = datetime.utcnow()
    if _taxonomy_cache["data"] and now < _taxonomy_cache["expires_at"]:
        return _taxonomy_cache["data"]

    taxonomy = {
        "categories": [],
        "subjects": [],
    }

    conn = None
    try:
        conn = _get_pg_connection()
        if conn is None:
            return taxonomy

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id, c.name, COALESCE(c.description, ''), COALESCE(rc.name, '')
                FROM categories c
                LEFT JOIN root_categories rc ON c.root_category_id = rc.id
                """
            )
            for row in cursor.fetchall():
                taxonomy["categories"].append(
                    {
                        "category_id": row[0],
                        "category_name": row[1] or "",
                        "description": row[2] or "",
                        "root_category": row[3] or "",
                    }
                )

            cursor.execute(
                """
                SELECT s.id, s.name, COALESCE(s.synonyms::text, ''), COALESCE(rs.name, '')
                FROM subjects s
                LEFT JOIN root_subjects rs ON s.root_subject_id = rs.id
                """
            )
            for row in cursor.fetchall():
                synonym_text = row[2] or ""
                synonyms = []
                if synonym_text:
                    try:
                        parsed = json.loads(synonym_text)
                        if isinstance(parsed, list):
                            synonyms = [str(item) for item in parsed if item]
                    except Exception:
                        synonyms = [item.strip() for item in synonym_text.split(",") if item.strip()]

                taxonomy["subjects"].append(
                    {
                        "subject_id": row[0],
                        "subject_name": row[1] or "",
                        "synonyms": synonyms,
                        "root_subject": row[3] or "",
                    }
                )

    except Exception as ex:
        print(f"PostgreSQL taxonomy fetch error: {ex}")
    finally:
        if conn:
            conn.close()

    _taxonomy_cache["data"] = taxonomy
    _taxonomy_cache["expires_at"] = now + timedelta(seconds=120)
    return taxonomy


def _build_vocabulary(taxonomy: dict) -> set:
    vocabulary = set()
    for term in TRANSLATION_DICT.values():
        vocabulary.update(_extract_tokens(term))

    for category in taxonomy.get("categories", []):
        vocabulary.update(_extract_tokens(category.get("category_name", "")))
        vocabulary.update(_extract_tokens(category.get("root_category", "")))

    for subject in taxonomy.get("subjects", []):
        vocabulary.update(_extract_tokens(subject.get("subject_name", "")))
        for synonym in subject.get("synonyms", []):
            vocabulary.update(_extract_tokens(synonym))

    return {item for item in vocabulary if len(item) >= 3}


def _correct_spelling(text: str, vocabulary: set) -> str:
    tokens = _extract_tokens(text)
    corrected = []
    for token in tokens:
        if token in vocabulary:
            corrected.append(token)
            continue

        if rapidfuzz_process and vocabulary:
            try:
                best = rapidfuzz_process.extractOne(token, list(vocabulary), score_cutoff=85)
                if best and best[0]:
                    corrected.append(best[0])
                    continue
            except Exception:
                pass

        matched = difflib.get_close_matches(token, list(vocabulary), n=1, cutoff=0.83)
        corrected.append(matched[0] if matched else token)
    return " ".join(corrected)


def _match_categories(query_terms: set, taxonomy: dict) -> list:
    query_terms = {term for term in query_terms if term}
    query_tokens = sorted(query_terms)
    query_ngrams = _build_query_ngrams(query_tokens, max_size=3)

    matches = []
    for item in taxonomy.get("categories", []):
        category_name = item.get("category_name", "")
        haystack = " ".join(
            [
                category_name,
                item.get("root_category", ""),
                item.get("description", ""),
            ]
        )
        haystack_tokens = set(_extract_tokens(haystack))
        common = sorted(query_terms.intersection(haystack_tokens))

        category_tokens = _extract_tokens(category_name)
        category_phrase = " ".join(category_tokens)
        phrase_hit = 1 if category_phrase and category_phrase in query_ngrams else 0
        lifecycle_boost = 0
        if {"life", "cycle"}.issubset(query_terms) and {"life", "cycles"}.issubset(set(category_tokens)):
            lifecycle_boost = 2

        if not common and phrase_hit == 0 and lifecycle_boost == 0:
            continue

        specific_common = [term for term in common if term not in GENERIC_CATEGORY_TOKENS]
        generic_common = [term for term in common if term in GENERIC_CATEGORY_TOKENS]

        score = (
            len(common) / max(1, len(haystack_tokens))
            + len(specific_common) * 2.2
            + len(generic_common) * 0.6
            + phrase_hit * 1.8
            + lifecycle_boost
        )
        matches.append(
            {
                "category_id": item.get("category_id"),
                "category_name": item.get("category_name"),
                "root_category": item.get("root_category"),
                "score": round(score, 4),
                "matched_terms": common,
            }
        )
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:8]


def _match_subjects(query_terms: set, taxonomy: dict) -> list:
    filtered_query_terms = {
        token for token in query_terms
        if token and token not in GENERIC_SUBJECT_TOKENS
    }

    results = []
    for item in taxonomy.get("subjects", []):
        aliases = [item.get("subject_name", "")] + item.get("synonyms", [])
        alias_tokens = set()
        for alias in aliases:
            alias_tokens.update(_extract_tokens(alias))
        common = sorted(filtered_query_terms.intersection(alias_tokens))
        if not common:
            continue
        score = len(common) / max(1, len(alias_tokens)) + len(common)
        results.append(
            {
                "subject_id": item.get("subject_id"),
                "subject_name": item.get("subject_name"),
                "root_subject": item.get("root_subject"),
                "score": round(score, 4),
                "matched_terms": common,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


def _extract_spatial_relationships(objects: list) -> list:
    if len(objects) < 2:
        return []

    triples = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            ax = a["center"]["x"]
            ay = a["center"]["y"]
            bx = b["center"]["x"]
            by = b["center"]["y"]

            relation = None
            if abs(ax - bx) >= abs(ay - by):
                relation = "left_of" if ax < bx else "right_of"
            else:
                relation = "above" if ay < by else "below"

            triples.append(
                {
                    "subject": a["translated_text"],
                    "relationship": relation,
                    "object": b["translated_text"],
                }
            )
            if len(triples) >= 12:
                return triples
    return triples


def _parse_text_triple(text_en: str) -> list:
    if not text_en:
        return []

    patterns = [
        r"([a-z0-9 ]{2,})\s+(left of|right of|above|below|around|inside|contains|part of|related to)\s+([a-z0-9 ]{2,})",
        r"([a-z0-9 ]{2,})\s*->\s*([a-z0-9_ ]{2,})\s*->\s*([a-z0-9 ]{2,})",
    ]

    triples = []
    for pattern in patterns:
        found = re.findall(pattern, text_en)
        for item in found:
            if len(item) == 3:
                triples.append(
                    {
                        "subject": item[0].strip(),
                        "relationship": item[1].strip().replace(" ", "_"),
                        "object": item[2].strip(),
                    }
                )
    return triples


def _annotate_image(image_np: np.ndarray, objects: list) -> str:
    image_with_boxes = image_np.copy()
    for obj in objects:
        bbox = obj["bbox"]
        cv2.rectangle(
            image_with_boxes,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 255, 0),
            2,
        )
        label = f"{obj['translated_text']} ({obj['confidence']:.1f}%)"
        cv2.putText(
            image_with_boxes,
            label,
            (bbox["x1"], max(20, bbox["y1"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def detect_objects_with_ocr(image_np: np.ndarray) -> list:
    results = reader.readtext(image_np)
    detected_objects = []

    for i, (bbox, text, prob) in enumerate(results):
        (top_left, _, bottom_right, _) = bbox
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)

        original_text = text.strip()
        translated_text = _translate_to_english(original_text)
        if not translated_text:
            continue

        detected_objects.append(
            {
                "id": i + 1,
                "original_text": original_text,
                "translated_text": translated_text,
                "confidence": float(prob) * 100,
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
            }
        )

    return detected_objects


def _read_image_from_upload(file_storage) -> np.ndarray:
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

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
    except Exception:
        pass

    return image_np


def _analyze_intent(query_text: str = "", image_np: np.ndarray = None, source_name: str = "") -> dict:
    taxonomy = _fetch_taxonomy_from_postgres()
    vocabulary = _build_vocabulary(taxonomy)

    text_en = _translate_to_english(query_text or "")
    corrected_text_en = _correct_spelling(text_en, vocabulary)

    objects = detect_objects_with_ocr(image_np) if image_np is not None else []
    detected_labels = [obj.get("translated_text") for obj in objects if obj.get("translated_text")]

    all_terms = set(_extract_tokens(corrected_text_en))
    for label in detected_labels:
        all_terms.update(_extract_tokens(label))

    category_candidates = _match_categories(all_terms, taxonomy)
    subject_candidates = _match_subjects(all_terms, taxonomy)

    sro_candidates = []
    sro_candidates.extend(_parse_text_triple(corrected_text_en))
    sro_candidates.extend(_extract_spatial_relationships(objects))
    sro_candidates = [dict(t) for t in {tuple(item.items()) for item in sro_candidates}]

    phase = "text"
    if image_np is not None and query_text:
        phase = "multimodal"
    elif image_np is not None:
        phase = "image"

    image_size = None
    annotated_image = None
    if image_np is not None:
        height, width = image_np.shape[:2]
        image_size = {"width": width, "height": height}
        annotated_image = _annotate_image(image_np, objects) if objects else None

    return {
        "success": True,
        "phase": phase,
        "metadata": {
            "source": source_name,
            "image_size": image_size,
            "total_objects": len(objects),
            "timestamp": datetime.utcnow().isoformat(),
            "taxonomy_size": {
                "categories": len(taxonomy.get("categories", [])),
                "subjects": len(taxonomy.get("subjects", [])),
            },
        },
        "normalized_query_en": text_en,
        "corrected_query_en": corrected_text_en,
        "objects": objects,
        "detected_labels": detected_labels,
        "category_candidates": category_candidates,
        "subject_candidates": subject_candidates,
        "sro_candidates": sro_candidates,
        "annotated_image": annotated_image,
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify(
        {
            'status': 'healthy',
            'service': 'STEM Multimodal Analyzer',
            'version': '2.0.0',
            'timestamp': datetime.utcnow().isoformat(),
        }
    )


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        image_np = _read_image_from_upload(file)
        result = _analyze_intent(query_text="", image_np=image_np, source_name=file.filename or "upload")
        return jsonify(result), 200
    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex), 'message': 'Analyze image failed'}), 500


@app.route('/api/analyze_url', methods=['POST'])
def analyze_from_url():
    try:
        data = request.get_json() or {}
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'success': False, 'error': 'No image URL provided'}), 400

        response = req.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to download image ({response.status_code})'}), 400

        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)
        result = _analyze_intent(query_text="", image_np=image_np, source_name=image_url)
        return jsonify(result), 200
    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex), 'message': 'Analyze URL failed'}), 500


@app.route('/api/analyze_intent', methods=['POST'])
def analyze_intent():
    try:
        query_text = request.form.get('query_text', '')
        image_np = None
        source_name = ""

        if 'image' in request.files and request.files['image'] and request.files['image'].filename:
            file = request.files['image']
            image_np = _read_image_from_upload(file)
            source_name = file.filename or "upload"
        elif request.is_json:
            data = request.get_json() or {}
            query_text = data.get('query_text', query_text)
            image_url = data.get('image_url')
            if image_url:
                response = req.get(image_url, timeout=30)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                image_np = np.array(image)
                source_name = image_url

        if not query_text and image_np is None:
            return jsonify({'success': False, 'error': 'Please provide query_text or image'}), 400

        result = _analyze_intent(query_text=query_text or "", image_np=image_np, source_name=source_name)
        return jsonify(result), 200
    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex), 'message': 'Analyze intent failed'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'=' * 60}")
    print(f"Starting STEM Multimodal Analyzer on port {port}")
    print("Service: OCR + text normalization + DB-driven category/subject matching")
    print(f"{'=' * 60}\n")
    app.run(host='0.0.0.0', port=port, debug=True)