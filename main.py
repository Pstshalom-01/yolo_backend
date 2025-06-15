import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr

# Load YOLOv8 model for candlestick detection
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8_candles.pt")
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
    model = None

# Initialize OCR reader (English language)
reader = easyocr.Reader(['en'], gpu=False)

app = FastAPI(title="AI Chart Scanner Backend", description="Detects candlestick patterns and extracts chart info", version="1.0")

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/scan")
async def scan_chart(file: UploadFile = File(...)):
    """
    Scan an uploaded chart image for candlestick patterns and indicators.
    Accepts an image file (screenshot or camera capture of a chart) and returns a JSON result.
    """
    # Read image file content
    image_data = await file.read()
    # Convert to OpenCV image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Unable to decode image"}

    # Run YOLO model to detect candlesticks (if model is loaded)
    candlesticks = []
    if model is not None:
        results = model.predict(source=img, verbose=False)
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    x1, y1, x2, y2 = map(int, coords)
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    corners = [int(gray[0,0]), int(gray[0,-1]), int(gray[-1,0]), int(gray[-1,-1])]
                    bg_brightness = float(np.mean(corners))
                    proc = gray.copy()
                    if bg_brightness > 127:
                        proc = cv2.bitwise_not(proc)
                    _, bw = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    rows_sum = np.sum(bw == 255, axis=1)
                    max_width = int(np.max(rows_sum)) if rows_sum.size > 0 else 0
                    body_top_rel = None
                    body_bottom_rel = None
                    if max_width > 0:
                        width_thresh = int(max_width * 0.8)
                        body_indices = np.where(rows_sum >= width_thresh)[0]
                        if body_indices.size > 0:
                            body_top_rel = int(body_indices[0])
                            body_bottom_rel = int(body_indices[-1])
                    if body_top_rel is None or body_bottom_rel is None:
                        body_indices = np.where(rows_sum > 0)[0]
                        if body_indices.size > 0:
                            body_top_rel = int(body_indices[0])
                            body_bottom_rel = int(body_indices[0])
                        else:
                            body_top_rel = 0
                            body_bottom_rel = 0
                    body_top_y = y1 + body_top_rel
                    body_bottom_y = y1 + body_bottom_rel
                    # Determine bull/bear via color analysis
                    mask = (bw == 255)
                    bull_flag = True
                    if mask.any():
                        # Compute mean color of candle pixels
                        candle_pixels = roi[mask]
                        if candle_pixels.size > 0:
                            mean_color = candle_pixels.mean(axis=0)
                            # mean_color is [B, G, R]
                            if mean_color[2] > mean_color[1]:
                                bull_flag = False
                    
                    candlesticks.append({
                        "x_center": (x1 + x2) / 2.0,
                        "open_y": body_bottom_y if bull_flag else body_top_y,
                        "close_y": body_top_y if bull_flag else body_bottom_y,
                        "high_y": y1,
                        "low_y": y2,
                        "bull": bull_flag,
                        "conf": conf,
                        "body_top_y": body_top_y,
                        "body_bottom_y": body_bottom_y
                    })
    candlesticks.sort(key=lambda c: c["x_center"])

    pattern_name = None
    signal_type = None
    pattern_conf = 0.0
    latest_index = -1
    # Single-candle pattern detection
    for idx, candle in enumerate(candlesticks):
        body_height = abs(candle["body_bottom_y"] - candle["body_top_y"])
        upper_wick = candle["body_top_y"] - candle["high_y"]
        lower_wick = candle["low_y"] - candle["body_bottom_y"]
        is_doji = body_height <= 2
        is_hammer = lower_wick > 0 and lower_wick >= 2 * body_height and lower_wick >= 2 * upper_wick
        is_pinbar = upper_wick > 0 and upper_wick >= 2 * body_height and upper_wick >= 2 * lower_wick
        pattern = None
        stype = None
        conf = 0.0
        if is_hammer:
            pattern = "Hammer"
            stype = "Bullish"
            conf = 0.9
        elif is_pinbar:
            pattern = "Pin Bar"
            stype = "Bearish"
            conf = 0.9
        elif is_doji:
            pattern = "Doji"
            stype = "Neutral"
            conf = 0.8
        if pattern and idx > latest_index:
            latest_index = idx
            pattern_name = pattern
            signal_type = stype
            pattern_conf = conf
    # Multi-candle pattern detection (Engulfing)
    for idx in range(1, len(candlesticks)):
        prev = candlesticks[idx - 1]
        curr = candlesticks[idx]
        if not prev["bull"] and curr["bull"]:
            if curr["open_y"] >= prev["close_y"] and curr["close_y"] <= prev["open_y"]:
                pattern = "Engulfing"
                stype = "Bullish"
                prev_body = abs(prev["close_y"] - prev["open_y"])
                curr_body = abs(curr["close_y"] - curr["open_y"])
                ratio = curr_body / (prev_body + 1e-6)
                conf = min(0.99, 0.8 + min(ratio, 3.0) * 0.1)
                if idx > latest_index:
                    latest_index = idx
                    pattern_name = pattern
                    signal_type = stype
                    pattern_conf = conf
        if prev["bull"] and not curr["bull"]:
            if curr["open_y"] <= prev["close_y"] and curr["close_y"] >= prev["open_y"]:
                pattern = "Engulfing"
                stype = "Bearish"
                prev_body = abs(prev["close_y"] - prev["open_y"])
                curr_body = abs(curr["close_y"] - curr["open_y"])
                ratio = curr_body / (prev_body + 1e-6)
                conf = min(0.99, 0.8 + min(ratio, 3.0) * 0.1)
                if idx > latest_index:
                    latest_index = idx
                    pattern_name = pattern
                    signal_type = stype
                    pattern_conf = conf
    # OCR extraction
    ocr_texts = []
    try:
        ocr_results = reader.readtext(img)
        for (_, text, _) in ocr_results:
            if text:
                ocr_texts.append(text)
    except Exception as e:
        print(f"OCR error: {e}")
    pair = None
    timeframe = None
    found_rsi = False
    found_volume = False
    import re
    for text in ocr_texts:
        t_upper = text.upper()
        if "RSI" in t_upper:
            found_rsi = True
        if "VOL" in t_upper or "VOLUME" in t_upper:
            found_volume = True
        if timeframe is None:
            compact = text.replace(" ", "")
            m = re.fullmatch(r"(\d+)([mMhHdDwW])", compact)
            if m:
                timeframe = (m.group(1) + m.group(2)).upper()
                if text.strip() != timeframe:
                    pair_candidate = text.replace(timeframe, "").strip(" -:/")
                    if len(pair_candidate) >= 3:
                        pair = pair_candidate
        if pair is None:
            if len(text) >= 6 and text.upper() == text and re.match(r"^[A-Z0-9/]+$", text):
                ignore = ["VOLUME", "RSI", "MACD", "BINANCE", "COINBASE", "FUTURES", "PERPETUAL", "USDT"]
                if t_upper not in ignore:
                    if "/" in text or len(text) > 5:
                        pair = text
    if pair is None and timeframe:
        try:
            idx = next(i for i,t in enumerate(ocr_texts) if timeframe in t)
            if idx > 0:
                cand = ocr_texts[idx-1].replace(":", "").strip()
                if len(cand) >= 3 and cand.upper() == cand:
                    pair = cand
        except StopIteration:
            pass
    if pair is None:
        pair = "Unknown"
    if timeframe is None:
        timeframe = "Unknown"
    notes = []
    if found_rsi:
        notes.append("RSI indicator detected")
    if found_volume:
        notes.append("Volume indicator detected")
    notes_str = "; ".join(notes)
    if pattern_name is None:
        pattern_name = "None"
        signal_type = "None"
        pattern_conf = 0.0
        if notes_str:
            notes_str = "No significant pattern found; " + notes_str
        else:
            notes_str = "No significant pattern found"
    result = {
        "signal_type": signal_type,
        "pattern": pattern_name,
        "confidence": round(pattern_conf, 2),
        "pair": pair,
        "timeframe": timeframe,
        "notes": notes_str
    }
    return JSONResponse(content=result)
