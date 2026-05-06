"""YOLO检测模块 — ONNX Runtime 推理 + 自动模型下载（支持 HuggingFace/ModelScope）。"""
import os, sys, json, shutil
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

MODEL_PATH = Path(os.environ.get("CAMERA_MODEL_PATH", "yolov8n.onnx"))
CONF_THRESH = float(os.environ.get("DETECT_CONF_THRESH", 0.45))
IOU_THRESH  = float(os.environ.get("DETECT_IOU_THRESH", 0.5))
MODEL_URL   = os.environ.get("MODEL_DOWNLOAD_URL", "")

COCO_80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
           "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
           "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
           "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
           "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
           "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
           "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
           "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
           "scissors","teddy bear","hair drier","toothbrush"]

_session_cache = {}
_input_dtype = None

def _download_model(url: str, dest: Path) -> None:
    """从指定 URL 下载模型文件，支持断点续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载模型: {url}", file=sys.stderr)
    req = Request(url, headers={"User-Agent": "camera-yolo-logger/1.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print("模型下载完成", file=sys.stderr)

def _get_model_path() -> Path:
    """获取模型路径，如不存在且设置了 MODEL_URL 则自动下载。"""
    p = MODEL_PATH
    if not p.exists() and MODEL_URL:
        _download_model(MODEL_URL, p)
    return p

def _get_session():
    """加载 ONNX Runtime session 并缓存，自动检测输入 dtype。"""
    global _input_dtype
    model_path = _get_model_path()
    if model_path not in _session_cache:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _session_cache[model_path] = sess
        try:
            t = sess.get_inputs()[0].type.lower()
            if "float16" in t:  _input_dtype = np.float16
            elif "float" in t:  _input_dtype = np.float32
            elif "double" in t: _input_dtype = np.float64
            elif "int32" in t:  _input_dtype = np.int32
            elif "uint8" in t:  _input_dtype = np.uint8
            else:                _input_dtype = np.float32
        except Exception:
            _input_dtype = np.float32
    return _session_cache[model_path]

def _nms(boxes, scores, conf_thresh, iou_thresh, class_ids=None):
    """贪心 NMS，按类别抑制。"""
    order = sorted([i for i,s in enumerate(scores) if s>=conf_thresh], key=lambda i:scores[i], reverse=True)
    keep, suppressed = [], [False]*len(boxes)
    for i in order:
        if suppressed[i]: continue
        keep.append(i)
        xi1, yi1, xi2, yi2 = boxes[i]
        ai = (xi2-xi1)*(yi2-yi1)
        for j in order:
            if suppressed[j] or i==j: continue
            if class_ids and class_ids[i]!=class_ids[j]: continue
            xj1,yj1,xj2,yj2 = boxes[j]; aj = (xj2-xj1)*(yj2-yj1)
            ix1,iy1 = max(xi1,xj1), max(yi1,yj1)
            ix2,iy2 = min(xi2,xj2), min(yi2,yj2)
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            if inter/(ai+aj-inter+1e-6) > iou_thresh: suppressed[j]=True
    return keep

def detect(image_path: str | Path) -> str:
    """识别图片中的物体，返回逗号分隔的中文描述字符串。"""
    model_path = _get_model_path()
    if not model_path.exists():
        return "模型文件不存在，请设置环境变量 MODEL_DOWNLOAD_URL 自动下载，或手动放置 yolov8n.onnx"

    try:
        from PIL import Image
        sess = _get_session()
        img = Image.open(str(image_path)).convert("RGB")
        orig_w, orig_h = img.size
        img_np = np.array(img.resize((640,640), Image.BILINEAR), dtype=_input_dtype or np.float32)/255.0
        img_chw = np.transpose(img_np, (2,0,1))
        img_batch = np.expand_dims(img_chw, 0)

        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: img_batch})[0]
        dets = out[0].astype(np.float32)   # (84,8400)

        scores = dets[4:].max(axis=0)
        mask = scores > CONF_THRESH
        if not mask.any(): return "未识别到物体"

        class_ids = dets[4:].argmax(axis=0)[mask]
        boxes = dets[:4,mask]
        scores = scores[mask]
        ox = boxes[0]; oy = boxes[1]; bw = boxes[2]; bh = boxes[3]
        x1 = ((ox-bw/2)*orig_w/640).astype(int)
        y1 = ((oy-bh/2)*orig_h/640).astype(int)
        x2 = ((ox+bw/2)*orig_w/640).astype(int)
        y2 = ((oy+bh/2)*orig_h/640).astype(int)
        boxes = np.stack([x1,y1,x2,y2], axis=1).tolist()

        kept = _nms(boxes, scores.tolist(), CONF_THRESH, IOU_THRESH, class_ids.tolist())
        if not kept: return "未识别到物体"

        from collections import Counter
        names = Counter(COCO_80[class_ids[i]] for i in kept)
        return ", ".join(f"{cnt}个{name}" for name,cnt in names.items())

    except ImportError as e:
        return f"缺少依赖: {e}"
    except Exception as e:
        return f"识别出错: {e}"
