"""YOLO检测模块 — ONNX Runtime 推理 + Detector 类 + 数字裁剪放大。"""
from __future__ import annotations

import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from camera_yolo_logger.schemas import BBox, Detection, DetectionResult

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _download_model(url: str, dest: Path) -> None:
    """从指定 URL 下载 ONNX 模型，支持断点续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载模型: {url}", file=sys.stderr)
    try:
        req = Request(url, headers={"User-Agent": "camera-yolo-logger/1.1"})
        with urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        print("模型下载完成", file=sys.stderr)
    except Exception as exc:
        # 清理不完整的下载
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"模型下载失败: {exc}") from exc


def _nms(boxes, scores, conf_thresh, iou_thresh, class_ids=None):
    """按类别的贪心 NMS。"""
    order = sorted(
        [i for i, s in enumerate(scores) if s >= conf_thresh],
        key=lambda i: scores[i], reverse=True,
    )
    keep, suppressed = [], [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        xi1, yi1, xi2, yi2 = boxes[i]
        ai = (xi2 - xi1) * (yi2 - yi1)
        for j in order:
            if suppressed[j] or i == j:
                continue
            if class_ids and class_ids[i] != class_ids[j]:
                continue
            xj1, yj1, xj2, yj2 = boxes[j]
            aj = (xj2 - xj1) * (yj2 - yj1)
            ix1, iy1 = max(xi1, xj1), max(yi1, yj1)
            ix2, iy2 = min(xi2, xj2), min(yi2, yj2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / (ai + aj - inter + 1e-6) > iou_thresh:
                suppressed[j] = True
    return keep


class Detector:
    """YOLOv8 ONNX 检测器 — 支持类过滤、数字裁剪放大、模型自动下载。

    用法:
        d = Detector(model_path="yolov8n.onnx", class_filter=["person", "car"])
        result = d.detect("photo.jpg")
        print(result.summary)        # "2个人, 1辆车"
        print(result.to_json())      # 结构化 JSON
    """

    _session_cache: dict[Path, "ort.InferenceSession"] = {}
    _input_dtype: np.dtype | None = None

    def __init__(
        self,
        model_path: str | Path = "yolov8n.onnx",
        model_download_url: str = "",
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.5,
        class_filter: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.model_download_url = model_download_url
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_filter = class_filter

    # ── 模型加载 ─────────────────────────────────────────

    def _get_model_path(self) -> Path:
        p = self.model_path
        if not p.exists() and self.model_download_url:
            _download_model(self.model_download_url, p)
        return p

    def _get_session(self):
        import onnxruntime as ort
        model_path = self._get_model_path()
        if model_path not in self._session_cache:
            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self._session_cache[model_path] = sess
            try:
                t = sess.get_inputs()[0].type.lower()
                if "float16" in t:
                    self.__class__._input_dtype = np.float16
                elif "float" in t:
                    self.__class__._input_dtype = np.float32
                elif "double" in t:
                    self.__class__._input_dtype = np.float64
                elif "int32" in t:
                    self.__class__._input_dtype = np.int32
                elif "uint8" in t:
                    self.__class__._input_dtype = np.uint8
                else:
                    self.__class__._input_dtype = np.float32
            except Exception:
                self.__class__._input_dtype = np.float32
        return self._session_cache[model_path]

    # ── 推理 ─────────────────────────────────────────────

    def detect(self, image_path: str | Path) -> DetectionResult:
        """对单张图片做完整检测流水线，返回 DetectionResult。"""
        t0 = time.perf_counter()
        model_path = self._get_model_path()

        if not model_path.exists():
            return DetectionResult(
                success=False,
                objects=[],
                summary="模型文件不存在，请设置 MODEL_DOWNLOAD_URL 或放置 ONNX 模型",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            from PIL import Image
            sess = self._get_session()
            img = Image.open(str(image_path)).convert("RGB")
            orig_w, orig_h = img.size

            # 预处理
            dtype = self._input_dtype or np.float32
            img_np = np.array(img.resize((640, 640), Image.BILINEAR), dtype=dtype) / 255.0
            img_batch = np.expand_dims(np.transpose(img_np, (2, 0, 1)), 0)

            # ONNX 推理
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: img_batch})[0]
            dets = out[0].astype(np.float32)  # (84, 8400)

            # 后处理
            scores = dets[4:].max(axis=0)
            mask = scores > self.conf_thresh
            if not mask.any():
                return DetectionResult(
                    success=True, objects=[], summary="未识别到物体",
                    image_path=str(image_path), image_size=(orig_w, orig_h),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            class_ids = dets[4:].argmax(axis=0)[mask]
            raw_boxes = dets[:4, mask]
            raw_scores = scores[mask]

            # 坐标转换：模型 (640,640) → 原图
            ox, oy = raw_boxes[0], raw_boxes[1]
            bw, bh = raw_boxes[2], raw_boxes[3]
            x1 = ((ox - bw / 2) * orig_w / 640).astype(int)
            y1 = ((oy - bh / 2) * orig_h / 640).astype(int)
            x2 = ((ox + bw / 2) * orig_w / 640).astype(int)
            y2 = ((oy + bh / 2) * orig_h / 640).astype(int)
            boxes_list = np.stack([x1, y1, x2, y2], axis=1).tolist()

            # NMS
            kept = _nms(boxes_list, raw_scores.tolist(), self.conf_thresh, self.iou_thresh,
                        class_ids.tolist())
            if not kept:
                return DetectionResult(
                    success=True, objects=[], summary="未识别到物体",
                    image_path=str(image_path), image_size=(orig_w, orig_h),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            # 构建 Detection 列表 + 类过滤
            objects = []
            for i in kept:
                class_name = COCO_80[int(class_ids[i])]
                if self.class_filter and class_name not in self.class_filter:
                    continue
                objects.append(Detection(
                    class_name=class_name,
                    class_id=int(class_ids[i]),
                    confidence=float(raw_scores[i]),
                    bbox=BBox(*[int(v) for v in boxes_list[i]]),
                ))

            if not objects:
                return DetectionResult(
                    success=True, objects=[], summary="未识别到目标类别",
                    image_path=str(image_path), image_size=(orig_w, orig_h),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            # 中文摘要
            names = Counter(o.class_name for o in objects)
            summary = ", ".join(f"{cnt}个{name}" for name, cnt in names.items())

            return DetectionResult(
                success=True, objects=objects, summary=summary,
                image_path=str(image_path), image_size=(orig_w, orig_h),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        except ImportError as e:
            return DetectionResult(
                success=False, objects=[], summary=f"缺少依赖: {e}",
                error=str(e), elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            return DetectionResult(
                success=False, objects=[], summary=f"识别出错: {e}",
                error=str(e), elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def detect_summary(self, image_path: str | Path) -> str:
        """向后兼容：返回中文摘要字符串。"""
        return self.detect(image_path).summary

    def detect_crop(
        self, image_path: str | Path, target_class: str = "person", margin: float = 0.2,
    ) -> DetectionResult:
        """两阶段检测实现「数字对焦放大」:

        1. 全帧检测 → 找到最大目标物体
        2. 裁剪到目标区域 → 二次检测获得更高精度结果
        """
        t0 = time.perf_counter()

        # 阶段 1：全帧检测
        full_result = self.detect(image_path)
        if not full_result.objects:
            full_result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return full_result

        # 找到目标类别中面积最大的检测结果
        targets = [o for o in full_result.objects if o.class_name == target_class]
        if not targets:
            targets = full_result.objects  # 回退到所有检测
        target = max(targets, key=lambda o: o.bbox.area)

        # 阶段 2：裁剪放大
        try:
            from PIL import Image
            img = Image.open(str(image_path))
            w, h = img.size
            mx = int(w * margin)
            my = int(h * margin)
            crop_box = (
                max(0, target.bbox.x1 - mx),
                max(0, target.bbox.y1 - my),
                min(w, target.bbox.x2 + mx),
                min(h, target.bbox.y2 + my),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                full_result.elapsed_ms = (time.perf_counter() - t0) * 1000
                return full_result

            cropped = img.crop(crop_box)
            crop_path = Path(str(image_path)).with_suffix(".crop.jpg")
            cropped.save(str(crop_path))

            # 二次检测
            crop_result = self.detect(crop_path)
            crop_path.unlink(missing_ok=True)  # 清理临时文件
            crop_result.elapsed_ms = (time.perf_counter() - t0) * 1000
            crop_result.image_path = str(image_path)
            return crop_result
        except Exception:
            full_result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return full_result

    @classmethod
    def from_settings(cls, settings) -> "Detector":
        """从 Settings 实例创建 Detector。"""
        return cls(
            model_path=settings.model_path,
            model_download_url=settings.model_download_url,
            conf_thresh=settings.conf_thresh,
            iou_thresh=settings.iou_thresh,
            class_filter=settings.class_filter,
        )


# ── 向后兼容的模块级函数 ──────────────────────────────────


def detect(image_path: str | Path) -> str:
    """向后兼容：返回中文摘要字符串。"""
    return Detector().detect_summary(image_path)
