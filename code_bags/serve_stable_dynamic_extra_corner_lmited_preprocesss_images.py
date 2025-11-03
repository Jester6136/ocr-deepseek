import datetime
import io
import json
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, TypeVar, Generic
import asyncio
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, File, Path, UploadFile, HTTPException
import uvicorn
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

app = FastAPI(title="DeepSeek OCR PDF API with Auto-Batching and Bottom-Left OCR (Improved Batching)")

FILE_PREPROCESS_LIMIT = 32
MAX_FILE_CONCURRENT = 32
BATCH_SIZE_MODEL = MAX_FILE_CONCURRENT * 4 + 20
MAX_LEN = 1300
# ======= Load model once =======
llm = LLM(
    model="models/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    max_num_seqs=BATCH_SIZE_MODEL,
    max_model_len=MAX_LEN,
    gpu_memory_utilization=0.29,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# ======= OCR params =======
sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_LEN,
    extra_args=dict(
        ngram_size=3,
        window_size=6,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

# ======= Async PDF → images =======
async def pdf_to_images_high_quality_async(pdf_data: bytes, dpi: int = 144) -> List[Image.Image]:
    """Convert PDF bytes to high-quality PIL Images asynchronously using ThreadPoolExecutor"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        images = await loop.run_in_executor(
            executor,
            _sync_pdf_to_images_helper,
            pdf_data,
            dpi
        )
    return images

def _sync_pdf_to_images_helper(pdf_data: bytes, dpi: int = 144):
    """Convert PDF bytes to list of PIL Images without temp files"""
    if len(pdf_data) < 4 or pdf_data[:4] != b'%PDF':
        raise ValueError("Invalid PDF: missing %PDF header")

    try:
        # Open directly from memory
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"PyMuPDF failed to open PDF stream: {e}") from e

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)

    doc.close()
    return images

# ======= Async Resize =======
async def resize_images_async(images: List[Image.Image], size=(1024, 1024)) -> List[Image.Image]:
    """Resize a list of images asynchronously using ThreadPoolExecutor"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        resized_images = await loop.run_in_executor(
            executor,
            _sync_resize_helper,
            images,
            size
        )
    return resized_images

def _sync_resize_helper(images: List[Image.Image], size):
    """Helper function to run synchronous image resizing in a thread pool"""
    return [img.resize(size, Image.LANCZOS) for img in images]

# ======= Async Extract Bottom-Left Region =======
async def extract_bottom_left_region_async(images: List[Image.Image]) -> List[Image.Image]:
    """Extract the bottom-left region (3/10 width, 1/3 height) from a list of images asynchronously using ThreadPoolExecutor"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        cropped_images = await loop.run_in_executor(
            executor,
            _sync_extract_bottom_left_helper,
            images
        )
    return cropped_images

def _sync_extract_bottom_left_helper(images: List[Image.Image]):
    """Helper function to run synchronous image cropping in a thread pool"""
    cropped_list = []
    for page_idx, img in enumerate(images):
        width, height = img.size
        left   = 0
        right  = int(width)
        bottom = height
        top    = height - int(height * 0.18)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_list.append(cropped_img)
    return cropped_list


def batch_process_images(
    images_per_file_list: List[Tuple[List[Image.Image], List[Image.Image]]], # [(main_images_file1, extra_images_file1), ...]
    batch_size: int = BATCH_SIZE_MODEL
) -> Tuple[List[List[str]], List[List[str]]]: # Returns (main OCR results per file, extra OCR results per file)
    """
    Xử lý batch các ảnh từ nhiều file. Gộp main và extra images vào một batch duy nhất để tối ưu hiệu suất.
    images_per_file_list: [(main_imgs_file1, extra_imgs_file1), (main_imgs_file2, extra_imgs_file2), ...]
    Trả về: ([main_texts_file1, main_texts_file2], [extra_texts_file1, extra_texts_file2])
    """
    all_images = []
    all_prompts = []
    # Lưu thông tin để phân tách kết quả sau này
    file_info = [] # [(file_idx, 'main'/'extra', page_idx), ...]

    for file_idx, (main_images, extra_images) in enumerate(images_per_file_list):
        # Thêm ảnh chính (main)
        for page_idx, img in enumerate(main_images):
            all_images.append(img)
            all_prompts.append("<image>\nConvert the document text only.")
            file_info.append((file_idx, 'main', page_idx))

        # Thêm ảnh phụ (extra)
        for page_idx, img in enumerate(extra_images):
            all_images.append(img)
            all_prompts.append("<image>\nConvert the document text only.")
            file_info.append((file_idx, 'extra', page_idx))

    all_outputs = []
    if all_images:
        model_inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for prompt, img in zip(all_prompts, all_images)
        ]

        for i in range(0, len(model_inputs), batch_size):
            batch_inputs = model_inputs[i:i + batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_param)
            all_outputs.extend(batch_outputs)

    # Khởi tạo danh sách kết quả cho từng file
    num_files = len(images_per_file_list)
    main_results: List[List[str]] = [[] for _ in range(num_files)]
    extra_results: List[List[str]] = [[] for _ in range(num_files)]

    # Phân phối kết quả dựa trên file_info
    for info, output in zip(file_info, all_outputs):
        file_idx, img_type, page_idx = info
        text = output.outputs[0].text.strip()

        if img_type == 'main':
            # Đảm bảo danh sách đủ dài
            while len(main_results[file_idx]) <= page_idx:
                main_results[file_idx].append("")
            main_results[file_idx][page_idx] = text
        elif img_type == 'extra':
            # Đảm bảo danh sách đủ dài
            while len(extra_results[file_idx]) <= page_idx:
                extra_results[file_idx].append("")
            extra_results[file_idx][page_idx] = text

    return main_results, extra_results


# ======= Async batch preprocessing =======
async def _preprocess_batch_files_async(pdf_datas: List[bytes]) -> List[Tuple[List[Image.Image], List[Image.Image]]]:
    """Tiền xử lý batch các file: PDF -> Images -> Resize -> Crop Bottom-Left"""
    # Chia nhỏ batch để không quá tải CPU
    batch_size = FILE_PREPROCESS_LIMIT
    all_results = []
    
    for i in range(0, len(pdf_datas), batch_size):
        batch_pdf_datas = pdf_datas[i:i + batch_size]
        
        # Xử lý async cho từng file trong batch nhỏ
        preprocess_tasks = [
            asyncio.create_task(_preprocess_single_file_internal_async(data))
            for data in batch_pdf_datas
        ]
        batch_results = await asyncio.gather(*preprocess_tasks)
        all_results.extend(batch_results)
        
        # Thêm delay nhỏ giữa các batch nhỏ để giảm áp lực CPU
        if i + batch_size < len(pdf_datas):
            await asyncio.sleep(0.1)
    
    return all_results

async def _preprocess_single_file_internal_async(pdf_data: bytes) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Tiền xử lý một file: PDF -> Images -> Resize -> Crop Bottom-Left (chỉ 2 trang đầu)."""
    images = await pdf_to_images_high_quality_async(pdf_data, dpi=144)
    resized_images = await resize_images_async(images)

    # Chỉ crop 2 trang đầu tiên
    first_two = resized_images[:2]
    cropped_images = []
    if first_two:
        cropped_images = await extract_bottom_left_region_async(first_two)

    return resized_images, cropped_images

# ======= Auto-Batching Processor for Single PDF Requests =======
# Define request structure
from pydantic import BaseModel

class OCRBatchProcessor:
    def __init__(self, max_batch_size: int = 10, batch_timeout: float = 0.5, check_delay: float = 0.1):
        self.queue = asyncio.Queue()
        self.results: Dict[str, Dict] = {} # {request_id: {'status': 'pending'/'done', 'result': ...}}
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.check_delay = check_delay
        self._processing_lock = asyncio.Lock()
        self.last_batch_time = time.time()

    async def add_request(self, pdf_data: bytes, filename: str) -> str:
        """Add a single PDF request to the queue and return a request ID."""
        request_id = str(uuid.uuid4())
        self.results[request_id] = {'status': 'pending', 'result': None}
        await self.queue.put((request_id, filename, pdf_data))
        return request_id

    async def get_result(self, request_id: str) -> Dict:
        """Poll for the result of a specific request."""
        while self.results.get(request_id, {}).get('status') != 'done':
            await asyncio.sleep(0.05)  # Poll every 50ms
        return self.results[request_id]['result']

    async def process_batch_loop(self):
        """Main loop to continuously check and process batches."""
        while True:
            await self._check_and_process_batch()
            await asyncio.sleep(self.check_delay)

    async def _check_and_process_batch(self):
        """Check conditions and process a batch if needed."""
        async with self._processing_lock:
            current_time = time.time()
            elapsed = current_time - self.last_batch_time
            qsize = self.queue.qsize()

            if qsize == 0:
                return

            # Condition 1: Batch size reached
            if qsize >= self.max_batch_size:
                print(f"Processing batch: max_batch_size ({self.max_batch_size}) reached. Queue size: {qsize}")
                await self._process_current_batch()
                self.last_batch_time = time.time()
                return

            # Condition 2: Timeout reached
            if elapsed >= self.batch_timeout:
                print(f"Processing batch: timeout ({self.batch_timeout}s) reached. Queue size: {qsize}")
                # Coalesce delay
                await asyncio.sleep(self.check_delay)
                await self._process_current_batch()
                self.last_batch_time = time.time()

    async def _process_current_batch(self):
        """Process all requests currently in the queue."""
        batch_items = []
        while not self.queue.empty():
            item = await self.queue.get()
            batch_items.append(item)
            if len(batch_items) >= self.max_batch_size:
                break

        if not batch_items:
            return

        print(f"Processing batch of {len(batch_items)} files...")

        # Extract data
        request_ids, filenames, pdf_datas = zip(*batch_items)

        # Preprocess batch asynchronously (get both main and extra images) - Batch riêng
        all_processed_data = await _preprocess_batch_files_async(pdf_datas)

        # Batch process OCR for both main and extra images (Gộp vào một batch)
        main_ocr_results, extra_ocr_results = batch_process_images(
            all_processed_data # Gửi danh sách tuple [(main_imgs, extra_imgs), ...]
        )

        # Distribute results back to individual requests
        for i, (req_id, filename, main_page_texts, extra_page_texts) in enumerate(zip(request_ids, filenames, main_ocr_results, extra_ocr_results)):
            self.results[req_id]['status'] = 'done'
            self.results[req_id]['result'] = {
                "filename": filename,
                "num_pages": len(main_page_texts), # Main OCR determines page count
                "pages": main_page_texts,
                "extra_ocr_bottom_left": extra_page_texts # Add the extra OCR results
            }
        print(f"Completed batch of {len(batch_items)} files.")

# Initialize the batch processor
ocr_batch_processor = OCRBatchProcessor(max_batch_size=MAX_FILE_CONCURRENT, batch_timeout=2.0, check_delay=0.1)  # Giảm batch size

# Start the batch processing loop in the background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ocr_batch_processor.process_batch_loop())

# ======= New Single File OCR endpoint using auto-batching =======
async def append_ocr_result(result: dict, filename: str, request_id: str):
    """Hàm bất đồng bộ để thêm result vào file duy nhất"""
    import json
    log_entry = {
        "result": result
    }
    
    def append_to_file():
        with open("ocr_results.txt", 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')
    
    # Chạy trong thread pool để không blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, append_to_file)

@app.post("/ocr/pdf")
async def ocr_pdf_single(file: UploadFile = File(...)):
    pdf_data = await file.read()
    request_id = await ocr_batch_processor.add_request(pdf_data, file.filename)
    result = await ocr_batch_processor.get_result(request_id)
    
    # Ghi result vào file duy nhất bất đồng bộ
    asyncio.create_task(append_ocr_result(result, file.filename, request_id))
    return result

@app.post("/ocr/pdf/batch/async")
async def ocr_pdf_batch_async(files: List[UploadFile] = File(...)):
    file_datas = [await f.read() for f in files]
    # Tiền xử lý batch (giữ nguyên logic hiện tại)
    all_processed_data = await _preprocess_batch_files_async(file_datas)

    # Xử lý OCR batch lớn (gộp main + extra)
    main_batch_results, extra_batch_results = batch_process_images(
        all_processed_data,
        batch_size=BATCH_SIZE_MODEL  # Dùng full batch size
    )

    for idx in range(len(main_batch_results)):
        num_pages = len(main_batch_results[idx])
        extra_pages = extra_batch_results[idx]
        padded_extra = ["" for _ in range(num_pages)]
        for j in range(min(2, len(extra_pages))):
            padded_extra[j] = extra_pages[j]
        extra_batch_results[idx] = padded_extra
    
    final_results = []
    for idx, (main_page_texts, extra_page_texts) in enumerate(zip(main_batch_results, extra_batch_results)):
        final_results.append({
            "filename": files[idx].filename,
            "num_pages": len(main_page_texts),
            "pages": main_page_texts,
            "extra_ocr_bottom_left": extra_page_texts
        })

    return {"files": final_results, "batch_size": len(files), "status": "completed"}

@app.post("/ocr/image")
async def ocr_image_single(file: UploadFile = File(...)):
    """OCR endpoint for a single image file."""
    # Read the uploaded image
    image_data = await file.read()
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Async preprocessing: resize + bottom-left crop
    resized_images = await resize_images_async([img])
    cropped_images = await extract_bottom_left_region_async(resized_images)

    # Run OCR in batch (main image + cropped)
    main_ocr_results, extra_ocr_results = batch_process_images(
        [(resized_images, cropped_images)],
        batch_size=BATCH_SIZE_MODEL
    )

    result = {
        "filename": file.filename,
        "num_images": 1,
        "pages": main_ocr_results[0],
        "extra_ocr_bottom_left": extra_ocr_results[0]
    }

    # Append result to the same log file asynchronously
    asyncio.create_task(append_ocr_result(result, file.filename, str(uuid.uuid4())))

    return result
if __name__ == "__main__":
    print("Starting DeepSeek-OCR API server with Auto-Batching and Bottom-Left OCR (Improved Batching)...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5556,
        reload=False,
        workers=1
    )