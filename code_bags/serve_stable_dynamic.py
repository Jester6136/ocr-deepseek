import io
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
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

app = FastAPI(title="DeepSeek OCR PDF API with Auto-Batching")

# ======= Load model once =======
llm = LLM(
    model="/home/vms/bags/ocr-deepseek/models/deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    max_num_seqs=60,
    max_model_len=1400,
    gpu_memory_utilization=0.8,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# ======= OCR params =======
sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=1400,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
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

def _sync_pdf_to_images_helper(pdf_data: bytes, dpi: int):
    """Helper function to run synchronous PDF processing in a thread pool"""
    images = []
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name

    try:
        pdf_document = fitz.open(temp_pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
        pdf_document.close()
    finally:
        os.unlink(temp_pdf_path)
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

# ======= Batch processing helper =======
def batch_process_images(
    images_list: List[List[Image.Image]],
    batch_size: int = 16  # Điều chỉnh kích thước batch theo khả năng GPU
) -> List[List[str]]:
    """
    Xử lý batch các ảnh từ nhiều file.
    images_list: [[img1_file1, img2_file1], [img1_file2], ...]
    Trả về: [[text1_file1, text2_file1], [text1_file2], ...]
    """
    # Flatten tất cả ảnh và ghi nhớ index file gốc
    flattened_images = []
    file_indices = []
    page_indices = []

    for file_idx, images in enumerate(images_list):
        for page_idx, img in enumerate(images):
            flattened_images.append(img)
            file_indices.append(file_idx)
            page_indices.append(page_idx)

    if not flattened_images:
        return [[] for _ in images_list]

    # Tạo input cho vLLM
    prompts = ["<image>\nFree OCR."] * len(flattened_images)
    model_inputs = [
        {"prompt": prompt, "multi_modal_data": {"image": img}}
        for prompt, img in zip(prompts, flattened_images)
    ]

    # Chia thành các batch nhỏ hơn nếu cần
    all_outputs = []
    for i in range(0, len(model_inputs), batch_size):
        batch_inputs = model_inputs[i:i + batch_size]
        batch_outputs = llm.generate(batch_inputs, sampling_param)
        all_outputs.extend(batch_outputs)

    # Gom kết quả theo file gốc
    results: List[List[str]] = [[] for _ in range(len(images_list))]
    for i, output in enumerate(all_outputs):
        file_idx = file_indices[i]
        text = output.outputs[0].text.strip()
        results[file_idx].append(text)

    return results

# ======= Auto-Batching Processor for Single PDF Requests =======
# Define request structure
from pydantic import BaseModel

class SinglePDFRequest(BaseModel):
    filename: str
    pdf_data: bytes

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
        print(f"Added request {request_id} for file {filename} to queue. Queue size: {self.queue.qsize()}")
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

        # Preprocess batch asynchronously
        preprocess_tasks = [
            asyncio.create_task(_preprocess_single_file_async(data))
            for data in pdf_datas
        ]
        all_resized_images_lists = await asyncio.gather(*preprocess_tasks)

        # Batch process OCR
        batch_results = batch_process_images(all_resized_images_lists)

        # Distribute results back to individual requests
        for i, (req_id, filename, page_texts) in enumerate(zip(request_ids, filenames, batch_results)):
            self.results[req_id]['status'] = 'done'
            self.results[req_id]['result'] = {
                "filename": filename,
                "num_pages": len(page_texts),
                "pages": page_texts
            }
        print(f"Completed batch of {len(batch_items)} files.")


# Initialize the batch processor
ocr_batch_processor = OCRBatchProcessor(max_batch_size=24, batch_timeout=2.0, check_delay=0.1)

# Start the batch processing loop in the background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ocr_batch_processor.process_batch_loop())

# ======= New Single File OCR endpoint using auto-batching =======
@app.post("/ocr/pdf")
async def ocr_pdf_single(file: UploadFile = File(...)):
    pdf_data = await file.read()
    request_id = await ocr_batch_processor.add_request(pdf_data, file.filename)
    result = await ocr_batch_processor.get_result(request_id)
    return result

# ======= Existing Batch OCR endpoint (still available) =======
@app.post("/ocr/pdf/batch")
async def ocr_pdf_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Đọc dữ liệu file
    file_datas = [await f.read() for f in files]

    # Tiền xử lý bất đồng bộ cho từng file (PDF -> Images -> Resize)
    preprocess_tasks = [
        asyncio.create_task(_preprocess_single_file_async(data))
        for data in file_datas
    ]
    all_resized_images_list = await asyncio.gather(*preprocess_tasks)

    # Gom tất cả ảnh lại để xử lý batch OCR
    batch_results = batch_process_images(all_resized_images_list)

    # Trả về kết quả theo từng file
    final_results = []
    for idx, page_texts in enumerate(batch_results):
        final_results.append({
            "filename": files[idx].filename,
            "num_pages": len(page_texts),
            "pages": page_texts
        })

    return {"files": final_results}

async def _preprocess_single_file_async(pdf_data: bytes):
    """Tiền xử lý một file: PDF -> Images -> Resize"""
    images = await pdf_to_images_high_quality_async(pdf_data, dpi=144)
    resized_images = await resize_images_async(images)
    return resized_images


if __name__ == "__main__":
    print("Starting DeepSeek-OCR API server with Auto-Batching...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5556,
        reload=False,
        workers=1 # Quan trọng cho vLLM
    )