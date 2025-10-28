import io
import os
import tempfile
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

app = FastAPI(title="DeepSeek OCR PDF API")

# ======= Load model once =======
llm = LLM(
    model="/home/vms/bags/ocr-deepseek/models/deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    max_num_seqs=40,
    max_model_len=1400,
    gpu_memory_utilization=0.6,
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

# ======= PDF → images =======
def pdf_to_images_high_quality(pdf_data: bytes, dpi: int = 144) -> List[Image.Image]:
    """Convert PDF bytes to high-quality PIL Images"""
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

# ======= Resize =======
def resize_image(image: Image.Image, size=(1024, 1024)) -> Image.Image:
    return image.resize(size, Image.LANCZOS)

# ======= OCR endpoint =======
@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    pdf_data = await file.read()
    images = pdf_to_images_high_quality(pdf_data, dpi=144)
    resized_images = [resize_image(img) for img in images]

    prompt = "<image>\nFree OCR."
    model_inputs = [
        {"prompt": prompt, "multi_modal_data": {"image": img}}
        for img in resized_images
    ]

    # Generate output
    outputs = llm.generate(model_inputs, sampling_param)
    page_texts = [out.outputs[0].text.strip() for out in outputs]

    return {"num_pages": len(page_texts), "pages": page_texts}

if __name__ == "__main__":
    print("Starting DeepSeek-OCR API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5556,
        reload=False,
    )