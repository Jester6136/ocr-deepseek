import asyncio
import aiohttp
import time
import os
from statistics import mean

# ========== CONFIG ==========
API_URL = "http://127.0.0.1:5556/ocr/pdf"
PDF_PATH = "/home/vms/bags/ocr-deepseek/samples/11950-GCN-BK 763735.pdf"

# Số lượng request chạy song song
CONCURRENCY = 10
# Tổng số request
NUM_REQUESTS = 40


async def send_request(session, pdf_bytes, idx):
    """Gửi 1 request OCR PDF"""
    t0 = time.perf_counter()
    data = aiohttp.FormData()
    data.add_field(
        "file",
        pdf_bytes,
        filename=f"sample_{idx}.pdf",
        content_type="application/pdf"
    )

    try:
        async with session.post(API_URL, data=data) as resp:
            dt = time.perf_counter() - t0
            if resp.status != 200:
                print(f"[{idx:02d}] ❌ HTTP {resp.status} ({dt:.2f}s)")
                return None
            result = await resp.json()
            num_pages = result.get("num_pages", 0)
            print(f"[{idx:02d}] ✅ {num_pages} pages in {dt:.2f}s")
            return (dt, num_pages)
    except Exception as e:
        print(f"[{idx:02d}] ❌ Error: {e}")
        return None


async def benchmark():
    """Chạy benchmark nhiều request song song"""
    if not os.path.exists(PDF_PATH):
        print(f"❌ File not found: {PDF_PATH}")
        return

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    print(f"🚀 Starting benchmark")
    print(f"→ Requests: {NUM_REQUESTS}  |  Concurrency: {CONCURRENCY}")
    print(f"→ PDF: {os.path.basename(PDF_PATH)} ({len(pdf_bytes)/1024:.1f} KB)\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with aiohttp.ClientSession() as session:

        async def bound_request(i):
            async with semaphore:
                res = await send_request(session, pdf_bytes, i)
                if res:
                    results.append(res)

        t_start = time.perf_counter()
        tasks = [bound_request(i) for i in range(NUM_REQUESTS)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    # ======= Kết quả tổng kết =======
    if not results:
        print("\n⚠️ No successful requests.")
        return

    durations = [r[0] for r in results]
    pages = [r[1] for r in results if r and r[1] is not None]

    total_pages = sum(pages)
    total_time = t_end - t_start
    avg_latency = mean(durations)
    rps = len(results) / total_time
    pps = total_pages / total_time if total_pages else 0

    print("\n===== 🧾 Benchmark Summary =====")
    print(f"✅ Completed: {len(results)}/{NUM_REQUESTS}")
    print(f"⏱️ Avg latency: {avg_latency:.2f} s")
    print(f"⚡ Requests/sec: {rps:.2f}")
    print(f"📄 Pages/sec: {pps:.2f}")
    print(f"⏰ Total elapsed: {total_time:.2f} s")
    print("===============================\n")


if __name__ == "__main__":
    asyncio.run(benchmark())
