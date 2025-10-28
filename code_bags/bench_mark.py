import aiohttp
import asyncio
import time
import json
from pathlib import Path

# Cấu hình
API_URL = "http://localhost:5556/ocr/pdf"  # Endpoint mới, xử lý đơn file nhưng có auto-batch
PDF_FILE_PATH = "/home/vms/bags/ocr-deepseek/samples/11950-GCN-BK 763735.pdf"
NUM_REQUESTS = 50  # Số lượng request gửi đi
CONCURRENT_REQUESTS = 24  # Số lượng request gửi đồng thời (điều chỉnh để test tải)
OUTPUT_FILE = "benchmark_results.jsonl" # Tên file output

async def send_single_request(session, file_path, request_idx):
    """Gửi một request đến API OCR đơn file (sẽ được auto-batch bên server)"""
    try:
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            # Gửi file đơn lẻ, endpoint /ocr/pdf nhận file đơn
            data.add_field('file', f, filename=Path(file_path).name, content_type='application/pdf')

            start_time = time.perf_counter()
            async with session.post(API_URL, data=data) as response:
                response_text = await response.text()
                response_time = time.perf_counter() - start_time
                status = response.status
                # Trả về kết quả chi tiết cho phân tích
                return {
                    "request_idx": request_idx,
                    "status": status,
                    "response_time": response_time,
                    "response_body": response_text, # Raw text response
                    "success": 200 <= status < 300
                }
    except Exception as e:
        print(f"Request {request_idx} failed with exception: {e}")
        # Trả về thông tin lỗi để phân tích sau
        return {
            "request_idx": request_idx,
            "status": None,
            "response_time": time.perf_counter(),
            "response_body": str(e),
            "success": False
        }


async def run_benchmark():
    """Chạy benchmark với số lượng request và mức độ đồng thời cụ thể"""
    print(f"Starting benchmark: {NUM_REQUESTS} requests, {CONCURRENT_REQUESTS} concurrent")
    print(f"Target URL: {API_URL}")
    print(f"Using file: {PDF_FILE_PATH}")
    print(f"Auto-batching config: max_batch_size=5, timeout=1.0s, check_delay=0.1s")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print("-" * 50)

    # Tăng timeout cho client để chờ OCR hoàn thành
    timeout = aiohttp.ClientTimeout(total=600)  # Timeout 10 phút cho mỗi request
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS, limit_per_host=CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_single_request(session, PDF_FILE_PATH, i) for i in range(NUM_REQUESTS)]
        
        start_total_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_total_time

    # Ghi kết quả từng request vào file JSONL với encoding UTF-8
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for result in results:
            if isinstance(result, Exception):
                # Trường hợp lỗi khi gather task (rất hiếm khi xảy ra nếu đã xử lý trong send_single_request)
                error_result = {
                    "request_idx": "unknown",
                    "status": None,
                    "response_time": time.perf_counter(),
                    "response_body": f"Gather Exception: {str(result)}",
                    "success": False
                }
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            else:
                # Ghi kết quả của từng request vào một dòng
                # ensure_ascii=False để giữ nguyên ký tự Unicode trong JSON
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Individual request results saved to {OUTPUT_FILE} (UTF-8)")

    # Phân tích kết quả tổng thể
    successful_requests = 0
    failed_requests = 0
    response_times = []
    errors = []

    for result in results:
        if isinstance(result, Exception):
            failed_requests += 1
            errors.append(str(result))
        else:
            # Dữ liệu từ send_single_request
            status = result['status']
            resp_time = result['response_time']
            success_flag = result['success']

            if success_flag:
                response_times.append(resp_time)
                successful_requests += 1
            else:
                failed_requests += 1
                errors.append(f"Request {result['request_idx']}: HTTP {status} or Exception: {result['response_body']}")


    if response_times:
        avg_time = sum(response_times) / len(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))] if len(response_times) > 1 else response_times[0]
        p99_time = sorted(response_times)[int(0.99 * len(response_times))] if len(response_times) > 1 else response_times[0]
        min_time = min(response_times)
        max_time = max(response_times)
    else:
        avg_time = p95_time = p99_time = min_time = max_time = 0

    print("-" * 50)
    print(f"Total Requests: {NUM_REQUESTS}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Total Time: {total_time:.2f}s")
    if successful_requests > 0:
        print(f"Requests per Second (RPS): {successful_requests / total_time:.2f}")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Min Response Time: {min_time:.2f}s")
        print(f"Max Response Time: {max_time:.2f}s")
        print(f"95th Percentile Time: {p95_time:.2f}s")
        print(f"99th Percentile Time: {p99_time:.2f}s")
    if errors:
        print("\nErrors:")
        for e in errors[:10]:  # Chỉ in 10 lỗi đầu tiên nếu có
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")


if __name__ == "__main__":
    # Đảm bảo file mẫu tồn tại
    if not Path(PDF_FILE_PATH).is_file():
        print(f"Error: File not found at {PDF_FILE_PATH}")
    else:
        asyncio.run(run_benchmark())
