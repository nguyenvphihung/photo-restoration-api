# 🚀 Hướng Dẫn Chạy & Test API Khôi Phục Ảnh (Phiên Bản V2 Tối Ưu)

Do cấu trúc Pipeline đã được tối ưu hóa sang dạng **Micro-Workers** (tải AI model 1 lần duy nhất vào RAM để chạy siêu tốc), luồng khởi động API giờ đây sẽ mở thêm 2 cửa sổ chạy ngầm thay vì chỉ 1 như trước.

Dưới đây là các bước thao tác tuần tự:

---

## BƯỚC 1: Khởi động Worker 1 (Bộ Khử Xước - ZeroScratches)
Mở một cửa sổ Terminal/PowerShell **MỚI** tại thư mục `RestorePhotos` và chạy lần lượt 2 lệnh sau:

```bash
conda activate rs-clean
python api/workers/zeroscratches_worker.py
```
*(Đợi vài giây cho đến khi thấy dòng chữ "Application startup complete" hiện ra. Worker này sẽ chạy ở cổng `8001`)*

---

## BƯỚC 2: Khởi động Worker 2 (Bộ Làm Nét Mặt + Phục Hồi Quần Áo/Nền)
Mở cửa sổ Terminal/PowerShell **THỨ 2** tại thư mục `RestorePhotos` và chạy:

```bash
conda activate gfpgan-clean
python api/workers/gfpgan_worker.py
```
*(Lưu ý: Mới bật nó sẽ tốn chút thời gian để load model Real-ESRGAN và GFPGAN vào RAM. Nó sẽ chạy ở cổng `8002`)*

---

## BƯỚC 3: Khởi động Worker 3 (Bộ Tô Màu - Colorization)
Mở cửa sổ Terminal/PowerShell **THỨ 3** tại thư mục `RestorePhotos` và chạy:

```bash
conda activate gfpgan-clean
python api/workers/colorization_worker.py
```
*(Worker tô màu này sẽ chạy ngầm ở cổng `8003`)*

---

## BƯỚC 4: Khởi động Worker 4 (Bộ Làm Nét Nâng Cao - Enhancer)
Mở cửa sổ Terminal/PowerShell **THỨ 4** tại thư mục `RestorePhotos` và chạy:

```bash
conda activate gfpgan-clean
python api/workers/enhancer_worker.py
```
*(Worker nâng nét này sẽ chạy ngầm ở cổng `8004`)*

---

## BƯỚC 5: Khởi động Cổng API Chính (FastAPI - Cho ReactJS kết nối)
Mở cửa sổ Terminal/PowerShell **THỨ 5** tại thư mục `RestorePhotos`. 
*Lưu ý: Môi trường ở Terminal 5 này cần được cài đặt sẵn các thư viện cơ bản bằng lệnh: `pip install -r api/requirements.txt` (nếu bạn chưa từng cài).*

Sau đó chạy lệnh quen thuộc:
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*(Nếu cần expose ra mạng ngoài, vẫn chạy lệnh `ngrok http 8000` ở một tab mới).*

---

## 🛠 DANH SÁCH 3 API CHO FRONTEND (REACTJS)

Dự án hiện tại cung cấp 3 Endpoint độc lập tùy theo nhu cầu của người dùng trên Web:

**1. Chỉ phục chế (Xóa xước + Nét mặt + Làm nét):**
```bash
curl -X POST "http://127.0.0.1:8000/api/restore" -F "file=@img1.jpg"
```

**2. Chỉ tô màu (Dùng cho ảnh Trắng/Đen không bị rách):**
```bash
curl -X POST "http://127.0.0.1:8000/api/colorize" -F "file=@img1.jpg"
```

**3. Phục chế toàn diện (Xóa xước + Nét mặt + Tô màu + Làm nét):**
```bash
curl -X POST "http://127.0.0.1:8000/api/restore-and-colorize" -F "file=@img1.jpg"
```

---

## 🛑 BÍ KÍP ĐI XÚY CHO DEVS (CÁCH HOÀN TÁC)
Nếu vì lý do gì đó mà thiết kế V2 này bị lỗi hoặc bạn muốn quay về code cũ ngày xưa cớ load mỗi lần một models, tôi đã tạo các file Backup rồi:

1. Xóa (hoặc đổi tên) file hiện tại: `api/restoration_service.py` -> `api/restoration_service_v2.py`
2. Đổi tên file Backup trở lại: `api/restoration_service.py.bak` -> `api/restoration_service.py`
3. Làm tương tự với `restore_photo.py` và `api/app.py` (bỏ đuôi `.bak`).
Thế là dự án sẽ về trạng thái nguyên thủy như chưa từng có cuộc chia ly!
