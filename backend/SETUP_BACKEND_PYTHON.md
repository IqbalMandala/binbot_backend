# BinBot Backend Python Environment Setup (Python 3.10/3.11)

## 1. Download & Install Python 3.10.x atau 3.11.x
- Download dari https://www.python.org/downloads/
- Install, centang "Add to PATH"

## 2. Buat Virtual Environment Baru
Buka terminal di folder backend:

```
python3.10 -m venv .venv
```
atau
```
python3.11 -m venv .venv
```

## 3. Aktifkan Virtual Environment
```
.venv\Scripts\activate
```

## 4. Install NumPy & OpenCV yang Kompatibel
```
pip install numpy==1.26.4 opencv-python==4.9.0.80
```

## 5. Install Semua Dependensi Lain
```
pip install -r requirements.txt
```

## 6. Jalankan Backend
```
uvicorn main:app --reload
```

---

**Catatan:**
- Jangan gunakan Python 3.12+ untuk backend ini.
- Jika ada error, pastikan virtualenv aktif dan versi Python sudah benar.
- Untuk deployment/production, gunakan perintah tanpa --reload.
