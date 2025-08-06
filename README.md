## ✅ Requirements

- Python 3.12 atau lebih baru
- Pip
- Virtualenv

---

## 🚀 Cara Menjalankan API

### 1. Clone Repo

```bash
git clone https://github.com/gerinsp/karate-classification-api.git
cd karate-classification-api
```

### 2. Buat Virtual Environment 

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan API

```bash
uvicorn main_holistic:app --host 0.0.0.0 --port 8000
```

API akan tersedia di [http://127.0.0.1:8000](http://127.0.0.1:8000)
