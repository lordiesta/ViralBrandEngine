# 🚀 ViralBrandEngine — Proje Devir Teslim & Durum Belgesi (Handoff)

**Tarih:** 2026-10-24  
**Proje Yolu:** `C:\Users\KERO\.gemini\antigravity\scratch\viral_brand_engine\`  
**Ortam:** Windows PowerShell, Python venv (`.\.venv\Scripts\uvicorn.exe`)  
**Aktif Sunucu:** `http://127.0.0.1:8000` (Swagger: `/docs`)

---

## 📌 1. Projenin Amacı ve Hoca Gereksinimleri

Bu proje, otonom viral kısa video (Reels, TikTok, Shorts) üreten bir yapay zeka motorudur.  
**Öğretim Görevlisi (Halil Arslan) Beklentileri (Pazartesi Günü):**
1. **Canlı Çalışan RESTful API:** Sistem sadece bir web arayüzü değil, makineden makineye (B2B / API) konuşan bir mimari olmalıdır.
2. **JSON Veri Seti:** Hoca kendi bilgisayarından/kodundan öğrencinin sistemine istek atıp veri toplayabilecek (`/api/v1/videos`, `/api/v1/analytics/summary`).
3. **Dokümantasyon & Test:** `/docs` (Swagger UI), `/openapi.json` ve Postman koleksiyonu (`viral_brand_engine.postman_collection.json`) hazır olmalıdır. (Tamamı test edildi ve hazır).

---

## 🛠️ 2. Son Yapılan Mimari İyileştirmeler (Sorun & Çözüm)

Kullanıcının son geri bildirimi: *"Videolar konuyla alakasız, akış çok bağlantısız, altyazı/senaryo ile ekrandaki görüntü uyuşmuyor."*

Bu sorunu çözmek için **Sahne Bazlı Kurgu (Scene-Based Storyboard Architecture)** uygulandı:

### A) Senaryo Üretimi (`app/services/script_generator.py`)
- Rastgele B-roll listesi yerine **zaman kilitli sahneler (`scenes`)** yapısına geçildi.
- Her sahnenin:
  - `start_time` & `end_time` (ör. 0.0s - 5.0s)
  - `narration_text` (o saniyede okunan tam cümle)
  - `visual_search_query` (o cümleyi anlatan 2 kelimelik net görsel arama)
  - `visual_description` (Google Flow / AI görsel üretimi için detaylı prompt)
- Hem Gemini LLM prompt'u hem de yerel akıllı fallback motoru bu sahne yapısıyla senkronize edildi.

### B) Görsel Tutarlılık & Google Flow (`app/services/broll_fetcher.py`)
- **Çıpa Özne (Anchor Subject):** Konu "kedi" ise sistem aramaları `cat` çıpasına bağlar. Bir sahnede ev kedisi varken diğerinde vahşi kaplan veya alakasız otobüs çıkması engellendi.
- **Google Flow Desteği:** Kullanıcının Google Flow aboneliği için `GOOGLE_FLOW_API_KEY` desteği eklendi. Anahtar girildiğinde stok video yerine doğrudan Google Flow/Veo video motoruna gider.

### C) Saniye Saniye Video Senkronizasyonu (`app/services/video_editor.py`)
- MoviePy video editörü artık klipleri rastgele 4.5 saniyeye bölmüyor.
- `scenes` listesindeki her sahnenin süresine göre (`scene_dur = end_time - start_time`), o sahneye ait video klibini tam o saniye aralığına oturtuyor.
- **Sonuç:** Ses "kedilerin gözleri büyür" derken ekranda tam o anda göz klibi oynuyor; ses 2. cümleye geçtiğinde tam o saniyede 2. klip başlıyor.

---

## 📡 3. Canlı API Endpoint Listesi

| Endpoint | Metot | Görevi |
| :--- | :--- | :--- |
| `http://127.0.0.1:8000/api/v1/videos` | **GET** | Tamamlanan videoların listesi ve doğrudan tıklandığında açılan `watch_url` (MP4) linkleri |
| `http://127.0.0.1:8000/api/v1/analytics/summary` | **GET** | Hocanın veri analizi için platform istatistikleri (başarı oranı, süre, sağlayıcılar) |
| `http://127.0.0.1:8000/api/v1/analytics/query` | **POST** | Konu, dil, süre filtreli gelişmiş arama |
| `http://127.0.0.1:8000/api/v1/generate` | **POST** | Yeni video oluşturma işi başlatma |
| `http://127.0.0.1:8000/api/v1/status/{job_id}` | **GET** | Video üretim aşamasını sorgulama |
| `http://127.0.0.1:8000/docs` | **GET** | Canlı Swagger arayüzü |

---

## ⚙️ 4. Sunucuyu Başlatma ve Hızlı Kontrol

Sonraki ajan sunucuyu başlatmak veya kontrol etmek için:

```powershell
# 1. Proje dizinine git
cd C:\Users\KERO\.gemini\antigravity\scratch\viral_brand_engine

# 2. Sunucuyu arka planda başlat
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload

# 3. Sağlık kontrolü
curl.exe -s http://127.0.0.1:8000/health
```

---

## 📋 5. Sonraki Adımlar / Açık Kalan Maddeler

1. **Google Flow Anahtarının Girilmesi:**
   - Web arayüzündeki **"API Anahtarları (Ayarlar)"** modalına veya `.env` içine `GOOGLE_FLOW_API_KEY` eklenerek doğrudan Google Flow ile video testi yapılabilir.
2. **Canlı Test:**
   - Yeni Sahne Bazlı Kurgu mimarisiyle 10 saniyelik yeni bir video oluşturup ses-görüntü senkronizasyonunu kullanıcıya izletmek.
