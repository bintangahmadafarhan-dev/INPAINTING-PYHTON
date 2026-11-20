# 🖼️ Deteksi Teks Merah Otomatis + Dual Panel Output (Versi GPU - CuPy)

Skrip Python ini digunakan untuk mendeteksi teks berwarna **merah terang** dalam gambar menggunakan **CuPy (GPU)**, lalu:

- Menampilkan gambar grayscale dengan teks putih
- Menampilkan mask hitam-putih (teks putih, latar hitam)
- Output tampil **berdampingan seperti contoh hasil akademik (dual panel)**

---

## 📌 Fitur

- ✅ Input hanya 1 gambar (tidak perlu mask terpisah)
- ✅ Deteksi teks merah otomatis berdasarkan RGB
- ✅ Proses cepat menggunakan GPU (NVIDIA + CUDA)
- ✅ Tampilkan dua panel: `Damaged Image` + `Mask`
- ✅ Simpan mask hasil ke file `.png`

---

## 🖥️ Contoh Tampilan Output

- **Kiri**: Gambar asli → grayscale → teks merah jadi putih
- **Kanan**: Hasil deteksi teks sebagai mask

---

## 📦 Ketergantungan

Install dependensi dengan pip:

```bash
pip install cupy-cuda12x numpy imageio matplotlib
