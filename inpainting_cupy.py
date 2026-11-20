import cupy as cp
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os
import sys

def load_image_to_gpu():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not path:
        print("❌ Tidak ada gambar dipilih.")
        sys.exit()

    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype(np.uint8)
    return cp.array(img), os.path.basename(path)

def extract_teks_mask_gpu(img_gpu):
    red = img_gpu[:, :, 0].astype(cp.float32)
    green = img_gpu[:, :, 1].astype(cp.float32)
    blue = img_gpu[:, :, 2].astype(cp.float32)

    mask = ((red > 150) & (green < 100) & (blue < 100)).astype(cp.float32)
    return mask

if __name__ == "__main__":
    img_gpu, nama_file = load_image_to_gpu()

    # Grayscale
    grayscale_gpu = cp.mean(img_gpu.astype(cp.float32), axis=2) / 255.0
    mask_gpu = extract_teks_mask_gpu(img_gpu)

    damaged_gpu = grayscale_gpu.copy()
    damaged_gpu[mask_gpu == 1] = 1.0  # teks jadi putih

    # Transfer ke CPU
    damaged = cp.asnumpy(damaged_gpu)
    mask_output = (mask_gpu * 255).astype(cp.uint8).get()
    grayscale_img = cp.asnumpy(img_gpu)

    # Tampilkan
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(damaged, cmap='gray')
    plt.title("Damaged Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_output, cmap='gray')
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()

    basename = os.path.splitext(nama_file)[0]
    outfile = f"{basename}_teks_mask_dualpanel_gpu.png"
    imageio.imwrite(outfile, mask_output)
    print(f"✅ Disimpan ke: {outfile}")
    plt.show()
