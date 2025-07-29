import os
import shutil
import torch
import clip
from PIL import Image

# Rutas
BASE_DIR = r"C:\Users\enriq\Desktop\Nueva carpeta"
NUEVAS_DIR = os.path.join(BASE_DIR, "nuevas")
REVISAR_DIR = os.path.join(BASE_DIR, "revisar_manual")
REFERENCIAS_DIR = os.path.join(BASE_DIR, "referencias_guardadas")

os.makedirs(REVISAR_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Cargar vectores de referencia desde archivos .pt
referencias = {}
print("🧠 Cargando vectores de referencia...")

for archivo in os.listdir(REFERENCIAS_DIR):
    if archivo.endswith(".pt"):
        personaje = archivo.replace(".pt", "")
        vector = torch.load(os.path.join(REFERENCIAS_DIR, archivo)).to(device)
        referencias[personaje] = vector
        print(f"✅ {personaje} cargado.")

print("\n📂 Clasificando imágenes nuevas...")

for archivo in os.listdir(NUEVAS_DIR):
    ruta_img = os.path.join(NUEVAS_DIR, archivo)
    try:
        imagen = preprocess(Image.open(ruta_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            vector_img = model.encode_image(imagen)
            vector_img /= vector_img.norm(dim=-1, keepdim=True)

        mejores = []
        for nombre, vector_ref in referencias.items():
            similitud = torch.cosine_similarity(vector_img, vector_ref, dim=-1).item()
            mejores.append((nombre, similitud))

        mejores.sort(key=lambda x: x[1], reverse=True)
        mejor_nombre, mejor_similitud = mejores[0]

        if mejor_similitud >= 0.75:
            destino = os.path.join(BASE_DIR, mejor_nombre, archivo)
            shutil.move(ruta_img, destino)
            print(f"✅ {archivo} → {mejor_nombre} (similitud: {mejor_similitud:.2f})")
        else:
            destino = os.path.join(REVISAR_DIR, archivo)
            shutil.move(ruta_img, destino)
            print(f"🟨 {archivo} → revisar_manual (similitud baja: {mejor_similitud:.2f})")

    except Exception as e:
        print(f"⚠️ Error con {archivo}: {e}")
