import os
import torch
import clip
from PIL import Image

# Ruta base con imágenes de ejemplo por personaje
BASE_DIR = r"C:\Users\enriq\Desktop\Nueva carpeta"
DESTINO = os.path.join(BASE_DIR, "referencias_guardadas")
os.makedirs(DESTINO, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("🧠 Guardando vectores de referencia por personaje...\n")

for carpeta in os.listdir(BASE_DIR):
    carpeta_path = os.path.join(BASE_DIR, carpeta)
    if os.path.isdir(carpeta_path) and carpeta.lower() not in ["nuevas", "revisar_manual", "referencias_guardadas"]:
        vectores = []
        for archivo in os.listdir(carpeta_path):
            ruta_img = os.path.join(carpeta_path, archivo)
            try:
                imagen = preprocess(Image.open(ruta_img)).unsqueeze(0).to(device)
                with torch.no_grad():
                    vector = model.encode_image(imagen)
                    vector /= vector.norm(dim=-1, keepdim=True)
                    vectores.append(vector)
            except:
                continue
        if vectores:
            media_vector = torch.mean(torch.stack(vectores), dim=0)
            torch.save(media_vector, os.path.join(DESTINO, f"{carpeta}.pt"))
            print(f"✅ {carpeta} guardado.")
print("\n🎉 Listo. Ahora puedes usar estos vectores sin necesidad de las imágenes.")
