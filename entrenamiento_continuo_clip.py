import os
import torch
import clip
from PIL import Image

# Ruta base
BASE_DIR = r"C:\Users\enriq\Desktop\Nueva carpeta"
REFERENCIAS_DIR = os.path.join(BASE_DIR, "referencias_guardadas")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("🔁 Aprendiendo de nuevas imágenes aprobadas...\n")

# Revisar todas las carpetas excepto especiales
for carpeta in os.listdir(BASE_DIR):
    carpeta_path = os.path.join(BASE_DIR, carpeta)
    if os.path.isdir(carpeta_path) and carpeta.lower() not in ["nuevas", "revisar_manual", "referencias_guardadas"]:
        nuevas_imgs = []
        for archivo in os.listdir(carpeta_path):
            ruta_img = os.path.join(carpeta_path, archivo)
            try:
                imagen = preprocess(Image.open(ruta_img)).unsqueeze(0).to(device)
                with torch.no_grad():
                    vector = model.encode_image(imagen)
                    vector /= vector.norm(dim=-1, keepdim=True)
                    nuevas_imgs.append(vector)
            except:
                continue

        if nuevas_imgs:
            nuevo_vector = torch.mean(torch.stack(nuevas_imgs), dim=0)
            ref_path = os.path.join(REFERENCIAS_DIR, f"{carpeta}.pt")
            if os.path.exists(ref_path):
                vector_antiguo = torch.load(ref_path).to(device)
                # Mezclar el vector anterior con el nuevo para actualizar
                vector_final = (vector_antiguo + nuevo_vector) / 2
                torch.save(vector_final, ref_path)
                print(f"🔄 {carpeta}: referencia actualizada.")
            else:
                torch.save(nuevo_vector, ref_path)
                print(f"🆕 {carpeta}: referencia creada.")
print("\n✅ Entrenamiento continuo completado.")
