# Organizador IA de Imágenes

Este proyecto utiliza CLIP (Contrastive Language–Image Pretraining) de OpenAI para organizar y clasificar imágenes automáticamente basándose en descripciones semánticas.

## 🧠 Funcionalidades

- Clasificación automática de imágenes por contexto y semántica.
- Memoria de descripciones para evitar duplicados o ambigüedad.
- Entrenamiento continuo usando referencias nuevas.
- Guarda las asociaciones imagen → texto para futuras consultas.

## 📁 Archivos principales

| Archivo                           | Descripción |
|----------------------------------|-------------|
| `clasificador_clip_con_memoria.py`     | Clasificador principal con memoria persistente. |
| `entrenamiento_continuo_clip.py`      | Entrena y ajusta las asociaciones con nuevas imágenes. |
| `guardar_referencias_clip.py`        | Guarda descripciones y vectores CLIP asociados. |

## ⚙️ Requisitos

Asegúrate de tener Python 3.8+ y ejecuta:

```bash
pip install -r requirements.txt
```

## ▶️ Uso básico

1. Coloca tus imágenes en una carpeta.
2. Ejecuta `clasificador_clip_con_memoria.py`.
3. Se crearán descripciones y se organizarán las imágenes automáticamente.

## 💡 Ejemplo de uso

```bash
python clasificador_clip_con_memoria.py --carpeta ./imagenes_a_ordenar
```

## 👨‍💻 Autor

Desarrollado por [@Harrius095](https://github.com/Harrius095)
