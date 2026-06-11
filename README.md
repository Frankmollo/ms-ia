# Microservicio de Inteligencia Artificial (PyTorch)

Este microservicio expone una API con FastAPI y contiene los scripts para entrenar una red neuronal que clasifica imágenes (Ej: Medidor Normal vs Manipulado, Documento Válido vs Inválido).

## Estructura Esperada del Dataset
Como aún no tenemos imágenes, debes agregarlas manualmente en carpetas. El script `train.py` usa `ImageFolder` de PyTorch, lo que significa que el nombre de la carpeta define la etiqueta (clase) de la imagen.

Crea esta estructura cuando tengas las fotos:

```
ms-ia/
│
├── dataset/
│   ├── normal/
│   │   ├── foto_medidor_1.jpg
│   │   └── foto_medidor_2.jpg
│   │
│   └── manipulado/
│       ├── foto_hurto_1.jpg
│       └── foto_hurto_2.jpg
```

## Instrucciones de Uso

1. **Instalar Dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar el Modelo:**
   Una vez que tengas fotos en las carpetas `dataset/`, corre el script:
   ```bash
   python train.py
   ```
   Esto generará un archivo `models/modelo_entrenado.pth`.

3. **Arrancar la API:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000
   ```
   La API estará lista para recibir imágenes de la App Móvil en `http://localhost:5000/predict`.
