from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from model_service import SignatureDetector

app = FastAPI(
    title="Microservicio de Inteligencia Artificial - Deep Learning",
    description="Clasificador de Imágenes usando PyTorch para detectar firmas en documentos",
    version="1.0.0"
)

# Inicializar la Red Neuronal (Cargará en memoria al arrancar)
detector = SignatureDetector()

@app.get("/")
def home():
    return {"status": "ok", "message": "API de Deep Learning para Firmas en línea"}

@app.post("/predict/signature")
async def predict_signature(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer los bytes de la imagen
        image_bytes = await file.read()
        
        # Realizar la inferencia con PyTorch
        result = detector.predict(image_bytes)
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "model_engine": "PyTorch ResNet-18 (Transfer Learning)"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
