from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import CIIUAgentService
import asyncio
import json
app = FastAPI()


class CIIURequest(BaseModel):
    descripcion: str

@app.post("/consultar-ciiu")
async def consultar_ciiu(request: CIIURequest):
    try:
        service = CIIUAgentService()
        resultado = await service.consultar_ciiu(descripcion=request.descripcion)
        json_response = json.loads(str(resultado))
        return json_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
