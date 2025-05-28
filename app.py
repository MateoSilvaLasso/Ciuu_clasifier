from agent import CIIUAgentService
import asyncio

async def consult():
    service = CIIUAgentService() 
    resultado = await service.consultar_ciiu(descripcion="Soy profesor de un colegio en un pueblo")
    print(resultado)

if __name__ == "__main__":
    asyncio.run(consult())