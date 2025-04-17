from agent import CIIUAgentService
import asyncio

async def consult():
    service = CIIUAgentService() 
    resultado = await service.consultar_ciiu(descripcion="Cultivo de arroz")
    print(resultado)

if __name__ == "__main__":
    asyncio.run(consult())