from llama_index.readers.file import PandasCSVReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from dotenv import load_dotenv
import os

class CIIUAgentService:
    def __init__(self):
        # Cargar variables de entorno
        load_dotenv()
        openai_token = os.getenv("OPENAI_API_KEY")

        # Configurar LLM y embeddings
        self.llm = OpenAI(model="gpt-4.1", token=openai_token)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small", token=openai_token)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Intentar cargar el índice desde almacenamiento
        try:
            storage_context = StorageContext.from_defaults(persist_dir="./storage/ciuu")
            self.ciuu_index = load_index_from_storage(storage_context)
        except Exception:
            # Si no existe, se crea desde los datos
            parser = PandasCSVReader(pandas_config={"encoding": "latin1"})
            file_extractor = {".csv": parser}

            ciuu_docs = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
            self.ciuu_index = VectorStoreIndex.from_documents(ciuu_docs)
            self.ciuu_index.storage_context.persist(persist_dir="./storage/ciuu")

        # Crear motor de consulta
        ciuu_engine = self.ciuu_index.as_query_engine(similarity_top_k=3)

        # Herramientas del agente
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=ciuu_engine,
                name="ciuu",
                description="use this tool to get information from the ciuu database",
            )
        ]

        # Definir agente
        self.agent = FunctionAgent(
            name="ciuu",
            description="use this agent to get information from the ciuu database",
            tools=query_engine_tools,
            llm=self.llm,
            system_prompt="""
                    Eres un asistente experto en clasificación económica que ayuda a las personas a identificar los códigos CIIU (Clasificación Industrial Internacional Uniforme) que mejor se ajustan a la descripción de su actividad económica.

                    Cuando el usuario proporcione una descripción de su trabajo, analiza cuidadosamente el texto para entender la actividad principal. Luego, busca y selecciona los códigos CIIU más relevantes según su similitud semántica y conceptual.

                    Tu respuesta debe ser un objeto JSON válido con una lista llamada "resultados", en la que cada elemento contiene dos claves: "codigo" y "descripcion".

                    Devuelve siempre entre 1 y 3 coincidencias relevantes, ordenadas por similitud. No incluyas texto adicional, explicaciones ni comentarios fuera del JSON. Devuelve solo el JSON.

                    Formato requerido:
                    {
                    "resultados": [
                        {
                        "codigo": "0112",
                        "descripcion": "Cultivo de arroz"
                        },
                        {
                        "codigo": "0113",
                        "descripcion": "Cultivo de hortalizas, raíces y tubérculos"
                        },
                        {
                        "codigo": "0111",
                        "descripcion": "Cultivo de cereales (excepto arroz), legumbres y semillas oleaginosas"
                        }
                    ]
                    }

                    Si la descripción es ambigua o insuficiente, devuelve exactamente esto:
                    {
                    "resultados": []
                    }
                    """        
        )

        

    async def consultar_ciiu(self, descripcion: str) -> str:
        """
        Ejecuta una consulta al agente CIIU con la descripción proporcionada.
        """
        handler = self.agent.run(descripcion)
        response = await handler
        return str(response)
