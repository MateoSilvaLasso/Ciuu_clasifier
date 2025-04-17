from llama_index.readers.file import PandasCSVReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from dotenv import load_dotenv
import os

class CIIUAgentService:
    def __init__(self):
        # Cargar variables de entorno
        load_dotenv()
        gemini_token = os.getenv("GOOGLE_API_KEY")

        # Configurar LLM y embeddings
        self.llm = GoogleGenAI(model="gemini-2.5-pro-exp-03-25", token=gemini_token)
        self.embed_model = GoogleGenAIEmbedding(model="gemini-embedding-exp-03-07", token=gemini_token)

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
        self.agent = ReActAgent(
            name="ciuu",
            description="use this agent to get information from the ciuu database",
            tools=query_engine_tools,
            llm=self.llm,
            system_prompt="""
                Eres un asistente experto en clasificación económica que ayuda a las personas a identificar el código CIIU (Clasificación Industrial Internacional Uniforme) que mejor se ajusta a la descripción de su actividad económica. Tu objetivo es analizar cuidadosamente la descripción que proporciona el usuario y encontrar el código CIIU más relevante y preciso.

                Tienes acceso a una base de datos con códigos CIIU, sus nombres y descripciones. Cuando recibas una descripción de trabajo, debes:

                1. Comprender el propósito principal de la actividad descrita.
                2. Buscar el código CIIU que más se asemeje en términos de sector, producto, o servicio ofrecido.
                3. Priorizar coincidencias conceptuales y semánticas, no solo palabras clave.
                4. En caso de dudas, elige las 3 opciones más específicas y directas que representen la actividad.
                5. Devuelve el resultado en el siguiente formato en json:

                Código CIIU: [CÓDIGO]  
                Nombre: [NOMBRE DE LA ACTIVIDAD]  

                Si el usuario no proporciona suficiente información, responde solicitando una descripción más detallada de la actividad.

                Responde siempre en español y con un lenguaje claro y formal.

                No te olvides nunca de responder en el formato dado y el código tiene que ser el número asociado al nombre de la actividad, no el nombre de la actividad solo.

                Si no encuentras el código CIIU, responde con "No se encontró el código CIIU para la actividad proporcionada".

                Responde con un json en el siguiente formato.
            """
        )

        # Crear contexto del agente
        self.ctx = Context(self.agent)

    async def consultar_ciiu(self, descripcion: str) -> str:
        """
        Ejecuta una consulta al agente CIIU con la descripción proporcionada.
        """
        handler = self.agent.run(descripcion, ctx=self.ctx)
        response = await handler
        return str(response)
