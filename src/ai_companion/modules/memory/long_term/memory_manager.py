import logging
import uuid
from datetime import datetime
from typing import List, Optional
import os

from ai_companion.core.prompts import MEMORY_ANALYSIS_PROMPT
from ai_companion.modules.memory.long_term.vextor_store import get_vector_store
from settings import settings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field 

class MemoryAnalysis(BaseModel): 
    """Result of ananlyzing a message for memory-worthy content."""
    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory"
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")

class MemoryManager: 
    """Manager class for handling long-term memory operations. """

    def __init__(self) -> None:
        self.vector_store = get_vector_store() 
        self.logger = logging.getLogger(__name__) 
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed""" 
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt) 
    
    async def extract_and_store_memories(self, message: BaseMessage) -> None: 
        """Extract import information from a message and store in vector store. """
        if message.type != "human":
            return 
        
        content = message.content
        # Sanitize the message content to remove problematic characters if not in cloud
        if not os.getenv("RUNNING_IN_CLOUD", "0").lower() in ("1", "true", "yes"):
            content = ''.join(char for char in content if char.isprintable() or char.isspace()).strip()

        #analyze the message for importance and formatting 
        analysis = await self._analyze_memory(content)
        if analysis.is_important and analysis.formatted_memory: 
            #check if similar memory exists 
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory) 
            if similar: 
                #skip storage if we already have a similar memory 
                self.logger.info(f"Similar memory already exists:" '{analysis.formatted_memory}')
                return 
            #store new memory 
            self.logger.info(f"Storing new memory: '{analysis.formatted_memory}'")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )
    def get_relevant_memories(self, context: str) -> List[str]: 
        """Retreive relevant memories based on the current context. """
        memories = self.vector_store.search_memories(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories: 
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]
    
    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)
    

def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    return MemoryManager()