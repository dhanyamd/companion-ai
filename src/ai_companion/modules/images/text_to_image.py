import base64 
import logging
import os
from typing import Optional

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from settings import settings, clean_api_key
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from together import Together

class ScenarioPrompt(BaseModel): 
    """Class for the scenario response""" 

    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")

class EnhancedPrompt(BaseModel):
    """Class for the enhanced prompt response"""
    content: str = Field(..., description="The enhanced prompt with additional details")

class TextToImage:
    """A class to handle text-to-image generation using Together AI."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TOGETHER_API_KEY"]

    def __init__(self):
        """Initialize the TextToImage class and validate environment variables."""
        self._validate_env_vars()
        self._together_client: Optional[Together] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    @property
    def together_client(self) -> Together: 
        """Get or create Together client instance using singleton pattern.""" 
        if self._together_client is None: 
            # Clean the API key and ensure it's properly formatted
            cleaned_api_key = clean_api_key(settings.TOGETHER_API_KEY)
            if not cleaned_api_key:
                raise ValueError("Invalid Together API key")
            self._together_client = Together(api_key=cleaned_api_key)
        return self._together_client
    
    async def generate_image(self, prompt: str, output_path: str = "") -> bytes: 
        """Generate an image from a prompt using Together AI""" 
        if not prompt.strip(): 
            raise ValueError("Prompt cannot be empty")
        
        try: 
            self.logger.info(f"Generating image for prompt: '{prompt}'") 
            
            # Ensure client is initialized
            client = self.together_client
            
            # Generate image using Together API
            response = await client.images.generate(
                prompt=prompt,
                model=settings.TTI_MODEL_NAME,
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json"
            )
            
            if not response or not response.data or not response.data[0].b64_json:
                raise TextToImageError("No image data received from Together API")
                
            image_data = base64.b64decode(response.data[0].b64_json)
            
            if output_path: 
                os.makedirs(os.path.dirname(output_path), exist_ok=True) 
                with open(output_path, "wb") as f: 
                    f.write(image_data) 
                self.logger.info(f"Image saved to {output_path}")

            return image_data 
        except Exception as e: 
            self.logger.error(f"Failed to generate image: {str(e)}")
            raise TextToImageError(f"Failed to generate image: {str(e)}") from e

    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt: 
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history."""
       
        try:
            formatted_history = "\n".join([f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]])

            self.logger.info("Creating scenario from chat history")
            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.4,
                max_retries=2
            )
            structured_llm = llm.with_structured_output(ScenarioPrompt)
            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT
                ) | structured_llm 
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario

        except Exception as e:
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        try:
            self.logger.info(f"Enhancing prompt: '{prompt}'")

            llm = ChatGroq(
                model=settings.SMALL_TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt}).content
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

            return enhanced_prompt

        except Exception as e:
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e