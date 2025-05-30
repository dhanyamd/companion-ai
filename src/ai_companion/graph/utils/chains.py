from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.graph.utils.helpers import AsteriskRemovalParser
from ai_companion.graph.utils.model_utils import get_chat_model
from settings import settings


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )


def get_router_chain():
    model = get_chat_model(
        temperature=0.3,
        model_name=settings.SMALL_TEXT_MODEL_NAME  # Use small model for routing
    ).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    ) 

    return prompt | model


def get_character_response_chain(summary: str = "", use_small_model: bool = False):
    """
    Get a character response chain.
    
    Args:
        summary: Optional conversation summary
        use_small_model: Whether to use the small model (for audio responses)
    """
    model = get_chat_model(
        temperature=0.7,
        model_name=settings.SMALL_TEXT_MODEL_NAME if use_small_model else settings.TEXT_MODEL_NAME
    )
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Ava and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()