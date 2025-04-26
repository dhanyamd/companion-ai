import os 
from uuid import uuid4 
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from graph.state import AICompanionState
from graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from settings import settings

async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    return {"workflow": response.response_type}

def context_injection_node(state: AICompanionState): 
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity",""):
        apply_activity = True
    else:
        apply_activity = False 
    return {"apply_activity": apply_activity, "cuurent_activity": schedule_context}

async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "") 
    
    chain = get_character_response_chain(state.get("summary", ""))
    response = await chain.ainvoke({
        "messages": state["messages"],
        "current_activity": current_activity,
        "memory_context": memory_context
    }, config)
    return {"messages": AIMessage(content=response)}

async def image_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(content=f"<image attached by Ava generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}

async def audio_node(state: AICompanionState, config: RunnableConfig): 
      current_activity = ScheduleContextGenerator.get_current_activity() 
      memory_context = state.get("memory_context","")

      chain = get_character_response_chain(state.get("summary", ""))
      text_to_speech_module = get_text_to_speech_module()

      response = await chain.ainvoke(
          {
              "messages": state["messages"], 
              "current_activity": current_activity,
              "memory_context": memory_context,
          },
          config
      )
      output_audio = await text_to_speech_module.synthesize(response)
      return {"messages": response, "audio_buffer": output_audio}
