"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS, get_appointments
from react_agent.utils import load_chat_model
from react_agent.prompts import ROUTER_PROMPT, DETERMINE_APPOINTMENT_PROMPT, DETERMINE_RESCHEDULE_OR_CANCEL_PROMPT
# Define the function that calls the model

async def router(state: State, config: RunnableConfig) -> Dict[str, int]:
    """Route the conversation to the correct stage based on the user's message."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    response = await model.ainvoke(
        [{"role": "system", "content": ROUTER_PROMPT}, *state.messages], config
    )
    
    # cast the string number to an integer type
    stage = int(response.content)

    # Return the stage in a dictionary to update the state
    return {"router_stage": stage}

async def determine_reschedule_or_cancel(
    state: State, config: RunnableConfig
) -> Dict[str, int]:
    """Determine whether the user wants to reschedule or cancel an appointment."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    response = await model.ainvoke(
        [{"role": "system", "content": DETERMINE_RESCHEDULE_OR_CANCEL_PROMPT}, *state.messages], config
    )
    
    # cast the string number to an integer type
    reschedule_or_cancel = int(response.content)

    # Return the reschedule or cancel to update the state
    return {"reschedule_or_cancel_decision": reschedule_or_cancel}

async def reschedule_message_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Node to output a message about rescheduling."""
    message = AIMessage(
        content="Let me escalate this with a real person."
    )
    return {"messages": [message]}


async def get_appointments_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[Any]]:
    """Fetch appointments for the customer and store them in the state.
    
    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the run.
        
    Returns:
        dict: A dictionary containing the appointments to update the state.
    """
    # Call the get_appointments tool
    appointments = get_appointments(config=config)
    
    # Return the appointments to update the state
    return {"appointments": appointments}


async def suggest_reschedule_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Node to suggest rescheduling instead of cancellation.
    
    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the run.
        
    Returns:
        dict: A dictionary containing the message to add to the state.
    """
    # Create a response message suggesting rescheduling
    response_message = AIMessage(
        content="I would love to cancel the appointment but would you like to reschedule instead?"
    )
    
    # Return the message to be added to the state
    return {"messages": [response_message]}

async def determine_appointment_to_cancel(
    state: State, config: RunnableConfig
) -> Dict[str, int]:
    """Determine which appointment to cancel based on the user's message."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    
    # Format the appointments into a string representation
    appointments_str = str(state.appointments)
    
    # Format the messages into a string representation
    messages_str = "\n".join([
        f"{msg.type}: \"{msg.content}\"" 
        for msg in state.messages
    ])
    
    # Format the prompt template with the appointments and messages
    formatted_prompt = DETERMINE_APPOINTMENT_PROMPT.format(
        appointments=appointments_str,
        messages=messages_str
    )
    
    # Create a proper message list with both system and a dummy user message
    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": "Determine which appointment to cancel."}
    ]
    
    # Invoke the model with proper message structure
    response = await model.ainvoke(messages, config)
    
    # Cast the string number to an integer type
    appointment_id = int(response.content)

    # Return the appointment id to update the state
    return {"appointment_id": appointment_id}
    
    
async def confirmation_message_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Node to output a confirmation message after appointment cancellation."""
    # Create a confirmation message
    confirmation_message = AIMessage(
        content="I will cancel the appointment for you! Thank you!"
    )
    
    # Return the message to be added to the state
    return {"messages": [confirmation_message]}


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will use
builder.add_node("router", router)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("get_appointments_node", get_appointments_node)
builder.add_node("suggest_reschedule_node", suggest_reschedule_node)
builder.add_node("determine_reschedule_or_cancel", determine_reschedule_or_cancel)
builder.add_node("determine_appointment_to_cancel", determine_appointment_to_cancel)
builder.add_node("confirmation_message_node", confirmation_message_node)
builder.add_node("reschedule_message_node", reschedule_message_node)
# Set the entrypoint as `router`
builder.add_edge("__start__", "router")

# Add a conditional edge from router to determine the next node based on the stage
def route_from_router(state: State) -> str:
    """Determine the next node based on the router's stage value."""
    # Get the stage from the state
    stage = state.router_stage
    
    # Route based on the stage value
    if stage == 1:
        print("Stage 1")
        return "get_appointments_node"  # Now routes to get_appointments_node
    elif stage == 2:
        # You could add other paths here
        print("Stage 2")
        return "call_model" 
    else:
        print("Stage 3")
        return "determine_reschedule_or_cancel"  

builder.add_conditional_edges("router", route_from_router)

# Add an edge from get_appointments_node to suggest_reschedule_node
builder.add_edge("get_appointments_node", "suggest_reschedule_node")

builder.add_edge("determine_appointment_to_cancel", "confirmation_message_node")

builder.add_edge("confirmation_message_node", "__end__")


# Add an edge from suggest_reschedule_node to end
builder.add_edge("suggest_reschedule_node", "__end__")

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Define the function that routes after reschedule_or_cancel BEFORE graph compilation
def route_after_reschedule_or_cancel(state: State) -> str:
    """Route based on whether user wants to reschedule or cancel."""
    if state.reschedule_or_cancel_decision == 2:  # 2 = Cancel
        return "determine_appointment_to_cancel"
    else:  # 1 = Reschedule
        # For now, just output a message that we don't handle rescheduling yet
        return "reschedule_message_node"


# Add an edge from reschedule_message to end
builder.add_edge("reschedule_message_node", "__end__")

# Now add the conditional edge before compiling
builder.add_conditional_edges(
    "determine_reschedule_or_cancel", 
    route_after_reschedule_or_cancel
)

# Now compile the graph (after all nodes and edges are defined)
graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "ReAct Agent"
