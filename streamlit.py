from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from ethdocker_expert import ethdocker_expert, EthDockerDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    try:
        if part.part_kind == 'system-prompt':
            with st.chat_message("system"):
                st.markdown(f"**System**: {part.content}")
        elif part.part_kind == 'user-prompt':
            with st.chat_message("user"):
                st.markdown(part.content)
        elif part.part_kind == 'text':
            with st.chat_message("assistant"):
                st.markdown(part.content)
        elif part.part_kind == 'tool-call':
            with st.chat_message("tool"):
                # Use tool_name instead of name
                st.markdown(f"🔧 **Tool Call**: {part.tool_name}")
                with st.expander("View Tool Call Details"):
                    # Handle arguments safely
                    if hasattr(part, 'arguments'):
                        try:
                            if isinstance(part.arguments, str):
                                args = json.loads(part.arguments)
                            else:
                                args = part.arguments
                            st.json(args)
                        except json.JSONDecodeError:
                            st.code(part.arguments)
                    elif hasattr(part, 'parameters'):
                        st.json(part.parameters)
        elif part.part_kind == 'tool-return':
            with st.chat_message("tool"):
                st.markdown("🔄 **Tool Result**")
                with st.expander("View Tool Result"):
                    try:
                        # Try to parse content as JSON if it's a string
                        if isinstance(part.content, str):
                            result = json.loads(part.content)
                            st.json(result)
                        else:
                            st.markdown(part.content)
                    except (json.JSONDecodeError, AttributeError):
                        # If content is not available or not JSON, try return_value
                        if hasattr(part, 'return_value'):
                            st.markdown(part.return_value)
                        else:
                            st.markdown(str(part))
    except Exception as e:
        logger.error(f"Error displaying message part: {str(e)}")
        st.error(f"Error displaying message part: {str(e)}")

async def run_agent_with_streaming(user_input: str):
    """
    Run the ETHDocker expert agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = EthDockerDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with ethdocker_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                           if not (hasattr(msg, 'parts') and 
                                 any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.set_page_config(
        page_title="ETHDocker Expert",
        page_icon="🐳",
        layout="wide"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

    st.title("🐳 ETHDocker Expert")
    
    # Add sidebar with information
    with st.sidebar:
        st.header("About ETHDocker")
        st.markdown("""
        ETHDocker is a comprehensive solution for running Ethereum staking full nodes. 
        It simplifies the process while giving you full control over client choices.
        
        **Key Features:**
        - Multi-client support
        - Linux/macOS compatible
        - Supports various CPU architectures
        - Grafana monitoring
        - Security-focused design
        
        **Hardware Requirements:**
        - 32 GiB RAM (16 GiB min)
        - 4 CPU cores
        - 4TB SSD (TLC & DRAM)
        """)
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.error_count = 0
            st.rerun()

    # Main chat interface
    st.markdown("""
    Welcome! I'm your ETHDocker expert assistant. I can help you with:
    - Node setup and configuration
    - Client selection and management
    - Validator key management
    - Staking workflow guidance
    - Troubleshooting and best practices
    - Security considerations
    """)

    # Create a container for the chat history
    chat_container = st.container()

    # Create a container for the input
    input_container = st.container()

    # Handle user input first
    with input_container:
        user_input = st.chat_input("Ask me anything about ETHDocker setup and management...")

    # Display chat history in the container
    with chat_container:
        try:
            for msg in st.session_state.messages:
                if isinstance(msg, (ModelRequest, ModelResponse)):
                    for part in msg.parts:
                        display_message_part(part)
        except Exception as e:
            logger.error(f"Error displaying chat history: {str(e)}")
            st.error("Error displaying chat history. Try clearing the chat.")
            st.session_state.error_count += 1
            if st.session_state.error_count > 3:
                st.session_state.messages = []
                st.session_state.error_count = 0
                st.rerun()

    if user_input:
        try:
            # Append new request to conversation
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )
            
            # Display user prompt in the chat container
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Display the assistant's response
                with st.chat_message("assistant"):
                    await run_agent_with_streaming(user_input)

        except Exception as e:
            logger.error(f"Error in chat interaction: {str(e)}")
            st.error("An error occurred. Please try again or clear the chat history.")
            if st.session_state.messages:
                st.session_state.messages.pop()
            st.session_state.error_count += 1

if __name__ == "__main__":
    asyncio.run(main()) 