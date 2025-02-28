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
            st.markdown(f"🔧 **Tool Call**: {part.name}")
            with st.expander("View Tool Call Details"):
                # Display tool arguments in a more readable format
                if hasattr(part, 'arguments'):
                    try:
                        args = json.loads(part.arguments)
                        st.json(args)
                    except json.JSONDecodeError:
                        st.code(part.arguments)
    elif part.part_kind == 'tool-return':
        with st.chat_message("tool"):
            st.markdown("🔄 **Tool Result**")
            with st.expander("View Tool Result"):
                try:
                    # Try to parse as JSON for better formatting
                    result = json.loads(part.content)
                    st.json(result)
                except json.JSONDecodeError:
                    # If not JSON, display as markdown
                    st.markdown(part.content)

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

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Ask me anything about ETHDocker setup and management...")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main()) 