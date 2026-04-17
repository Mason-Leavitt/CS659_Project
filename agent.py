# agent.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tools import TOOLS

load_dotenv()

# Model configuration
openAiModel = "gpt-4o-mini"  # Using a reliable model
llm = ChatOpenAI(model=openAiModel, temperature=0)

# Bind tools to the model
llm_with_tools = llm.bind_tools(TOOLS)

def run_query(query: str) -> str:
    """
    Run a query through the vision agent.
    
    Args:
        query: User's question about images
        
    Returns:
        Natural language response
    """
    try:
        # System message for the agent
        system_msg = """You are a vision assistant that analyzes images using object detection.

Your task is to help users understand what objects are in their images.

When analyzing images:
1. If asked about images in a folder, use the analyze_all_images tool
2. If asked about a specific image, use the detect_objects tool  
3. If asked to list images first, use the list_images tool
4. Always provide clear, natural language summaries

Be concise and helpful in your responses."""

        # First, let the model decide which tool to use
        response = llm_with_tools.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query}
        ])
        
        # Check if the model wants to use a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Find and execute the tool
            tool_to_use = None
            for tool in TOOLS:
                if tool.name == tool_name:
                    tool_to_use = tool
                    break
            
            if tool_to_use:
                # Execute the tool
                tool_result = tool_to_use.invoke(tool_args)
                
                # Get final response from model with tool result
                final_response = llm.invoke([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": f"I used {tool_name} and got: {tool_result}"},
                    {"role": "user", "content": "Please provide a natural language summary of these results."}
                ])
                
                return final_response.content
            else:
                return f"Tool {tool_name} not found"
        else:
            # No tool needed, return direct response
            return response.content
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error processing query: {str(e)}\n\nDetails:\n{error_details}\n\nPlease make sure:\n1. OpenAI API key is set in .env file\n2. Images folder exists\n3. YOLO model is downloaded"
