import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# On chat start
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hi, I'm your Healthcare Chatbot. Feel free to share the health problems that you are facing.").send()


ollama_base_url = "YOUR_LOCAL_OLLAMA_BASE_URL"
chat = ChatOpenAI(base_url=ollama_base_url, openai_api_key="apikey", model="llama3.1")

system_message = SystemMessage(content="You are a healthcare chatbot that is able to diagnose a disease based on symptoms as well as provide symptoms for any disease and provide preventive measures for the same. Make sure to answer prompts only related to healthcare and not anything else. Do not ask any questions to the user and address current prompt first and add any other content from previous context.After provding diagnosis , give suggestions to prevent and overcome the issues.Provide prescriptions too")
@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("message_history", [])
    
    if len(message.elements) > 0:
        for element in message.elements:
            messages.append({"role": "user", "content": element.content.decode("utf-8")})
            confirm_message = cl.Message(content=f"Uploaded file: {element.name}")
            await confirm_message.send()  

    user_message = HumanMessage(content=message.content)
    messages.append({"role": "user", "content": user_message.content})

    all_messages = [system_message] + messages
    try:
        response = chat.invoke(all_messages)

        parser = StrOutputParser()
        parsed_response = parser.invoke(response)
        
        msg = cl.Message(content=parsed_response)
        await msg.send()
        
        cl.user_session.set("message_history", all_messages)
        
    except Exception as e:
        # Handle and report errors
        error_message = f"An error occurred: {str(e)}"
        error_msg = cl.Message(content=error_message)
        await error_msg.send()
