from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")
llm.invoke("hi")

if __name__ == "__main__":
    response = llm.invoke("What are the two main ingredients in samosa")
    print(response.content)
