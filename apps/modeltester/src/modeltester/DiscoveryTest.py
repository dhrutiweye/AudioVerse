import os
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from llm_agent import DiscoveryAgent
from database import insert_documents
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,
                                 google_api_key=os.getenv('GOOGLE_AI_API_KEY'))
    _dis = DiscoveryAgent(llm=llm)
    with open("/Users/weye/Documents/code/python/testDb/ModelTester/file/doc2.txt", "r") as file:
        content = file.read()

    print(content.split("\n"))

    promt, data = _dis.handle(call_transcripts=content)

    insert_documents('agent', 'agent', [{
        'company': "wheelseye",
        'agent': 'DiscoveryAgent',
        'output': data,
        'time': datetime.now(),
        'input': promt,
    }])