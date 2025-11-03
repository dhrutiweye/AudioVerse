from datetime import datetime, timedelta
from typing import Dict, Any
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from llm_agent import SignleQueriAgent
from database import get_documents_by_date_range, get_documents_by_filter, insert_documents
load_dotenv()
# from WriteEmbarding import *

llm = ChatOpenAI(
    model="gpt-5-mini",  # or "gpt-4o"
    temperature=0.0,  # 0.0–0.3 for scoring/determinism
    api_key=os.getenv('OPENAI_API_KEY'),
)

queries = [
    'Dmart opening at Sec 48 Gurgaon', 'Customer is asking for discount on diesel sensor',
    'Customer declined purchased because he is not confident on the quality of delivery in Diesel sensor',
    'Customer has agreed for taking home loan',
    'Per perplexity comet is the future of browsing',
    'Customer declined purchased because blackbuck / competition is offering cheaper diesel sensor',
    'Customer wants refunds for his diesel sensor because of below expectations performance',
    'Customer agreed to purchase diesel sensor',
    'Customer has heard bad reviews of diesel sensor from his colleagues',
    'Customer is really angry and using bad language about diesel sensor', 'Price too high for diesel sensor',
    'Customer would like to get complete vitamin profile done after a month’s time',
    'Diesel sensor gives wrong readings', 'Open Ai is soon going to launch a job platform',
    'Customer wants to add certain feature in product', 'Not repaying back loan and asking for more time',
    'Happy about India stance on US tariffs ',
    'Annual Fastag plans is live from mid of Aug 2025 and can be purchased from banks',
    'Pitched Diesel sensor to customer',
    'Customer is very happy with diesel sensor performance and would like to buy for his other vehicles',
    'Customer wants to know when he will get the blood test report',
    'Customer postponed purchase of diesel sensor, will buy later',
    'Conversation reached till price stage in sales conversation for diesel sensor',
    'Customer is facing a bug in the Mobile app', 'Home loan interest rates are too high',
    'Certain feature in product is not working properly'
]


def get_complite_transcript(doc):
    x = doc.get("metadata", {}).get("diarized_transcript", {}).get("entries", []) or []
    convertion = ""
    for i in x:
        # prefer end_time_seconds if present; else approximate 2s window
        start_s = float(i.get("start_time_seconds", 0) or 0.0)
        end_s = float(i.get("end_time_seconds", start_s) or start_s)
        speaker = i.get("speaker_id", "")
        text = i.get("transcript", "")
        convertion += f"{speaker}: {text}\n"
    return convertion


def set_recall_data(data: Dict[str, Any]):
    agent = SignleQueriAgent(llm=llm)
    for _q in queries:
        _result = get_documents_by_filter(db_name="call_iq",
                                           collection_name="call_test_data",
                                           data= {'query': _q, 'call_id': int(data.get('_id'))})
        if(_result is not None and len(_result) > 0):
            continue
        result = agent.handle(
            query_string=_q,
            transcript_text=None,
            diarized_transcript=get_complite_transcript(data)
        )
        new_doc = [{
            "call_id": data.get('_id'),
            "query": _q,
            "llm_score": result.get("overall_score", 0.0),
            "llm_is_related": result.get("is_related"),
            "llm_verdict": result.get("verdict"),
            "llm_meta": str(result),  # quick way; or use json.dumps(result)
            "created_at": datetime.utcnow()
        }]
        print(insert_documents(db_name='call_iq', collection_name="call_test_data", data_list=new_doc))
        print(result)

def set_recall_dataV2(data: Dict[str, Any]):
    agent = SignleQueriAgent(llm=llm)
    _result = get_documents_by_filter(db_name="call_iq",
                                       collection_name="call_test_data",
                                       data= {'call_id': int(data.get('_id'))})
    if(_result is not None and len(_result) > 0):
        return
    for _q in queries:
        try:
            result = agent.handle(
                query_string=_q,
                transcript_text=None,
                diarized_transcript=get_complite_transcript(data)
            )
        except Exception as ex:
            print(f"get Exception {ex}")
            continue

        new_doc = [{
            "call_id": data.get('_id'),
            "query": _q,
            "llm_score": result.get("overall_score", 0.0),
            "llm_is_related": result.get("is_related"),
            "llm_verdict": result.get("verdict"),
            "llm_meta": str(result),
            "created_at": datetime.utcnow()
        }]
        print(insert_documents(db_name='call_iq', collection_name="call_test_data_v2", data_list=new_doc))
        print(f"call_id: {data.get('_id')}")
        print(result)


if __name__ == "__main__":

    today = datetime.now()
    yesterday = today - timedelta(days=6)
    print(f"start recall sync for {yesterday}")

    start = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)
    end = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
    page = 2
    batch_size = 20
    while True:
        docs = get_documents_by_date_range(db_name="call_iq",
                                           collection_name="audios",
                                           start_date=start,
                                           end_date=end,
                                           limit=batch_size,
                                           offset=(page * batch_size))
        print(f"page: {page} size: {len(docs)}")
        if len(docs) < batch_size:
            break
        for i in docs:
            set_recall_dataV2(i)
        page += 1

    print(f"end recall sync for {yesterday}")

