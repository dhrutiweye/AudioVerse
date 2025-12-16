from datetime import datetime
import json
import os

from database import get_document_by_id, insert_documents
from llm_agent import TestAgentOne, MetricGenerationAgent
from llm_agent import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm = ChatOpenAI(
        model="gpt-5-mini",  # or "gpt-4o"
        temperature=0.7,  # 0.0–0.3 for scoring/determinism
        api_key=os.getenv('OPENAI_API_KEY'),
    )
    doc = get_document_by_id(
        db_name="call_iq",
        collection_name="audios",
        doc_id=int(36626))

    transcript = (doc.get('metadata', {}).get('transcript', ""))

    _ageny = MetricGenerationAgent(llm=llm)
    metric_json = {'metric_name': 'GPS Location Accuracy Issue Rate', 'definition_calculation': '(Number of calls reporting incorrect/no GPS location / Total calls related to GPS functionality) * 100', 'why_it_matters': 'Directly impacts core product value; high rate indicates product/service reliability issues or installation quality.', 'recommended_threshold_target': '<10%', 'frequency_of_measurement': 'Daily'}
    metric_json2 = {
        "metric_name": "Upgrade Attempt Rate (UAR)",
        "definition": "Percentage of eligible renewal/recharge calls where the agent pitches an upgrade (e.g., Normal to GPS Pro, 2G to 4G).",
        "calculation": "Count of Upgrade Attempts / Total Eligible Renewal Calls",
        "business_rationale": "This is a key outbound sales strategy for revenue maximization from the existing customer base.",
        "recommended_target": "> 85%",
        "priority": "critical"
      }
    metric_json3 = {
        "metric_name": "Customer Follow-up Commitment Tracking",
        "definition": "Agent's reliability in completing promised follow-up actions (e.g., 'I will WhatsApp the order ID' [cite: 2361][cite_start], 'I will call you tomorrow' [cite: 2341][cite_start], 'I will update the location').",
        "calculation": "Count of Completed Commitments / Total Commitments Made",
        "business_rationale": "Failure to follow up (especially on account issues) breaks the trust built by the 'dedicated manager' model.",
        "recommended_target": "> 95%",
        "priority": "high"
      }
    _promt_in, _promt = _ageny.handle(json.dumps(metric_json2))
    print(_promt)

    insert_documents('agent', 'agent', [{
        'company': "wheelseye",
        'agent': 'MetricGenerationAgent',
        'output': _promt,
        'time': datetime.now(),
        'input': _promt_in,
    }])

    _agent = TestAgentOne(llm=llm, prompt=_promt, name=metric_json.get("metric_name"))
    input_promt, output = _agent.handle(transcript_text=transcript)

    insert_documents('agent', 'agent', [{
        'company': "wheelseye",
        'agent': 'TestAgentOne',
        'output': output,
        'time': datetime.now(),
        'input': input_promt,
    }])

    print(output)
