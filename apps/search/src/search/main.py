from SearchTranscript import search
from SearchRequest import SearchRequest
import csv
from database import get_documents_by_filter
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    queries = [
        'Dmart opening at Sec 48 Gurgaon', 'Customer is asking for discount on diesel sensor',
        'Customer declined purchased because he is not confident on the quality of delivery in Diesel sensor',
        'Customer has agreed for taking home loan', 'Certain feature in product is not working properly',
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
        'Customer is facing a bug in the Mobile app', 'Home loan interest rates are too high'
    ]
    data = []
    d2=[]
    for _q in queries:
        m_data = []
        t_data = []
        req = SearchRequest(query=_q,
                            size=100, page=0, group_hits=1, min_score=0.2, rerank_gate_prob=0.0000001)
        results = search(req)
        print(results)
        for i in results.get('results', []):
            m_data.append(i.get('call_id'))
        [data.append([_q, i.get('call_id')]) for i in results.get('results', [])]
        docs = get_documents_by_filter(
            db_name='call_iq',
            collection_name='call_test_data_v2',
            data={
                'query': _q,
                'llm_is_related': True
            }
        )
        for i in docs:
            t_data.append(i.get('call_id'))
        print(results)
        print(docs)
        print(f"m: {m_data} t:{t_data}")

        common = [x for x in m_data if x in t_data]
        # l2_diff = [x for x in t_data if x not in m_data]
        d2.append([_q,
                   float(len(common)/len(m_data)) if len(m_data) > 0 else 0,
                   float(len(common)/len(t_data)) if len(t_data) > 0 else 0 ])

    print(data)
    with open("call_data_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "call_id"])  # header
        writer.writerows(data)

    with open("recall_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "presition", "recall"])  # header
        writer.writerows(d2)

    print("✅ CSV file 'call_data.csv' created successfully!")
