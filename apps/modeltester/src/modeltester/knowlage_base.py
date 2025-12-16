import datetime

from bson import ObjectId
from langchain_openai import ChatOpenAI
from llm_agent import KnowlageAgent
from database import get_document_by_id, insert_documents

# --------- Initialize Mem0 ----------
import os
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

os.environ[
    "OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": f"{os.getenv('QDRANT_HOST', '10.20.4.235')}", "port": 6333},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-5-mini", "temperature": 0.1},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
}

client = Memory.from_config(config)

# --------- Your Knowledge Base (paste your JSON dict here) ---------
llm = ChatOpenAI(
    model="gpt-5",  # or "gpt-4o"
    temperature=0.7,  # 0.0–0.3 for scoring/determinism
    api_key=os.getenv('OPENAI_API_KEY'),
)

_KnowlageAgent = KnowlageAgent(llm=llm)


# --------- Utility: Flatten & Chunk the JSON ---------
def flatten_json(obj, parent_key=""):
    """
    Converts nested dict/list JSON into flat (key_path -> value_text) pairs.
    """
    items = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            items.extend(flatten_json(v, new_key))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(v, new_key))

    else:
        # Base (key, value) pair → Mem0-friendly chunk
        text = f"{parent_key}: {obj}"
        items.append(text)

    return items


def get_conpany_json_data():
    """
    ToDo: we will fet company data from DiscoveryAgent response
    {key: doc.get('output')[key] for key in ['company_knowledge_base']
    Returns:

    """
    return {
    "company_knowledge_base": {
        "company_profile": {
            "name": "Wheelseye (Transcribed variously as Win Sai, Vilsai, Will Say)",
            "industry": "Logistics Tech / Telematics",
            "business_model": "B2B Subscription (SaaS + Hardware)",
            "target_segments": [
                "Fleet Owners",
                "Individual Truck Drivers",
                "Logistics Companies"
            ],
            "key_values": [
                "Relationship Management (Account Managers)",
                "Security/Anti-theft",
                "24/7 Support"
            ]
        },
        "operational_info": {
            "processes": [
                "Account Manager aligned to specific customers",
                "Ticket creation for technical issues (offline devices)",
                "Technician assignment for installation/repair"
            ],
            "policies": {
                "Warranty/Service": "On-site service is often free or included in subscription",
                "Renewals": "Subscriptions expire annually; grace periods discussed",
                "Upgrades": "Exchange offers available for upgrading old devices to 4G"
            },
            "compliance_requirements": [
                "KYC documentation for account changes",
                "Fastag linking restrictions",
                "AIS 140 Government Mandate compliance"
            ]
        },
        "communication_standards": {
            "required_elements": [
                "Greeting (Namaskar/Hello)",
                "Identification as Account Manager",
                "Confirmation of vehicle numbers"
            ],
            "brand_voice": "Consultative, Persuasive, Informal (Hinglish)",
            "scripted_components": [
                "Pitching the '12+4 month' Diwali offer",
                "Asking for referrals/new vehicle additions"
            ]
        }
    },
    "product_catalog": [
        {
            "product_id": "GPS_BASIC",
            "name": "Standard/Basic GPS (Likely 2G)",
            "category": "Hardware",
            "description": "Basic tracking device, wired connection.",
            "key_features": [
                "Real-time tracking",
                "Playback history"
            ],
            "pricing_model": "Subscription based (~₹2200-₹3000 range mentioned)",
            "common_issues": [
                "Network loss",
                "Wire cutting",
                "Device offline"
            ],
            "target_users": "Cost-conscious fleet owners",
            "mention_frequency": 15
        },
        {
            "product_id": "GPS_PRO_4G",
            "name": "Pro GPS / 4G Device",
            "category": "Hardware",
            "description": "Advanced tracking device with better network connectivity.",
            "key_features": [
                "4G Connectivity",
                "Engine Lock/Cut (Ignition kill)",
                "Siren/Alarm integration",
                "Parking Mode/Watchman feature"
            ],
            "pricing_model": "Premium tier, often sold via upgrade exchange offers",
            "common_issues": [
                "Installation incorrect location"
            ],
            "target_users": "High-value cargo transporters, BS4/BS6 vehicle owners",
            "mention_frequency": 10
        },
        {
            "product_id": "AIS_140",
            "name": "AIS 140 Certified Device",
            "category": "Compliance Hardware",
            "description": "Government mandated GPS for commercial vehicles.",
            "key_features": [
                "RTO Compliance",
                "Panic Button support (implied)",
                "Certificate issuance"
            ],
            "pricing_model": "Higher price point, mandatory for fitness passing",
            "common_issues": [
                "Certificate generation delays",
                "RTO integration issues"
            ],
            "target_users": "Commercial vehicles requiring fitness certificate",
            "mention_frequency": 5
        },
        {
            "product_id": "FASTAG",
            "name": "Fastag",
            "category": "Service",
            "description": "Toll payment tag.",
            "key_features": [
                "Wallet integration"
            ],
            "pricing_model": "Prepaid/Wallet",
            "common_issues": [
                "Blacklisting due to low balance",
                "Registered to wrong mobile number"
            ],
            "target_users": "All toll-passing vehicles",
            "mention_frequency": 3
        }
    ],
    "sales_service_strategy": {
        "sales_approach": {
            "style": "Relationship-driven (Account Manager model)",
            "value_propositions": [
                "16 months service for price of 12 (Diwali Offer)",
                "Free on-site service/no technician charges",
                "Security: Remote engine lock to prevent theft"
            ],
            "objection_patterns": [
                "\"Rate is too high\"",
                "\"Vehicle is standing/not working\"",
                "\"Competitor (local mechanic) is cheaper\""
            ],
            "upsell_techniques": [
                "Exchange Offer: Replace old device with new 4G device for better network",
                "Bundle Deal: Discounts for renewing multiple vehicles at once"
            ]
        },
        "resolution_patterns": {
            "common_issues": [
                "Device Offline (Network/Power issue)",
                "Recharge Expired",
                "Technician scheduling delays"
            ],
            "resolution_paths": {
                "Technical": "Create ticket -> Assign Technician -> On-site visit",
                "Billing": "Payment link via App/Whatsapp -> Order ID generation"
            },
            "escalation_triggers": [
                "Repeat offline issues",
                "Delay in technician arrival",
                "Threatening to switch to competitor"
            ]
        }
    }
}

# --------- Store chunks in Mem0 ---------
if __name__ == "__main__":

    company_data = get_conpany_json_data()
    mem_ids = []
    promt, data = _KnowlageAgent.handle(company_data)
    print(data)

    print(insert_documents('agent', 'agent', [{
        'company': "wheelseye",
        'agent': 'KnowlageAgent',
        'time': datetime.datetime.now(),
        'input': promt,
        'output': data
    }]))



    # for chunk in data.get('chunks', []):
    #     if (chunk.get("text", None)):
    #         continue
    #     res = client.add(f'{chunk.get("category", "category")} : {chunk.get("text", "")}', user_id="wheelseye_kb")
    #     print(res)
    #     print("Added memory:", chunk)
    #
    # print("\n✔ All memory chunks saved to Mem0.")
    # print("Memory IDs:", mem_ids)
