# Agent Testing Framework Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Flow](#system-flow)
3. [AgentTester](#agenttester)
4. [DiscoveryTester](#discoverytester)
5. [Knowledge Base](#knowledge-base)
6. [Agent Details](#agent-details)
   - [DiscoveryAgent](#discoveryagent)
   - [KnowledgeAgent](#knowledgeagent)
   - [MetricGenerationAgent](#metricgenerationagent)
   - [TestAgentOne](#testagentone)

---

## Overview

The Agent Testing Framework is a comprehensive system for analyzing call transcripts, building organizational knowledge bases, generating evaluation metrics, and testing agent performance. The framework consists of multiple specialized agents that work together to extract insights from customer call data.

### Key Components
- **AgentTester**: Tests metric evaluation on individual call transcripts
- **DiscoveryTester**: Analyzes batches of transcripts to build company knowledge
- **Knowledge Base**: Converts structured knowledge into vector-searchable chunks
- **Four Specialized Agents**: Each handling a specific aspect of call analysis

---

## System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Call Transcripts Input                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DiscoveryAgent                                │
│  • Analyzes 100+ call transcripts                               │
│  • Extracts company knowledge base                              │
│  • Builds product/service catalog                               │
│  • Identifies sales/service strategies                           │
│  • Recommends actionable metrics                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Structured JSON Output                               │
│  • company_knowledge_base                                        │
│  • product_catalog                                              │
│  • sales_service_strategy                                        │
│  • recommended_metrics                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KnowledgeAgent                                │
│  • Converts JSON to atomic facts                                 │
│  • Creates vector-searchable chunks                              │
│  • Categorizes by domain                                        │
│  • Stores in Mem0/Qdrant                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MetricGenerationAgent                         │
│  • Takes metric definition (JSON)                                │
│  • Generates evaluation prompt                                   │
│  • Converts metrics to per-call format                          │
│  • Creates detection rules                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TestAgentOne                                  │
│  • Evaluates single call transcript                              │
│  • Applies metric-specific prompt                               │
│  • Extracts metric_value and evidence                           │
│  • Returns structured JSON result                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Results Storage                               │
│  • MongoDB: agent collection                                    │
│  • Tracks inputs/outputs per agent                              │
│  • Timestamps and metadata                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Flow Steps

1. **Discovery Phase** (DiscoveryTester)
   - Input: Batch of call transcripts (100+)
   - Process: DiscoveryAgent analyzes patterns, extracts knowledge
   - Output: Comprehensive JSON with company knowledge, products, metrics

2. **Knowledge Base Creation** (knowlage_base.py)
   - Input: DiscoveryAgent output JSON
   - Process: KnowledgeAgent converts to atomic chunks
   - Output: Vector embeddings stored in Qdrant via Mem0

3. **Metric Evaluation** (AgentTester)
   - Input: Single call transcript + metric definition
   - Process: 
     - MetricGenerationAgent creates evaluation prompt
     - TestAgentOne applies prompt to transcript
   - Output: Metric value with evidence

---

## AgentTester

**File**: `apps/modeltester/src/modeltester/AgentTester.py`

### Purpose
Tests metric evaluation on individual call transcripts by:
1. Generating a metric-specific evaluation prompt
2. Applying the prompt to a call transcript
3. Extracting metric values and evidence
4. Storing results in MongoDB

### Workflow

```python
# 1. Initialize LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)

# 2. Retrieve call transcript from database
doc = get_document_by_id(db_name="call_iq", collection_name="audios", doc_id=36626)
transcript = doc.get('metadata', {}).get('transcript', "")

# 3. Generate metric evaluation prompt
_agent = MetricGenerationAgent(llm=llm)
metric_json = {
    "metric_name": "Upgrade Attempt Rate (UAR)",
    "definition": "Percentage of eligible renewal/recharge calls...",
    "calculation": "Count of Upgrade Attempts / Total Eligible Renewal Calls",
    ...
}
prompt_input, evaluation_prompt = _agent.handle(json.dumps(metric_json))

# 4. Store generated prompt
insert_documents('agent', 'agent', [{
    'company': "wheelseye",
    'agent': 'MetricGenerationAgent',
    'output': evaluation_prompt,
    'input': prompt_input,
    'time': datetime.now(),
}])

# 5. Evaluate transcript with TestAgentOne
_agent = TestAgentOne(llm=llm, prompt=evaluation_prompt, name=metric_json.get("metric_name"))
input_prompt, output = _agent.handle(transcript_text=transcript)

# 6. Store evaluation results
insert_documents('agent', 'agent', [{
    'company': "wheelseye",
    'agent': 'TestAgentOne',
    'output': output,
    'input': input_prompt,
    'time': datetime.now(),
}])
```

### Key Features
- **Metric Definition Input**: Accepts JSON metric definitions
- **Prompt Generation**: Creates evaluation prompts automatically
- **Transcript Evaluation**: Applies prompts to real call data
- **Result Persistence**: Stores all inputs/outputs for audit trail

### Usage Example

```python
# Define a metric
metric_json = {
    "metric_name": "Customer Follow-up Commitment Tracking",
    "definition": "Agent's reliability in completing promised follow-up actions",
    "calculation": "Count of Completed Commitments / Total Commitments Made",
    "business_rationale": "Failure to follow up breaks trust",
    "recommended_target": "> 95%",
    "priority": "high"
}

# Generate and apply
prompt_input, prompt = metric_agent.handle(json.dumps(metric_json))
result_input, result = test_agent.handle(transcript_text=transcript)
```

---

## DiscoveryTester

**File**: `apps/modeltester/src/modeltester/DiscoveryTest.py`

### Purpose
Analyzes large batches of call transcripts to build comprehensive organizational knowledge bases. This is the first step in the pipeline, discovering patterns and extracting structured information.

### Workflow

```python
# 1. Initialize LLM (typically Google Gemini for large context)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    google_api_key=os.getenv('GOOGLE_AI_API_KEY')
)

# 2. Initialize DiscoveryAgent
_dis = DiscoveryAgent(llm=llm)

# 3. Load call transcripts (from file or database)
with open("call_transcripts.txt", "r") as file:
    content = file.read()

# 4. Process transcripts
prompt, data = _dis.handle(call_transcripts=content)

# 5. Store results
insert_documents('agent', 'agent', [{
    'company': "wheelseye",
    'agent': 'DiscoveryAgent',
    'output': data,
    'input': prompt,
    'time': datetime.now(),
}])
```

### Output Structure

The DiscoveryAgent returns a comprehensive JSON structure:

```json
{
  "analysis_summary": {
    "transcripts_processed": 150,
    "date_range": "2024-01-01 to 2024-03-31",
    "confidence_level": "High"
  },
  "company_knowledge_base": {
    "company_profile": {...},
    "operational_info": {...},
    "communication_standards": {...}
  },
  "product_catalog": [...],
  "sales_service_strategy": {...},
  "recommended_metrics": {...},
  "implementation_recommendations": {...},
  "insights_and_observations": {...}
}
```

### Key Features
- **Batch Processing**: Handles 100+ transcripts efficiently
- **Pattern Recognition**: Identifies repeated phrases and themes
- **Comprehensive Extraction**: Builds knowledge base from scratch
- **Metric Recommendations**: Suggests business-specific metrics

---

## Knowledge Base

**File**: `apps/modeltester/src/modeltester/knowlage_base.py`

### Purpose
Converts structured JSON knowledge (from DiscoveryAgent) into atomic, vector-searchable chunks for storage in Mem0/Qdrant. This enables semantic search and retrieval of company knowledge.

### Workflow

```python
# 1. Initialize Mem0 with Qdrant backend
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "10.20.4.235", "port": 6333},
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

# 2. Initialize KnowledgeAgent
_KnowlageAgent = KnowlageAgent(llm=llm)

# 3. Get company data (from DiscoveryAgent output)
doc = get_document_by_id(db_name='agent', collection_name='agent', doc_id=ObjectId('...'))
company_data = {
    'company_knowledge_base': doc.get('output')['company_knowledge_base'],
    'product_catalog': doc.get('output')['product_catalog'],
    'sales_service_strategy': doc.get('output')['sales_service_strategy']
}

# 4. Convert to chunks
prompt, data = _KnowlageAgent.handle(company_data)

# 5. Store chunks in Mem0
for chunk in data.get('chunks', []):
    if chunk.get("text"):
        res = client.add(
            f'{chunk.get("category")} : {chunk.get("text")}', 
            user_id="wheelseye_kb"
        )
```

### Chunk Structure

Each chunk contains:
- **id**: Unique numeric identifier
- **category**: One of:
  - `company_profile`
  - `operational_info`
  - `communication_standards`
  - `Product_catalog`
  - `sales_service_strategy`
- **text**: Atomic fact in natural language

### Key Features
- **Atomic Facts**: Each chunk contains exactly one fact
- **Standalone Context**: Chunks are self-contained
- **Vector Searchable**: Stored in Qdrant for semantic retrieval
- **Categorized**: Organized by domain for better retrieval

### Utility Functions

#### `flatten_json(obj, parent_key="")`
Recursively flattens nested JSON structures into key-value pairs for processing.

#### `get_conpany_json_data()`
Returns example company knowledge base structure (currently returns hardcoded data; TODO: fetch from DiscoveryAgent response).

---

## Agent Details

### DiscoveryAgent

**Location**: `libs/llm_agent/src/llm_agent/CallAnaliserAgent.py`

#### Overview
The DiscoveryAgent is an intelligent analysis agent designed to process large corpora of call transcripts (100+) and build comprehensive organizational knowledge bases. It extracts patterns, understands business context, and recommends actionable metrics.

#### Core Responsibilities

1. **Company Knowledge Base Creation**
   - Extracts company profile (name, industry, business model, target segments)
   - Identifies operational information (processes, policies, compliance)
   - Documents communication patterns (greetings, brand voice, scripts)
   - Analyzes customer context (personas, journey stages, pain points)

2. **Product/Service Catalog**
   - Documents all products/services mentioned
   - Captures key features, pricing, common issues
   - Identifies target users and competitive positioning
   - Tracks upsell opportunities and integrations

3. **Sales & Service Strategy Analysis**
   - Analyzes sales approach (inbound/outbound, consultative/transactional)
   - Documents customer acquisition and retention tactics
   - Identifies problem resolution patterns
   - Tracks escalation triggers

4. **Metric Recommendations**
   - Core performance metrics (talk/listen ratio, handle time, FCR)
   - Quality metrics (script adherence, compliance, empathy)
   - Customer experience metrics (sentiment, effort score)
   - Sales & revenue metrics (upsell rates, objection handling)
   - Agent performance metrics (product knowledge, rapport building)
   - Business intelligence metrics (feature requests, churn indicators)
   - Custom business-specific metrics

#### Input/Output

**Input**:
- `call_transcripts` (str): Batch of call transcripts (100+ recommended)

**Output**:
```python
AnalysisOutput(
    analysis_summary: dict,
    company_knowledge_base: dict,
    product_catalog: list,
    sales_service_strategy: dict,
    recommended_metrics: dict,
    implementation_recommendations: dict,
    insights_and_observations: dict
)
```

#### Analysis Framework

The agent follows a structured 4-phase analysis:

1. **Phase 1: Company Knowledge Base**
   - Company Profile
   - Operational Information
   - Communication Patterns
   - Customer Context

2. **Phase 2: Product/Service Catalog**
   - Product documentation with full details
   - Feature extraction
   - Issue tracking

3. **Phase 3: Sales & Service Strategy**
   - Sales approach analysis
   - Customer acquisition/retention
   - Problem resolution patterns

4. **Phase 4: Recommended Metrics**
   - 7 categories of metrics
   - Business-specific custom metrics

#### Best Practices
- **Evidence-Based**: All findings must reference actual transcript content
- **Pattern Validation**: Requires multiple examples before declaring patterns
- **Gap Identification**: Explicitly notes missing or unclear information
- **Prioritization**: Ranks recommendations by business impact
- **Confidence Scoring**: Indicates confidence level for each finding

#### Red Flags Detected
- Compliance violations or gaps
- Inconsistent product/policy information
- Poor customer outcomes
- Systemic process failures
- Training gaps
- Technology limitations
- Customer sentiment deterioration

#### Usage Example

```python
from llm_agent import DiscoveryAgent
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
agent = DiscoveryAgent(llm=llm)

# Load transcripts
with open("transcripts.txt", "r") as f:
    transcripts = f.read()

# Analyze
prompt, results = agent.handle(call_transcripts=transcripts)

# Access results
company_kb = results.company_knowledge_base
products = results.product_catalog
metrics = results.recommended_metrics
```

---

### KnowledgeAgent

**Location**: `libs/llm_agent/src/llm_agent/KnowlageAgent.py`

#### Overview
The KnowledgeAgent (spelled "KnowlageAgent" in code) is an expert knowledge-base architect that converts structured JSON knowledge into atomic, vector-searchable memory chunks optimized for Mem0-style storage and retrieval.

#### Core Responsibilities

1. **Chunk Creation**
   - Converts JSON to atomic facts (one fact per chunk)
   - Writes chunks in clear natural language
   - Ensures chunks are standalone with sufficient context
   - Categorizes chunks by domain

2. **Vector Optimization**
   - Optimizes chunks for vector retrieval
   - Ensures Mem0-compatible format
   - Maintains semantic meaning in atomic units

#### Input/Output

**Input**:
- `company_data` (Dict): JSON object containing:
  - `company_knowledge_base`
  - `product_catalog`
  - `sales_service_strategy`

**Output**:
```python
ChunkList(
    chunks: List[Chunk]
)

# Where Chunk is:
Chunk(
    id: int,
    category: str,  # One of: company_profile, operational_info, 
                    #         communication_standards, Product_catalog,
                    #         sales_service_strategy
    text: str       # Atomic natural-language fact
)
```

#### Chunk Creation Rules

1. **One Fact Per Chunk**: Each chunk contains exactly one atomic fact
2. **Natural Language**: Written in clear, natural language (not JSON)
3. **Standalone**: Contains enough context to understand without original JSON
4. **Categorized**: Includes appropriate category tag
5. **Complete**: Includes every detail from input JSON (no summarization)
6. **No Combination**: Never combines multiple facts

#### Categories

- `company_profile`: Company information, values, positioning
- `operational_info`: Processes, policies, compliance requirements
- `communication_standards`: Greetings, brand voice, scripts
- `Product_catalog`: Product/service information
- `sales_service_strategy`: Sales approaches, resolution patterns

#### Usage Example

```python
from llm_agent import KnowlageAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
agent = KnowlageAgent(llm=llm)

# Company data from DiscoveryAgent
company_data = {
    "company_knowledge_base": {...},
    "product_catalog": [...],
    "sales_service_strategy": {...}
}

# Convert to chunks
prompt, chunks = agent.handle(company_data=company_data)

# Store in Mem0
for chunk in chunks.chunks:
    memory.add(
        f"{chunk.category}: {chunk.text}",
        user_id="company_kb"
    )
```

#### Integration with Mem0

The chunks produced by KnowledgeAgent are designed to be stored in Mem0 (with Qdrant backend) for semantic search:

```python
from mem0 import Memory

memory = Memory.from_config({
    "vector_store": {"provider": "qdrant", ...},
    "llm": {"provider": "openai", ...},
    "embedder": {"provider": "openai", ...}
})

# Add chunks
for chunk in chunks.chunks:
    memory.add(f"{chunk.category}: {chunk.text}", user_id="wheelseye_kb")
```

---

### MetricGenerationAgent

**Location**: `libs/llm_agent/src/llm_agent/MetricGenerationAgent.py`

#### Overview
The MetricGenerationAgent converts metric definitions (in JSON format) into single-call evaluation prompts. It transforms high-level metric definitions into actionable detection rules that can be applied to individual call transcripts.

#### Core Responsibilities

1. **Metric Conversion**
   - Converts percentage/ratio metrics to per-call boolean detection
   - Maintains per-call metrics in appropriate format
   - Derives detection rules from metric definitions

2. **Prompt Generation**
   - Creates evaluation prompts with clear instructions
   - Defines eligibility criteria
   - Specifies evidence requirements
   - Includes output schema

#### Input/Output

**Input**:
- `metric_json` (str): JSON string containing metric definition:
  ```json
  {
    "metric_name": "Upgrade Attempt Rate (UAR)",
    "definition": "Percentage of eligible renewal/recharge calls...",
    "calculation": "Count of Upgrade Attempts / Total Eligible Renewal Calls",
    "business_rationale": "...",
    "recommended_target": "> 85%",
    "priority": "critical"
  }
  ```

**Output**:
- `(prompt_input, evaluation_prompt)`: Tuple containing:
  - `prompt_input`: The formatted input sent to LLM
  - `evaluation_prompt`: The generated prompt for TestAgentOne

#### Conversion Rules

1. **Percentage/Ratio Metrics** → **Per-Call Boolean**
   - `true`: Event occurred in this call
   - `false`: Event did not occur (but call is eligible)
   - `null`: Call not eligible or insufficient evidence
   - Adds rule: "Do NOT compute any percentage at the call level"

2. **Per-Call Metrics** → **Maintain Format**
   - Keeps appropriate data_type (boolean, int, float, text)
   - No conversion needed

3. **Detection Rules**
   - What counts as an event?
   - What makes the call eligible?
   - What phrases/patterns indicate positive detection?

#### Output Schema

The generated prompt instructs TestAgentOne to return:

```json
{
  "metric_name": "<metric_name>",
  "data_type": "<list | int | float | percentage | boolean | text | null>",
  "metric_value": <value or null>,
  "evidence": "<short reasoning with transcript quotes>"
}
```

#### Prompt Structure

The generated prompt includes:

1. **Definition**: Rewritten metric definition in simple detection terms
2. **Task Steps**:
   - Determine call eligibility
   - Detect if event occurred
   - Use only explicit transcript evidence
3. **Evidence Selection**: Up to 3 exact transcript quotes
4. **Instructions**: 
   - Prefer diarized transcript if available
   - Return valid JSON only
   - Follow output schema

#### Usage Example

```python
from llm_agent import MetricGenerationAgent
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
agent = MetricGenerationAgent(llm=llm)

# Define metric
metric = {
    "metric_name": "Customer Follow-up Commitment Tracking",
    "definition": "Agent's reliability in completing promised follow-up actions",
    "calculation": "Count of Completed Commitments / Total Commitments Made",
    "business_rationale": "Failure to follow up breaks trust",
    "recommended_target": "> 95%",
    "priority": "high"
}

# Generate prompt
prompt_input, evaluation_prompt = agent.handle(json.dumps(metric))

# Use evaluation_prompt with TestAgentOne
test_agent = TestAgentOne(llm=llm, prompt=evaluation_prompt)
result = test_agent.handle(transcript_text=transcript)
```

#### Key Features

- **Automatic Conversion**: Handles complex metrics automatically
- **Evidence-Based**: Requires explicit transcript quotes
- **Flexible Input**: Accepts various metric definition formats
- **Structured Output**: Generates prompts with consistent schema

---

### TestAgentOne

**Location**: `libs/llm_agent/src/llm_agent/TestAgentOne.py`

#### Overview
TestAgentOne is a flexible evaluation agent that applies custom prompts (typically generated by MetricGenerationAgent) to call transcripts. It extracts metric values and evidence from individual calls.

#### Core Responsibilities

1. **Transcript Evaluation**
   - Applies metric-specific prompts to call transcripts
   - Extracts metric values based on detection rules
   - Collects supporting evidence from transcripts

2. **Flexible Prompting**
   - Accepts any prompt template
   - Supports both plain and diarized transcripts
   - Returns structured JSON results

#### Input/Output

**Input**:
- `prompt` (str): Evaluation prompt (usually from MetricGenerationAgent)
- `transcript_text` (str): Plain call transcript
- `diarized_transcript` (str, optional): Speaker-separated transcript

**Output**:
- `(input_prompt, output)`: Tuple containing:
  - `input_prompt`: The formatted prompt sent to LLM
  - `output`: JSON result with metric_value and evidence

#### Output Format

The agent returns JSON matching the schema defined in the prompt (typically from MetricGenerationAgent):

```json
{
  "metric_name": "Upgrade Attempt Rate (UAR)",
  "data_type": "boolean",
  "metric_value": true,
  "evidence": "Agent mentioned upgrade: 'Aap 4G device le sakte hain, sirf ₹500 extra' [line 45]"
}
```

#### Transcript Types

1. **Plain Transcript**: Standard text transcript
2. **Diarized Transcript**: Speaker-separated format (preferred if available)
   - Format: `[Speaker]: [Text]`
   - Example: `[Agent]: Hello, how can I help? [Customer]: My GPS is offline`

#### Usage Example

```python
from llm_agent import TestAgentOne, MetricGenerationAgent
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)

# Step 1: Generate evaluation prompt
metric_agent = MetricGenerationAgent(llm=llm)
metric_json = {
    "metric_name": "Upgrade Attempt Rate",
    "definition": "...",
    ...
}
_, evaluation_prompt = metric_agent.handle(json.dumps(metric_json))

# Step 2: Evaluate transcript
test_agent = TestAgentOne(
    llm=llm, 
    prompt=evaluation_prompt,
    name="Upgrade Attempt Rate"
)

# With plain transcript
input_prompt, result = test_agent.handle(transcript_text=transcript)

# With diarized transcript (preferred)
input_prompt, result = test_agent.handle(
    transcript_text=transcript,
    diarized_transcript=diarized_transcript
)

# Result contains:
# {
#   "metric_name": "Upgrade Attempt Rate",
#   "data_type": "boolean",
#   "metric_value": true,
#   "evidence": "..."
# }
```

#### Integration with AgentTester

TestAgentOne is typically used in the AgentTester workflow:

```python
# In AgentTester.py
_agent = MetricGenerationAgent(llm=llm)
_, evaluation_prompt = _agent.handle(json.dumps(metric_json))

_agent = TestAgentOne(llm=llm, prompt=evaluation_prompt, name=metric_name)
input_prompt, output = _agent.handle(transcript_text=transcript)
```

#### Key Features

- **Flexible**: Accepts any prompt template
- **Dual Transcript Support**: Handles both plain and diarized formats
- **Structured Output**: Returns consistent JSON format
- **Evidence Extraction**: Includes transcript quotes in results

---

## Data Storage

All agents store their inputs and outputs in MongoDB:

**Collection**: `agent.agent`

**Document Structure**:
```json
{
  "company": "wheelseye",
  "agent": "DiscoveryAgent | KnowlageAgent | MetricGenerationAgent | TestAgentOne",
  "input": "...",
  "output": {...},
  "time": ISODate("2024-01-01T00:00:00Z")
}
```

This enables:
- **Audit Trail**: Track all agent operations
- **Reproducibility**: Re-run analyses with same inputs
- **Debugging**: Inspect intermediate results
- **Analytics**: Analyze agent performance over time

---

## Best Practices

### For DiscoveryAgent
- Process 100+ transcripts for reliable patterns
- Batch process in groups of 10-20 for efficiency
- Validate patterns with multiple examples
- Review recommended metrics for business relevance

### For KnowledgeAgent
- Ensure input JSON is complete and structured
- Verify chunks are atomic and standalone
- Use appropriate categories for better retrieval
- Store chunks immediately after generation

### For MetricGenerationAgent
- Define metrics with clear definitions and calculations
- Include business rationale for context
- Specify recommended targets/thresholds
- Test generated prompts before production use

### For TestAgentOne
- Prefer diarized transcripts when available
- Review evidence quotes for accuracy
- Handle null values appropriately (call not eligible)
- Store results for aggregation across calls

---

## Troubleshooting

### DiscoveryAgent Issues
- **Low Confidence**: Increase transcript count (100+)
- **Missing Products**: Check transcript quality and completeness
- **Generic Metrics**: Review transcripts for business-specific patterns

### KnowledgeAgent Issues
- **Invalid JSON**: Ensure input is properly formatted
- **Missing Chunks**: Check that all JSON fields are included
- **Poor Retrieval**: Verify chunks are atomic and well-categorized

### MetricGenerationAgent Issues
- **Unclear Prompts**: Ensure metric definition is specific
- **Wrong Data Type**: Review conversion rules for percentage metrics
- **Missing Evidence**: Adjust prompt to require explicit quotes

### TestAgentOne Issues
- **Null Results**: Check call eligibility criteria
- **Incorrect Values**: Review detection rules in prompt
- **Missing Evidence**: Ensure transcript contains relevant content

---

## Future Enhancements

1. **Batch Evaluation**: Process multiple transcripts in parallel
2. **Metric Aggregation**: Automatically compute percentages from per-call results
3. **Confidence Scores**: Add confidence levels to all agent outputs
4. **A/B Testing**: Compare different prompt variations
5. **Real-time Processing**: Stream transcripts for live analysis
6. **Multi-language Support**: Handle transcripts in multiple languages
7. **Custom Categories**: Allow user-defined chunk categories

---

## Conclusion

The Agent Testing Framework provides a comprehensive solution for analyzing call transcripts, building knowledge bases, and evaluating metrics. Each agent plays a specific role in the pipeline, from initial discovery to final evaluation. By following the documented workflows and best practices, you can effectively extract insights from customer call data and improve agent performance.

