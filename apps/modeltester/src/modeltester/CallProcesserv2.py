import os
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings
from database import get_document_by_id
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class ChunkData(BaseModel):
    chunk_id: str = Field(description="Unique identifier like chunk_0, chunk_1, etc")
    start_time: str = Field(description="Start timestamp in format MM:SS or HH:MM:SS")
    end_time: str = Field(description="End timestamp in format MM:SS or HH:MM:SS")
    speaker_primary: str = Field(description="Primary speaker in this chunk")
    phase: str = Field(description="Call phase: intro/discovery/pitch/negotiation/closing/followup")
    summary: str = Field(description="2-3 sentence summary of this chunk")
    topics: List[str] = Field(description="Main topics discussed in this chunk")
    importance_score: float = Field(description="Importance score from 0.0 to 1.0")
    importance_reasoning: str = Field(description="Brief explanation of importance score")
    emotional_tone: str = Field(description="Emotional tone: positive/negative/neutral/mixed/tense")
    keywords: List[str] = Field(description="Key terms: products, pain points, commitments, pricing terms")

    # Metadata flags
    contains_question: bool = Field(description="Contains customer questions")
    contains_commitment: bool = Field(description="Contains commitments or promises")
    contains_product_mention: bool = Field(description="Mentions specific products")
    contains_pricing: bool = Field(description="Discusses pricing or budget")
    contains_objection: bool = Field(description="Contains objections or concerns")
    contains_pain_point: bool = Field(description="Customer expresses pain points")

    # Evidence
    key_quotes: List[str] = Field(description="Important quotes from this chunk (max 3)")


class CriticalMoment(BaseModel):
    timestamp: str = Field(description="When this moment occurred")
    chunk_id: str = Field(description="Which chunk contains this moment")
    moment_type: str = Field(
        description="Type: objection/commitment/pricing_discussion/pain_point/product_pitch/emotional_shift")
    description: str = Field(description="What happened at this moment")
    quote: str = Field(description="Specific quote demonstrating this moment")
    impact: str = Field(description="Impact level: high/medium/low")


class CallTag(BaseModel):
    tag_category: str = Field(
        description="Category: sentiment/pain_point/product/next_step/relationship/upsell/stock/product_fit")
    tag_value: str = Field(description="The actual tag value")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    evidence: List[str] = Field(description="Chunk IDs that support this tag")
    reasoning: str = Field(description="Why this tag was applied")
    quotes: List[str] = Field(description="Supporting quotes")


class PreprocessingOutput(BaseModel):
    """Complete preprocessing output from single LLM call"""

    # Core chunks
    chunks: List[ChunkData] = Field(description="Semantically chunked segments")

    # Summaries at different levels
    section_summaries: Dict[str, str] = Field(description="Summaries for each phase/section")
    executive_summary: str = Field(description="2-3 paragraph overview of entire call")

    # Critical moments
    critical_moments: List[CriticalMoment] = Field(description="Key moments in the call")

    # Initial tags
    tags: List[CallTag] = Field(description="All identified tags with evidence")

    # Call metadata
    overall_sentiment: str = Field(
        description="Overall sentiment: very_positive/positive/neutral/negative/very_negative")
    relationship_quality: str = Field(description="Relationship quality: excellent/good/neutral/poor")
    call_outcome: str = Field(description="Outcome: closed/advancing/stalled/lost/unclear")
    next_steps_clarity: str = Field(description="Next steps clarity: crystal_clear/clear/vague/missing")

    # Products and opportunities
    products_mentioned: List[str] = Field(description="All products mentioned")
    products_pitched: List[str] = Field(description="Products actively pitched")
    upsell_opportunities: List[str] = Field(description="Upsell opportunities identified")
    out_of_stock_mentions: List[str] = Field(description="Products mentioned as out of stock")

    # Customer insights
    customer_pain_points: List[str] = Field(description="Identified customer pain points")
    product_fit_assessment: str = Field(description="Product fit: excellent/good/fair/poor/misaligned")
    product_fit_reasoning: str = Field(description="Why this product fit assessment")


# ============================================================================
# MAIN PREPROCESSING AGENT
# ============================================================================

class PureLLMPreprocessor:
    """
    Pure LLM approach for call transcript preprocessing.
    Single comprehensive LLM call does all analysis.
    """

    def __init__(
            self,
            llm: Optional[ChatGoogleGenerativeAI] = None,
            qdrant_client: Optional[QdrantClient] = None,
            qdrant_collection_name: str = "call_transcripts_pure_llm",
            embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                                 temperature=0.7,
                                                 google_api_key=os.getenv('GOOGLE_AI_API_KEY'))

        self.qdrant_client = qdrant_client or QdrantClient(":memory:")
        self.collection_name = qdrant_collection_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Setup Qdrant collection
        self._setup_qdrant_collection()

        # Setup parser
        self.parser = JsonOutputParser(pydantic_object=PreprocessingOutput)

        # Setup prompt
        self._setup_prompt()

    def _setup_qdrant_collection(self):
        """Setup Qdrant collection for vector storage"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"[PureLLMPreprocessor] Using existing collection: {self.collection_name}")
        except:
            vector_size = 384  # For all-MiniLM-L6-v2
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"[PureLLMPreprocessor] Created collection: {self.collection_name}")

    def _setup_prompt(self):
        """Setup comprehensive preprocessing prompt"""

        self.preprocessing_prompt = PromptTemplate(
            template="""You are an expert sales call analyst. You will analyze a complete call transcript and provide comprehensive preprocessing for downstream tagging agents.

                Your task is to perform ALL preprocessing in a single pass:
                
                1. **INTELLIGENT CHUNKING**: Create semantic chunks based on natural conversation boundaries
                   - Identify topic shifts, phase changes, speaker transitions
                   - Keep related discussions together (e.g., question + answer)
                   - Chunks should be 2-5 minutes of conversation
                   - Ensure chunks align with conversation phases
                
                2. **MULTI-LEVEL SUMMARIZATION**:
                   - Create summary for each chunk
                   - Create section summaries (group chunks by phase)
                   - Create executive summary of entire call
                
                3. **COMPREHENSIVE TAGGING**: Identify ALL of the following:
                   - **Sentiment**: Overall emotional tone and trajectory
                   - **Customer Pain Points**: Explicit and implicit problems mentioned
                   - **Products Pitched**: What solutions were offered
                   - **Next Steps**: Action items and commitments
                   - **Relationship Quality**: How well the conversation went
                   - **Upsell Mentions**: Any upsell opportunities
                   - **Out of Stock Mentions**: Products unavailable
                   - **Product Fit**: How well pitched products match customer needs
                
                4. **CRITICAL MOMENTS**: Flag key moments like:
                   - Customer objections
                   - Pricing discussions
                   - Commitments made
                   - Emotional shifts
                   - Product demonstrations
                
                5. **EVIDENCE COLLECTION**: For every tag and assessment, provide:
                   - Specific chunk IDs where evidence exists
                   - Direct quotes supporting the tag
                   - Reasoning for confidence level
                
                IMPORTANT GUIDELINES:
                - Base everything on actual transcript content
                - Provide specific evidence (quotes, timestamps, chunk references)
                - Use importance scores to highlight critical segments
                - Maintain context awareness throughout analysis
                - For product fit, compare pain points discussed with products pitched
                
                Call Transcript:
                {diarized_transcript}
                
                Additional Context (if provided):
                {transcript_text}
                
                {format_instructions}
                
                Provide your complete analysis as structured JSON.""",
            input_variables=["diarized_transcript", "transcript_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def _extract_chunks_from_output(
            self,
            preprocessing_output: PreprocessingOutput,
            diarized_transcript: str
    ) -> List[Dict]:
        """
        Extract actual chunk text from transcript based on LLM-identified boundaries.
        """
        chunks_with_text = []
        lines = diarized_transcript.strip().split('\n')

        for chunk_data in preprocessing_output.chunks:
            # Extract text for this chunk based on timestamps
            chunk_text_lines = []

            for line in lines:
                if not line.strip():
                    continue

                # Try to extract timestamp from line
                # Expected format: [MM:SS-MM:SS] or [HH:MM:SS-HH:MM:SS] Speaker: text
                if '[' in line and ']' in line:
                    timestamp_part = line[line.find('[') + 1:line.find(']')]

                    # Check if this line falls within chunk boundaries
                    # Simple heuristic: check if timestamp mentions overlap
                    if chunk_data.start_time in line or chunk_data.end_time in line:
                        chunk_text_lines.append(line)
                    else:
                        # More sophisticated: parse and compare times
                        # For now, we'll include lines near the boundaries
                        chunk_text_lines.append(line)

            # If we couldn't extract by timestamp, fall back to keyword matching
            if not chunk_text_lines:
                for line in lines:
                    # Check if line contains any of the chunk's keywords or topics
                    line_lower = line.lower()
                    if any(keyword.lower() in line_lower for keyword in chunk_data.keywords[:3]):
                        chunk_text_lines.append(line)

            chunk_text = '\n'.join(chunk_text_lines) if chunk_text_lines else "Content extracted from full transcript"

            chunks_with_text.append({
                'chunk_data': chunk_data,
                'text': chunk_text
            })

        return chunks_with_text

    def _store_in_qdrant(
            self,
            chunks_with_text: List[Dict],
            preprocessing_output: PreprocessingOutput,
            call_id: str
    ):
        """Store chunk embeddings and metadata in Qdrant"""
        points = []

        for chunk_item in chunks_with_text:
            chunk_data = chunk_item['chunk_data']
            chunk_text = chunk_item['text']

            # Create embedding
            embedding = self.embeddings.embed_query(chunk_text)

            # Prepare payload
            payload = {
                "call_id": call_id,
                "chunk_id": chunk_data.chunk_id,
                "text": chunk_text,
                "summary": chunk_data.summary,
                "start_time": chunk_data.start_time,
                "end_time": chunk_data.end_time,
                "speaker": chunk_data.speaker_primary,
                "phase": chunk_data.phase,
                "importance_score": chunk_data.importance_score,
                "emotional_tone": chunk_data.emotional_tone,
                "topics": chunk_data.topics,
                "keywords": chunk_data.keywords,
                "contains_question": chunk_data.contains_question,
                "contains_commitment": chunk_data.contains_commitment,
                "contains_product_mention": chunk_data.contains_product_mention,
                "contains_pricing": chunk_data.contains_pricing,
                "contains_objection": chunk_data.contains_objection,
                "contains_pain_point": chunk_data.contains_pain_point,
                "key_quotes": chunk_data.key_quotes
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"[PureLLMPreprocessor] ✓ Stored {len(points)} chunks in Qdrant")

    def process(
            self,
            transcript_text: str,
            diarized_transcript: str,
            call_id: Optional[str] = None
    ) -> Dict:
        """
        Main preprocessing method - single LLM call does everything.

        Args:
            transcript_text: Plain text transcript (optional, can be empty)
            diarized_transcript: Transcript with speaker labels and timestamps
            call_id: Unique identifier for the call

        Returns:
            Dictionary with preprocessing output and metadata
        """
        print("[PureLLMPreprocessor] Starting pure LLM preprocessing...")
        print(f"[PureLLMPreprocessor] Transcript length: {len(diarized_transcript)} chars")

        call_id = call_id or str(uuid.uuid4())

        # Single comprehensive LLM call
        print("[PureLLMPreprocessor] Making single comprehensive LLM call...")

        chain = self.preprocessing_prompt | self.llm | self.parser

        try:
            result = chain.invoke({
                "diarized_transcript": diarized_transcript,
                "transcript_text": transcript_text if transcript_text else "See diarized transcript above"
            })

            preprocessing_output = PreprocessingOutput(**result)

            print(f"[PureLLMPreprocessor] ✓ LLM analysis complete")
            print(f"[PureLLMPreprocessor]   - Chunks created: {len(preprocessing_output.chunks)}")
            print(f"[PureLLMPreprocessor]   - Critical moments: {len(preprocessing_output.critical_moments)}")
            print(f"[PureLLMPreprocessor]   - Tags identified: {len(preprocessing_output.tags)}")
            print(f"[PureLLMPreprocessor]   - Overall sentiment: {preprocessing_output.overall_sentiment}")
            print(f"[PureLLMPreprocessor]   - Product fit: {preprocessing_output.product_fit_assessment}")

        except Exception as e:
            print(f"[PureLLMPreprocessor] ✗ Error in LLM processing: {str(e)}")
            raise

        # Extract actual chunk texts
        print("[PureLLMPreprocessor] Extracting chunk texts...")
        chunks_with_text = self._extract_chunks_from_output(
            preprocessing_output,
            diarized_transcript
        )

        # Store in Qdrant
        print("[PureLLMPreprocessor] Storing in Qdrant...")
        self._store_in_qdrant(chunks_with_text, preprocessing_output, call_id)

        # Prepare final output
        final_output = {
            "call_id": call_id,
            "preprocessing_output": preprocessing_output,
            "chunks_with_text": chunks_with_text,
            "metadata": {
                "total_chunks": len(preprocessing_output.chunks),
                "high_importance_chunks": sum(
                    1 for c in preprocessing_output.chunks if c.importance_score > 0.7
                ),
                "overall_sentiment": preprocessing_output.overall_sentiment,
                "relationship_quality": preprocessing_output.relationship_quality,
                "product_fit": preprocessing_output.product_fit_assessment,
                "products_pitched": preprocessing_output.products_pitched,
                "customer_pain_points": preprocessing_output.customer_pain_points,
                "next_steps_clarity": preprocessing_output.next_steps_clarity,
                "processed_at": datetime.now().isoformat()
            }
        }

        print(f"[PureLLMPreprocessor] ✓✓✓ Preprocessing complete for call {call_id}")
        self._print_summary(preprocessing_output)

        return final_output

    def _print_summary(self, output: PreprocessingOutput):
        """Print a nice summary of preprocessing results"""
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)

        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"  Sentiment: {output.overall_sentiment}")
        print(f"  Relationship: {output.relationship_quality}")
        print(f"  Outcome: {output.call_outcome}")
        print(f"  Product Fit: {output.product_fit_assessment}")
        print(f"  Next Steps: {output.next_steps_clarity}")

        print(f"\n📦 PRODUCTS:")
        print(f"  Mentioned: {', '.join(output.products_mentioned) if output.products_mentioned else 'None'}")
        print(f"  Pitched: {', '.join(output.products_pitched) if output.products_pitched else 'None'}")

        print(f"\n💡 CUSTOMER INSIGHTS:")
        print(f"  Pain Points: {len(output.customer_pain_points)}")
        for i, pain_point in enumerate(output.customer_pain_points[:3], 1):
            print(f"    {i}. {pain_point}")

        print(f"\n⚡ CRITICAL MOMENTS: {len(output.critical_moments)}")
        for moment in output.critical_moments[:5]:
            print(f"  [{moment.timestamp}] {moment.moment_type}: {moment.description}")

        print(f"\n🏷️  TAGS IDENTIFIED: {len(output.tags)}")
        tags_by_category = {}
        for tag in output.tags:
            if tag.tag_category not in tags_by_category:
                tags_by_category[tag.tag_category] = []
            tags_by_category[tag.tag_category].append(tag.tag_value)

        for category, values in tags_by_category.items():
            print(f"  {category}: {', '.join(values[:3])}")

        print("\n" + "=" * 80 + "\n")

    def search_chunks(
            self,
            call_id: str,
            query: str,
            top_k: int = 5,
            min_importance: float = 0.0,
            filter_phase: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search for relevant chunks.

        Args:
            call_id: The call to search within
            query: Search query
            top_k: Number of results
            min_importance: Minimum importance score filter
            filter_phase: Filter by phase (intro/discovery/pitch/etc)

        Returns:
            List of matching chunks with metadata
        """
        query_embedding = self.embeddings.embed_query(query)

        # Build filter
        must_conditions = [
            {"key": "call_id", "match": {"value": call_id}},
            {"key": "importance_score", "range": {"gte": min_importance}}
        ]

        if filter_phase:
            must_conditions.append(
                {"key": "phase", "match": {"value": filter_phase}}
            )

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter={"must": must_conditions},
            limit=top_k
        )

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "text": r.payload["text"],
                "summary": r.payload["summary"],
                "score": r.score,
                "timestamp": f"{r.payload['start_time']}-{r.payload['end_time']}",
                "importance": r.payload["importance_score"],
                "phase": r.payload["phase"],
                "topics": r.payload["topics"],
                "key_quotes": r.payload["key_quotes"]
            }
            for r in results
        ]

    def get_chunks_by_criteria(
            self,
            call_id: str,
            has_questions: Optional[bool] = None,
            has_objections: Optional[bool] = None,
            has_pricing: Optional[bool] = None,
            min_importance: float = 0.0,
            phase: Optional[str] = None
    ) -> List[Dict]:
        """
        Get chunks matching specific criteria (useful for specialized agents).

        Args:
            call_id: The call to search
            has_questions: Filter chunks with customer questions
            has_objections: Filter chunks with objections
            has_pricing: Filter chunks discussing pricing
            min_importance: Minimum importance score
            phase: Filter by conversation phase

        Returns:
            List of matching chunks
        """
        must_conditions = [
            {"key": "call_id", "match": {"value": call_id}},
            {"key": "importance_score", "range": {"gte": min_importance}}
        ]

        if has_questions is not None:
            must_conditions.append(
                {"key": "contains_question", "match": {"value": has_questions}}
            )

        if has_objections is not None:
            must_conditions.append(
                {"key": "contains_objection", "match": {"value": has_objections}}
            )

        if has_pricing is not None:
            must_conditions.append(
                {"key": "contains_pricing", "match": {"value": has_pricing}}
            )

        if phase:
            must_conditions.append(
                {"key": "phase", "match": {"value": phase}}
            )

        # Scroll through all matching results
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter={"must": must_conditions},
            limit=100
        )

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "text": r.payload["text"],
                "summary": r.payload["summary"],
                "timestamp": f"{r.payload['start_time']}-{r.payload['end_time']}",
                "importance": r.payload["importance_score"],
                "phase": r.payload["phase"],
                "topics": r.payload["topics"],
                "key_quotes": r.payload["key_quotes"]
            }
            for r in results
        ]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def get_transacript(call_id):
    doc = get_document_by_id(
        db_name='call_iq',
        collection_name='audios',
        doc_id=int(call_id),
    )
    print(format_diarized_transcript(data=doc))


def format_diarized_transcript(data, speaker_map=None):
    """
    Convert diarized transcript into readable conversation format.

    Args:
        data (dict): Input JSON containing metadata -> diarized_transcript -> entries
        speaker_map (dict): Optional mapping like {"SPEAKER_00": "Speaker 1"}

    Returns:
        str: Formatted transcript
    """

    def sec_to_mmss(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    entries = data["metadata"]["diarized_transcript"]["entries"]

    if speaker_map is None:
        speaker_map = {}

    output = []
    current_speaker = None
    current_text = []
    start_time = None
    end_time = None

    for entry in entries:
        speaker = entry["speaker_id"]
        text = entry["transcript"].strip()
        s_time = entry["start_time_seconds"]
        e_time = entry["end_time_seconds"]

        speaker_label = speaker_map.get(speaker, speaker.replace("SPEAKER_", "Speaker "))

        if speaker != current_speaker:
            if current_speaker is not None:
                output.append(
                    f"[{sec_to_mmss(start_time)}-{sec_to_mmss(end_time)}] "
                    f"{speaker_map.get(current_speaker, current_speaker.replace('SPEAKER_', 'Speaker '))}: "
                    f"{' '.join(current_text)}"
                )

            current_speaker = speaker
            current_text = [text]
            start_time = s_time
            end_time = e_time
        else:
            current_text.append(text)
            end_time = e_time

    # Flush last speaker
    if current_speaker is not None:
        output.append(
            f"[{sec_to_mmss(start_time)}-{sec_to_mmss(end_time)}] "
            f"{speaker_map.get(current_speaker, current_speaker.replace('SPEAKER_', 'Speaker '))}: "
            f"{' '.join(current_text)}"
        )

    return "\n\n".join(output)



if __name__ == "__main__":
    get_transacript(36797)
#
#     # Sample diarized transcript
#     sample_transcript = """"""
#
#     # Initialize preprocessor
#     preprocessor = PureLLMPreprocessor(
#         llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,
#                                  google_api_key=os.getenv('GOOGLE_AI_API_KEY'))
#     )
#
#     # Process the call
#     result = preprocessor.process(
#         transcript_text="",
#         diarized_transcript=sample_transcript,
#         call_id="demo_call_001"
#     )
#
#     print("\n\n" + "=" * 80)
#     print("EXAMPLE QUERIES")
#     print("=" * 80)
#
#     # Example: Search for pricing discussions
#     print("\n1. Searching for 'pricing discussions':")
#     pricing_chunks = preprocessor.search_chunks(
#         call_id="demo_call_001",
#         query="pricing and budget discussions",
#         top_k=3
#     )
#     for chunk in pricing_chunks:
#         print(f"\n  Chunk: {chunk['chunk_id']} (Score: {chunk['score']:.3f})")
#         print(f"  Time: {chunk['timestamp']}")
#         print(f"  Summary: {chunk['summary'][:100]}...")
#
#     # Example: Get chunks with objections
#     print("\n2. Getting chunks with objections:")
#     objection_chunks = preprocessor.get_chunks_by_criteria(
#         call_id="demo_call_001",
#         has_objections=True
#     )
#     for chunk in objection_chunks:
#         print(f"\n  Chunk: {chunk['chunk_id']}")
#         print(f"  Quote: {chunk['key_quotes'][0] if chunk['key_quotes'] else 'N/A'}")
#
#     # Example: Get discovery phase chunks
#     print("\n3. Getting discovery phase chunks:")
#     discovery_chunks = preprocessor.get_chunks_by_criteria(
#         call_id="demo_call_001",
#         phase="discovery",
#         min_importance=0.5
#     )
#     for chunk in discovery_chunks:
#         print(f"\n  Chunk: {chunk['chunk_id']}")
#         print(f"  Topics: {', '.join(chunk['topics'])}")