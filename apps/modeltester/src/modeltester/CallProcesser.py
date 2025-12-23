from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings
import uuid
import re
from datetime import datetime


# Pydantic Models for Structured Outputs
class ChunkSummary(BaseModel):
    summary: str = Field(description="Brief 2-3 sentence summary of the chunk")
    key_topics: List[str] = Field(description="List of main topics discussed")
    speaker_focus: str = Field(description="Primary speaker in this chunk")
    emotional_tone: str = Field(description="Emotional tone: positive/negative/neutral/mixed")


class ImportanceScore(BaseModel):
    score: float = Field(description="Importance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the score")
    contains_question: bool = Field(description="Whether chunk contains questions")
    contains_commitment: bool = Field(description="Whether chunk contains commitments or action items")
    contains_product_mention: bool = Field(description="Whether chunk mentions products")
    contains_pricing: bool = Field(description="Whether chunk discusses pricing or budget")
    contains_objection: bool = Field(description="Whether chunk contains objections or concerns")


class CallSummary(BaseModel):
    executive_summary: str = Field(description="2-3 paragraph overview of entire call")
    key_moments: List[str] = Field(description="List of critical moments with timestamps")
    overall_sentiment: str = Field(description="Overall call sentiment")
    call_phase: str = Field(description="Current phase: intro/discovery/pitch/negotiation/closing")


# Data Classes
@dataclass
class ProcessedChunk:
    chunk_id: str
    text: str
    start_time: str
    end_time: str
    speaker: str
    summary: str
    key_topics: List[str]
    emotional_tone: str
    importance_score: float
    importance_reasoning: str
    keywords: List[str]
    embedding: List[float]
    metadata: Dict


@dataclass
class PreprocessedCall:
    call_id: str
    chunks: List[ProcessedChunk]
    chunk_summaries: List[str]
    section_summaries: Dict[str, str]
    full_call_summary: str
    metadata: Dict
    keyword_index: Dict[str, List[str]]  # keyword -> [chunk_ids]


class TranscriptPreprocessor:
    """
    Preprocessing agent for call transcripts.
    Handles chunking, summarization, embedding, keyword extraction, and importance scoring.
    """

    def __init__(
            self,
            llm: Optional[ChatAnthropic] = None,
            qdrant_client: Optional[QdrantClient] = None,
            qdrant_collection_name: str = "call_transcripts",
            embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.llm = llm or ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        self.qdrant_client = qdrant_client or QdrantClient(":memory:")
        self.collection_name = qdrant_collection_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize text splitter for semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # ~3-5 minutes of conversation
            chunk_overlap=200,  # Overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", "Speaker:", ". ", " ", ""]
        )

        # Setup Qdrant collection
        self._setup_qdrant_collection()

        # Initialize parsers
        self.chunk_summary_parser = JsonOutputParser(pydantic_object=ChunkSummary)
        self.importance_parser = JsonOutputParser(pydantic_object=ImportanceScore)
        self.call_summary_parser = JsonOutputParser(pydantic_object=CallSummary)

        # Setup prompts
        self._setup_prompts()

    def _setup_qdrant_collection(self):
        """Setup Qdrant collection for vector storage"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"[Preprocessor] Using existing Qdrant collection: {self.collection_name}")
        except:
            vector_size = 384  # For all-MiniLM-L6-v2
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"[Preprocessor] Created new Qdrant collection: {self.collection_name}")

    def _setup_prompts(self):
        """Setup prompt templates for different preprocessing tasks"""

        # Chunk summarization prompt
        self.chunk_summary_prompt = PromptTemplate(
            template="""You are analyzing a segment of a sales call transcript.
                Chunk text:
                {chunk_text}
                
                Timestamp: {timestamp}
                Speaker: {speaker}
                
                Provide a structured analysis of this chunk.
                
                {format_instructions}""",
            input_variables=["chunk_text", "timestamp", "speaker"],
            partial_variables={"format_instructions": self.chunk_summary_parser.get_format_instructions()}
        )

        # Importance scoring prompt
        self.importance_prompt = PromptTemplate(
            template="""You are evaluating the importance of a call transcript segment for downstream analysis.

                Consider these factors for high importance:
                - Contains customer questions or concerns
                - Mentions products, pricing, or budget
                - Contains commitments or action items
                - Shows emotional moments (positive or negative)
                - Contains objections or hesitations
                - Discusses next steps or timeline
                
                Chunk text:
                {chunk_text}
                
                Chunk summary: {chunk_summary}
                
                Assign an importance score from 0.0 (trivial small talk) to 1.0 (critical moment).
                
                {format_instructions}""",
            input_variables=["chunk_text", "chunk_summary"],
            partial_variables={"format_instructions": self.importance_parser.get_format_instructions()}
        )

        # Call summary prompt
        self.call_summary_prompt = PromptTemplate(
            template="""You are creating a comprehensive summary of a sales call.
                Full transcript:
                {transcript}
                
                Chunk summaries:
                {chunk_summaries}
                
                Create an executive summary that captures:
                1. The main purpose and outcome of the call
                2. Key discussion points and customer concerns
                3. Products/solutions discussed
                4. Critical moments or turning points
                5. Overall sentiment and relationship quality
                
                {format_instructions}""",
            input_variables=["transcript", "chunk_summaries"],
            partial_variables={"format_instructions": self.call_summary_parser.get_format_instructions()}
        )

    def _parse_diarized_transcript(self, diarized_transcript: str) -> List[Dict]:
        """
        Parse diarized transcript into structured segments.
        Expected format: "Speaker: text\nSpeaker: text\n..."
        """
        segments = []
        lines = diarized_transcript.strip().split("\n")

        current_speaker = None
        current_text = []

        for line in lines:
            if not line.strip():
                continue

            # Check if line starts with speaker label
            if ":" in line and line.split(":")[0].strip() in ["Speaker", "Agent", "Customer", "Rep"]:
                # Save previous segment
                if current_speaker and current_text:
                    segments.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip()
                    })

                # Start new segment
                parts = line.split(":", 1)
                current_speaker = parts[0].strip()
                current_text = [parts[1].strip()] if len(parts) > 1 else []
            else:
                current_text.append(line.strip())

        # Add last segment
        if current_speaker and current_text:
            segments.append({
                "speaker": current_speaker,
                "text": " ".join(current_text).strip()
            })

        return segments

    def _chunk_transcript(
            self,
            transcript_text: str,
            diarized_transcript: str
    ) -> List[Dict]:
        """
        Create semantic chunks from transcript with speaker awareness.
        """
        segments = self._parse_diarized_transcript(diarized_transcript)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_speakers = set()

        for i, segment in enumerate(segments):
            segment_text = f"{segment['speaker']}: {segment['text']}"
            segment_length = len(segment_text)

            # Check if adding this segment exceeds chunk size
            if current_length + segment_length > 1500 and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "speakers": list(chunk_speakers),
                    "primary_speaker": max(chunk_speakers,
                                           key=lambda s: sum(1 for c in current_chunk if c.startswith(s))),
                    "segment_indices": list(range(i - len(current_chunk), i))
                })

                # Start new chunk with overlap (last 20% of previous chunk)
                overlap_size = max(1, len(current_chunk) // 5)
                current_chunk = current_chunk[-overlap_size:]
                current_length = sum(len(c) for c in current_chunk)
                chunk_speakers = set(c.split(":")[0] for c in current_chunk if ":" in c)

            current_chunk.append(segment_text)
            current_length += segment_length
            chunk_speakers.add(segment['speaker'])

        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "speakers": list(chunk_speakers),
                "primary_speaker": max(chunk_speakers, key=lambda s: sum(1 for c in current_chunk if c.startswith(s))),
                "segment_indices": list(range(len(segments) - len(current_chunk), len(segments)))
            })

        return chunks

    def _estimate_timestamp(self, chunk_index: int, total_chunks: int, call_duration_minutes: int = 60) -> Tuple[
        str, str]:
        """
        Estimate timestamp range for a chunk based on its position.
        """
        chunk_duration = call_duration_minutes / total_chunks
        start_minutes = chunk_index * chunk_duration
        end_minutes = (chunk_index + 1) * chunk_duration

        def format_time(minutes: float) -> str:
            m = int(minutes)
            s = int((minutes - m) * 60)
            return f"{m:02d}:{s:02d}"

        return format_time(start_minutes), format_time(end_minutes)

    def _summarize_chunk(self, chunk: Dict, timestamp: str) -> ChunkSummary:
        """Generate summary for a single chunk"""
        chain = self.chunk_summary_prompt | self.llm | self.chunk_summary_parser

        result = chain.invoke({
            "chunk_text": chunk["text"],
            "timestamp": timestamp,
            "speaker": chunk["primary_speaker"]
        })

        return ChunkSummary(**result)

    def _score_importance(self, chunk: Dict, summary: ChunkSummary) -> ImportanceScore:
        """Calculate importance score for a chunk"""
        chain = self.importance_prompt | self.llm | self.importance_parser

        result = chain.invoke({
            "chunk_text": chunk["text"],
            "chunk_summary": summary.summary
        })

        return ImportanceScore(**result)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords using simple heuristics.
        Can be enhanced with NER or keyword extraction models.
        """
        # Common sales/product keywords
        important_patterns = [
            r'\b(product|solution|feature|price|cost|budget|pricing)\w*\b',
            r'\b(concern|issue|problem|pain point|challenge)\w*\b',
            r'\b(discount|deal|offer|promotion)\w*\b',
            r'\b(competitor|alternative|current solution)\w*\b',
            r'\b(timeline|deadline|urgency|asap)\w*\b',
            r'\b(decision|approve|commit|contract|agreement)\w*\b'
        ]

        keywords = set()
        text_lower = text.lower()

        for pattern in important_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.update(matches)

        return list(keywords)

    def _generate_call_summary(self, transcript: str, chunk_summaries: List[str]) -> CallSummary:
        """Generate overall call summary"""
        # Limit transcript length for summary
        transcript_excerpt = transcript[:8000] if len(transcript) > 8000 else transcript
        summaries_text = "\n".join([f"- {s}" for s in chunk_summaries[:20]])

        chain = self.call_summary_prompt | self.llm | self.call_summary_parser

        result = chain.invoke({
            "transcript": transcript_excerpt,
            "chunk_summaries": summaries_text
        })

        return CallSummary(**result)

    def _build_keyword_index(self, chunks: List[ProcessedChunk]) -> Dict[str, List[str]]:
        """Build inverted index: keyword -> list of chunk_ids"""
        keyword_index = {}

        for chunk in chunks:
            for keyword in chunk.keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(chunk.chunk_id)

        return keyword_index

    def _store_in_qdrant(self, chunks: List[ProcessedChunk], call_id: str):
        """Store chunk embeddings in Qdrant"""
        points = []

        for chunk in chunks:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk.embedding,
                    payload={
                        "call_id": call_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "summary": chunk.summary,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "speaker": chunk.speaker,
                        "importance_score": chunk.importance_score,
                        "keywords": chunk.keywords,
                        "topics": chunk.key_topics
                    }
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"[Preprocessor] Stored {len(points)} chunks in Qdrant for call {call_id}")

    def process(
            self,
            transcript_text: str,
            diarized_transcript: str,
            call_id: Optional[str] = None,
            call_duration_minutes: int = 60
    ) -> PreprocessedCall:
        """
        Main preprocessing pipeline.

        Args:
            transcript_text: Full transcript text
            diarized_transcript: Transcript with speaker labels
            call_id: Unique identifier for the call
            call_duration_minutes: Estimated call duration for timestamp calculation

        Returns:
            PreprocessedCall object with all processed data
        """
        print("[Preprocessor] Starting preprocessing pipeline...")

        call_id = call_id or str(uuid.uuid4())

        # Step 1: Chunk the transcript
        print("[Preprocessor] Step 1/5: Chunking transcript...")
        raw_chunks = self._chunk_transcript(transcript_text, diarized_transcript)
        print(f"[Preprocessor] Created {len(raw_chunks)} chunks")

        # Step 2: Process each chunk
        print("[Preprocessor] Step 2/5: Summarizing and scoring chunks...")
        processed_chunks = []

        for i, chunk in enumerate(raw_chunks):
            start_time, end_time = self._estimate_timestamp(i, len(raw_chunks), call_duration_minutes)
            timestamp = f"{start_time}-{end_time}"

            # Summarize chunk
            summary_obj = self._summarize_chunk(chunk, timestamp)

            # Score importance
            importance_obj = self._score_importance(chunk, summary_obj)

            # Extract keywords
            keywords = self._extract_keywords(chunk["text"])

            # Create embedding
            embedding = self.embeddings.embed_query(chunk["text"])

            processed_chunk = ProcessedChunk(
                chunk_id=f"chunk_{i}",
                text=chunk["text"],
                start_time=start_time,
                end_time=end_time,
                speaker=chunk["primary_speaker"],
                summary=summary_obj.summary,
                key_topics=summary_obj.key_topics,
                emotional_tone=summary_obj.emotional_tone,
                importance_score=importance_obj.score,
                importance_reasoning=importance_obj.reasoning,
                keywords=keywords,
                embedding=embedding,
                metadata={
                    "speakers": chunk["speakers"],
                    "contains_question": importance_obj.contains_question,
                    "contains_commitment": importance_obj.contains_commitment,
                    "contains_product_mention": importance_obj.contains_product_mention,
                    "contains_pricing": importance_obj.contains_pricing,
                    "contains_objection": importance_obj.contains_objection
                }
            )

            processed_chunks.append(processed_chunk)
            print(f"[Preprocessor] Processed chunk {i + 1}/{len(raw_chunks)} (importance: {importance_obj.score:.2f})")

        # Step 3: Generate summaries at different levels
        print("[Preprocessor] Step 3/5: Generating multi-level summaries...")

        chunk_summaries = [c.summary for c in processed_chunks]

        # Section summaries (group every 3-4 chunks)
        section_summaries = {}
        section_size = 4
        for i in range(0, len(processed_chunks), section_size):
            section_chunks = processed_chunks[i:i + section_size]
            section_text = "\n\n".join([c.summary for c in section_chunks])
            section_name = f"section_{i // section_size}"
            section_summaries[section_name] = section_text

        # Full call summary
        call_summary_obj = self._generate_call_summary(transcript_text, chunk_summaries)

        # Step 4: Build keyword index
        print("[Preprocessor] Step 4/5: Building keyword index...")
        keyword_index = self._build_keyword_index(processed_chunks)

        # Step 5: Store in Qdrant
        print("[Preprocessor] Step 5/5: Storing embeddings in Qdrant...")
        self._store_in_qdrant(processed_chunks, call_id)

        # Create final preprocessed call object
        preprocessed_call = PreprocessedCall(
            call_id=call_id,
            chunks=processed_chunks,
            chunk_summaries=chunk_summaries,
            section_summaries=section_summaries,
            full_call_summary=call_summary_obj.executive_summary,
            metadata={
                "total_chunks": len(processed_chunks),
                "call_duration_minutes": call_duration_minutes,
                "overall_sentiment": call_summary_obj.overall_sentiment,
                "call_phase": call_summary_obj.call_phase,
                "key_moments": call_summary_obj.key_moments,
                "processed_at": datetime.now().isoformat()
            },
            keyword_index=keyword_index
        )

        print(f"[Preprocessor] ✓ Preprocessing complete for call {call_id}")
        print(f"[Preprocessor] Total chunks: {len(processed_chunks)}")
        print(f"[Preprocessor] High importance chunks: {sum(1 for c in processed_chunks if c.importance_score > 0.7)}")

        return preprocessed_call

    def search_chunks(
            self,
            call_id: str,
            query: str,
            top_k: int = 5,
            min_importance: float = 0.0
    ) -> List[ProcessedChunk]:
        """
        Semantic search for relevant chunks within a call.

        Args:
            call_id: The call to search within
            query: Search query
            top_k: Number of results to return
            min_importance: Minimum importance score filter

        Returns:
            List of relevant ProcessedChunk objects
        """
        query_embedding = self.embeddings.embed_query(query)

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter={
                "must": [
                    {"key": "call_id", "match": {"value": call_id}},
                    {"key": "importance_score", "range": {"gte": min_importance}}
                ]
            },
            limit=top_k
        )

        # Convert back to ProcessedChunk objects
        chunks = []
        for result in results:
            payload = result.payload
            chunk = ProcessedChunk(
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                start_time=payload["start_time"],
                end_time=payload["end_time"],
                speaker=payload["speaker"],
                summary=payload["summary"],
                key_topics=payload["topics"],
                emotional_tone="",  # Not stored in payload
                importance_score=payload["importance_score"],
                importance_reasoning="",  # Not stored in payload
                keywords=payload["keywords"],
                embedding=[],  # Not needed for retrieval
                metadata={}
            )
            chunks.append(chunk)

        return chunks