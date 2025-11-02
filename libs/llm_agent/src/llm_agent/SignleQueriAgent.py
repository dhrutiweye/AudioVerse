from typing import Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .BaseLLMAgent import BaseLLMAgent as LLMEnabledAgent


class EvidenceItem(BaseModel):
    speaker: Optional[str] = Field(None, description="Speaker label if available (e.g., Agent, S1)")
    start_ms: Optional[int] = Field(None, description="Start time in milliseconds if available")
    end_ms: Optional[int] = Field(None, description="End time in milliseconds if available")
    text: str = Field(description="Exact supporting snippet from transcript")
    why_it_supports: str = Field(description="Brief reason this snippet supports the match")


class Scores(BaseModel):
    semantic_similarity: float = Field(description="0.00–1.00")
    context_alignment: float = Field(description="0.00–1.00")
    answerability: float = Field(description="0.00–1.00")
    speaker_alignment: float = Field(description="0.00–1.00 (use 0.50 if N/A)")
    temporal_alignment: float = Field(description="0.00–1.00 (use 0.50 if N/A)")


class SignleQueriAgent(LLMEnabledAgent):
    class RelevanceOutput(BaseModel):
        is_related: bool = Field(description="True if transcript is related to the query")
        verdict: str = Field(description="one of: strong | partial | no")
        overall_score: float = Field(description="0.00–1.00")
        scores: Scores
        contradiction: float = Field(description="0.00–1.00 where 1.00 means explicit contradiction")
        notes: Optional[str] = Field(default=None, description="Short reasoning or caveats")

    # ===== Parser & Prompt =====
    RelevanceOutput.model_rebuild()
    parser = JsonOutputParser(pydantic_object=RelevanceOutput)

    prompt = PromptTemplate(
        template=(
            # --- SYSTEM ROLE INSTRUCTIONS ---
            "You are “Transcript Relevance Rater”. Given a user query and a transcript "
            "(diarized or plain), decide how related the transcript is and output STRICT JSON only.\n\n"
            "Rules:\n"
            "- Work multilingually and with code-mixing (e.g., Hinglish). Normalize case and handle transliteration.\n"
            "- Use ONLY the provided transcript text; do NOT rely on outside knowledge.\n"
            "- Prefer exact evidence: quote the 1–3 most relevant sentences/turns; include speaker/timestamps if available.\n"
            "- Be conservative: if evidence is weak or generic, lower the scores.\n\n"
            "Scoring (0.00–1.00 floats, 2 decimals expected):\n"
            "- semantic_similarity: topical closeness ignoring exact wording.\n"
            "- context_alignment: does the transcript cover the specific context implied by the query "
            "(entities, constraints, actions, outcomes)?\n"
            "- answerability: could the transcript directly answer the query (fully/partially)?\n"
            "- speaker_alignment: if the query implies a speaker/role, do the matched turns support it? "
            "If not implied or N/A, use 0.50.\n"
            "- temporal_alignment: if ordering/timestamps matter, does the sequence fit? If N/A, use 0.50.\n"
            "- contradiction: 0.00–1.00 where 1.00 means the transcript clearly contradicts the query’s claim.\n\n"
            "Overall score:\n"
            "base = 0.35*semantic_similarity + 0.35*context_alignment + 0.15*answerability "
            "+ 0.10*speaker_alignment + 0.05*temporal_alignment\n"
            "overall_score = round(base * (1 - 0.7*contradiction), 2)\n\n"
            "Verdict:\n"
            "- strong if overall_score ≥ 0.75\n"
            "- partial if 0.40 ≤ overall_score < 0.75\n"
            "- no if overall_score < 0.40\n"
            "- Set is_related = (overall_score ≥ 0.40)\n\n"
            # "Evidence selection:\n"
            # "- Choose the most on-point snippets (max 3). Prefer turns that directly mention key terms or resolve the intent.\n\n"
            "OUTPUT: {format_instructions}\n\n"
            # --- USER INPUTS ---
            "Query:\n{query_string}\n\n"
            "Plain transcript (optional):\n{transcript_text}\n\n"
            "Diarized transcript (preferred if provided):\n{diarized_transcript}\n\n"
            "Instructions:\n"
            "1) If diarized transcript is provided, use it; otherwise fall back to the plain transcript.\n"
            "2) Apply the scoring rubric and formula EXACTLY.\n"
            "3) Return VALID JSON ONLY (no markdown, no extra text)."
        ),
        input_variables=["query_string", "transcript_text", "diarized_transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    def __init__(self, llm: None, name: str = "TranscriptRelevanceAgent", ):
        super().__init__(name=name, prompt_template=self.prompt, output_parser=self.parser, llm=llm)

    def handle(self, query_string: str, transcript_text: str = "", diarized_transcript: str = ""):
        print(f"[{self.name}] scoring transcript relevance...")
        return self.generate_responseV2({
            "query_string": query_string,
            "transcript_text": transcript_text,
            "diarized_transcript": diarized_transcript
        })
