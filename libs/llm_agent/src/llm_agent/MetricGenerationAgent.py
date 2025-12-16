
from typing import Optional, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from .BaseLLMAgent import BaseLLMAgent as LLMEnabledAgent


class MetricGenerationAgent(LLMEnabledAgent):
    parser = StrOutputParser()
    prompt = PromptTemplate(
        # template=(
        #     # --- SYSTEM ROLE INSTRUCTIONS ---
        #     """
        #         Given the following metric definition:
        #         {metric_json}
        #
        #         Generate a concise instruction prompt that:
        #         1. Explains exactly what must be extracted from a call transcript based on the metric’s definition.
        #         2. Converts the metric definition into clear detection rules, listing what counts as evidence.
        #         3. Defines the task steps needed to compute the metric_value based strictly on the transcript.
        #         4. Specifies evidence requirements (short, exact transcript quotes; max 3 snippets).
        #         5. Mentions diarized transcript preference.
        #         6. Ends with this final instruction block:
        #
        #             Instructions:
        #                 1) If diarized transcript is provided, use it; otherwise fall back to the plain transcript.
        #                 2) Return VALID JSON ONLY (no markdown, no extra text).
        #                 3) OUTPUT: {{format_instructions}}
        #                 Plain transcript: {{transcript_text}}
        #                 Diarized transcript (preferred if provided):{{diarized_transcript}}
        #
        #         Your output should follow the style and tightness of this example:
        #         "Extract all information in the transcript related to the metric <metric_name>.
        #          Definition: <rewrite the metric definition in simple detection terms>.
        #          Task:
        #            1. …
        #            2. …
        #            3. …
        #          Evidence selection:
        #            - …"
        #         ONLY generate the instruction prompt. Do NOT generate any metric output.
        #     """),
        # template = ("""You are a Metric-Prompt Generator.
        #
        # Your job is to convert ANY metric JSON into a SINGLE-CALL evaluation prompt that can be attached to call transcripts.
        #
        # ### INPUT METRIC JSON:
        # {metric_json}
        #
        # ### YOUR GOAL
        # Rewrite this metric definition into a clear instruction prompt that will evaluate **one call at a time**, even if the metric is originally computed across multiple calls.
        #
        # ### RULES FOR CONVERSION
        #
        # 1. **If the metric is a percentage, ratio, or aggregation metric:**
        #    - Convert it into a *per-call binary detection metric*:
        #         - metric_value = true  → evidence of the event in this call
        #         - metric_value = false → call is eligible but event did NOT occur
        #         - metric_value = null  → call is not eligible or evidence missing
        #    - Add a line:
        #      “Do NOT compute any percentage at the call level. Only return whether the event occurred.”
        #
        # 2. **If the metric is naturally per-call (e.g., adherence, presence/absence, quality checks):**
        #    - Keep it as a boolean, int, float, or text metric as appropriate.
        #
        # 3. **You MUST generate detection rules** by translating the metric definition into:
        #    - What counts as an “event”
        #    - What makes a call “eligible”
        #    - What transcript phrases or patterns indicate the metric
        #
        # 4. **Evidence Requirements**
        #    - Maximum 3 transcript quotes
        #    - Must be short, exact text from transcript
        #    - Must directly support the metric_value
        #
        # 5. **Prompt Structure to Generate**
        #    Your output MUST follow this format:
        #    "Extract all information in the transcript related to the metric <metric_name>.
        #      Definition: <rewrite the metric definition in simple detection terms>.
        #      Task:
        #         1. Identify whether this call is eligible for evaluation. Clearly define eligibility based on the metric.
        #         2. Detect whether the event defined by the metric occurred in this call.
        #         3. Output metric_value:
        #         - true = event occurred
        #         - false = call eligible but event did not occur
        #         - null = call not eligible or insufficient evidence
        #         4. Use ONLY explicit transcript evidence.
        #     Evidence selection:
        #     - Choose up to 3 exact transcript quotes.
        #     - Select only the most relevant lines proving whether the event occurred.
        #
        #     Instructions:
        #     1) If diarized transcript is provided, use it; otherwise fall back to the plain transcript.
        #     2) Return VALID JSON ONLY (no markdown, no extra text).
        # OUTPUT: {{format_instructions}}
        # INPUTE:
        #     Plain transcript: {{transcript_text}}
        #     Diarized transcript (preferred if provided): {{diarized_transcript}}
        # ### FINAL REQUIREMENT
        # Generate ONLY the instruction prompt.
        # Do NOT output any metric results.
        # Do NOT explain your reasoning.
        # Do NOT generate any metric output
        # """),
        template="""
        You are a Metric-Prompt Generator.
        
        IMPORTANT FORMATTING RULE:
        When producing curly braces for JSON examples inside the instruction prompt, ALWAYS output them literally as '{{{{' and '}}}}'. 
        Do NOT collapse them into '{{}}' and do NOT interpret them as JSON. 
        Treat '{{{{' and '}}}}' as plain text characters. 
        Never auto-correct or simplify them.
        
        Your job is to convert ANY metric JSON into a SINGLE-CALL evaluation prompt that produces output in the strict schema:
        
          {{"metric_name": "<metric_name>",
          "data_type": "<list | int | float | percentage | boolean | text | null>",
          "metric_value": <value or null>,
          "evidence": "<short reasoning with transcript quotes>"}}
        
        ### INPUT METRIC JSON:
        {metric_json}
        
        ### CONVERSION RULES
        
        1. If the metric is percentage-based, ratio-based, or aggregated across many calls:
             - Convert it into a per-call boolean metric:
                  - true  = event occurred in this call
                  - false = event did NOT occur in this call (but call is eligible)
                  - null  = call not eligible or insufficient evidence
             - data_type must be “boolean”.
             - Add a rule: “Do NOT compute any percentage at the call level.”
        
        2. If the metric is naturally per-call (adherence, presence, quality):
             - Keep the appropriate data_type (boolean, int, float, text).
        
        3. Derive detection rules from the metric’s definition:
             - What counts as an event?
             - What makes the call eligible?
             - What phrases or patterns indicate a positive detection?
        
        4. Evidence requirements:
             - Up to 3 transcript quotes (short, exact, relevant)
             - Must support the metric_value logically
        
        5. OUTPUT PROMPT STRUCTURE
        Generate an instruction prompt in the following format:
        
            Extract all information in the transcript related to the metric “<metric_name>”.
        
            Definition:
                <Rewrite the metric definition into simple detection rules.>
        
            Task:
                1. Determine whether the call is eligible for evaluation.
                2. Detect whether the event defined by the metric occurred.
                4. Use ONLY explicit evidence from the transcript.
        
            Evidence selection:
                - Return up to 3 exact transcript lines.
                - Choose lines that most clearly show presence/absence of the event.
        
            Instructions:
                1) If diarized transcript is provided, use it; otherwise fall back to the plain transcript.
                2) Return VALID JSON ONLY (no markdown, no extra text).
                3) OUTPUT: {{{{"metric_name": "<metric_name>",
                      "data_type": "<list | int | float | percentage | boolean | text | null>",
                      "metric_value": <value or null>,
                      "evidence": "<short reasoning with transcript quotes>"}}}}
                Plain transcript: {{transcript_text}}
                Diarized transcript (preferred if provided): {{diarized_transcript}}
        
        ## FINAL REQUIREMENT
        Generate ONLY the instruction prompt.
        Do NOT output any metric results.
        Do NOT explain your reasoning.
        Do NOT generate any metric output
        
        """,
        input_variables=["metric_json"]
    )

    def __init__(self, llm: None, name: str = "TestAgentOne", ):
        super().__init__(name=name, prompt_template=self.prompt, output_parser=None, llm=llm)

    def handle(self, metric_json: str):
        print(f"[{self.name}] working ...")
        # print(self.log_promt({"metric_json": metric_json}))
        return self.generate_responseV3({
            "metric_json": metric_json
        })
