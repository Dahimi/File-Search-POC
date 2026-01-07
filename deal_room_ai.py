"""
Deal Room AI - Core AI/LLM logic for document intelligence

This module contains all Google GenAI / File Search related functionality,
separated from the UI layer for easy integration into production APIs.
"""
from google import genai
from google.genai import types
import time
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional


# --- System Prompt for PE Due Diligence ---
DEFAULT_SYSTEM_PROMPT = """You are a senior Private Equity analyst conducting rigorous due diligence. You have access to deal documents and must provide comprehensive, evidence-based analysis.

## CRITICAL: ZERO HALLUCINATION POLICY

- ONLY state facts explicitly found in the documents
- If information isn't in the documents, say "Not found in available documents"
- NEVER invent numbers, dates, names, or facts
- Clearly distinguish between: document facts vs. your analytical inferences

## QUERY DECOMPOSITION — Think Through Multiple Angles

Before answering, mentally break down the question into sub-aspects. For any query, consider:

**If asking about a company/deal:**
- Financial performance (revenue, margins, EBITDA, growth trends, cash flow)
- Operational aspects (business model, key processes, capacity, technology)
- Market position (market size, competition, customer base, barriers to entry)
- Management & organization (team experience, structure, key person dependencies)
- Risks & red flags (concentration risks, inconsistencies, missing information)

**If asking about risks:**
- Financial risks (revenue concentration, margin pressure, debt levels, working capital)
- Operational risks (key person, supplier dependency, capacity constraints)
- Market risks (competition, disruption potential, regulatory exposure)
- Deal risks (valuation concerns, integration challenges, hidden liabilities)

**If asking for a summary or overview:**
- Investment thesis — why this deal makes sense
- Key strengths and value drivers
- Critical concerns and risk factors
- Information gaps — what needs more diligence

## COMPREHENSIVE COVERAGE — Don't Leave Gaps

1. **Address ALL aspects** of the question — never give partial answers
2. **Search thoroughly** — information may be scattered across different document sections
3. **Try alternative angles** — if you don't find something one way, think about where else it might be mentioned
4. **If one aspect has no data**, explicitly state that and move to the next
5. **Connect the dots** — relate findings across different sections when relevant

## EVIDENCE-BASED RESPONSES

Every claim must be grounded in the documents:
- ✅ "The CIM states 2023 revenue was **$15.2M**, representing 12% YoY growth"
- ✅ "Per the financial statements, EBITDA margin is **18.5%**"
- ✅ "Management bios indicate the CEO has **15 years** of industry experience"
- ❌ "The company has strong financials" (vague, unsubstantiated)
- ❌ "Revenue is approximately $15M" (don't approximate if you don't have the number)

## RESPONSE STRUCTURE

For comprehensive questions, use this structure:

### Key Findings
[Direct answer to the question — lead with the most important takeaway]

### Detailed Analysis
[Organized by relevant dimensions, with specific evidence]
- Use exact figures: **$X**, **Y%**, **Z years**
- Cite sources: "Per the CIM...", "The financials show...", "According to..."
- Cover each angle identified in your query decomposition

### Information Gaps
[What's NOT in the documents that would be relevant]
- Be specific about what's missing
- Explain why it matters for the analysis

### Risk Factors & Concerns
[Issues, red flags, or areas requiring attention]

### Recommended Next Steps
[Follow-up questions, documents to request, areas needing deeper diligence]

## WHEN YOU CAN'T FIND INFORMATION

If your initial search doesn't surface relevant information:
1. Think about alternative terms or phrasings that might be used
2. Consider which document sections might contain the information
3. Look for related or adjacent information that partially addresses the question
4. If truly not available, clearly state what's missing and suggest what documents might help

## QUALITY STANDARDS

- **Depth over brevity** — Thorough analysis beats short answers
- **Specificity** — Exact numbers, not "significant" or "substantial"
- **Skepticism** — Question optimistic projections, look for supporting evidence
- **Completeness** — Cover every aspect; if you can't find info on something, say so explicitly
- **Intellectual honesty** — Distinguish between what you know vs. what you're inferring"""


@dataclass
class ChatResponse:
    """Response from a chat query."""
    text: str
    citations: list[str]
    grounding: dict
    thinking: Optional[str] = None  # Model's thinking/reasoning if available


@dataclass
class StoreInfo:
    """Information about a deal room/store."""
    name: str
    display_name: Optional[str]
    active_documents_count: int
    pending_documents_count: int
    failed_documents_count: int


class DealRoomAI:
    """
    Core AI client for Deal Room document intelligence.
    
    This class wraps Google's GenAI File Search functionality
    for RAG-based document Q&A.
    
    Usage:
        ai = DealRoomAI(api_key="your-key")
        
        # Create a deal room
        store = ai.create_store("Acme Acquisition")
        
        # Upload documents
        ai.upload_file(store.name, "/path/to/cim.pdf", "CIM.pdf")
        
        # Query documents
        response = ai.chat(store.name, "What are the key risks?")
        print(response.text)
        print(response.citations)
        print(response.thinking)  # Model's reasoning process
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash",
        system_prompt: Optional[str] = None,
        thinking_budget: int = -1
    ):
        """
        Initialize the AI client.
        
        Args:
            api_key: Google AI API key. If None, uses GEMINI_API_KEY env var.
            model: Model to use for chat. Default: gemini-2.5-flash
            system_prompt: Custom system prompt. If None, uses default PE analyst prompt.
            thinking_budget: Token budget for model thinking/reasoning.
                           0 = disabled, -1 = dynamic, >0 = fixed budget.
                           Default: 2048
        """
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.thinking_budget = thinking_budget
        
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
    
    def list_stores(self) -> list:
        """
        List all deal rooms (file search stores).
        
        Returns:
            List of store objects.
            
        Raises:
            Exception: If the API call fails.
        """
        return list(self.client.file_search_stores.list())
    
    def create_store(self, name: str):
        """
        Create a new deal room.
        
        Args:
            name: Display name for the deal room.
            
        Returns:
            The created store object.
            
        Raises:
            Exception: If the API call fails.
        """
        return self.client.file_search_stores.create(config={"display_name": name})
    
    def delete_store(self, store_name: str) -> bool:
        """
        Delete a deal room and all its documents.
        
        Args:
            store_name: The full store name (e.g., "fileSearchStores/xxx").
            
        Returns:
            True if successful.
            
        Raises:
            Exception: If the API call fails.
        """
        self.client.file_search_stores.delete(name=store_name, config={"force": True})
        return True
    
    def get_store_info(self, store_name: str) -> Optional[StoreInfo]:
        """
        Get information about a deal room.
        
        Args:
            store_name: The full store name.
            
        Returns:
            StoreInfo object with document counts, or None if not found.
            
        Raises:
            Exception: If the API call fails.
        """
        store = self.client.file_search_stores.get(name=store_name)
        return StoreInfo(
            name=store.name,
            display_name=store.display_name,
            active_documents_count=int(store.active_documents_count or 0),
            pending_documents_count=int(store.pending_documents_count or 0),
            failed_documents_count=int(store.failed_documents_count or 0),
        )
    
    def upload_file(
        self, 
        store_name: str, 
        file_path: str, 
        display_name: Optional[str] = None,
        poll_interval: float = 1.0
    ) -> bool:
        """
        Upload a document to a deal room.
        
        Args:
            store_name: The full store name.
            file_path: Path to the file to upload.
            display_name: Display name for the file. Defaults to filename.
            poll_interval: Seconds between status checks. Default: 1.0
            
        Returns:
            True if successful.
            
        Raises:
            Exception: If the upload fails.
        """
        if display_name is None:
            display_name = os.path.basename(file_path)
        
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=file_path,
            config={"display_name": display_name}
        )
        
        # Wait for processing to complete
        while not self.client.operations.get(operation).done:
            time.sleep(poll_interval)
        
        return True
    
    def upload_file_bytes(
        self,
        store_name: str,
        file_bytes: bytes,
        filename: str,
        poll_interval: float = 1.0
    ) -> bool:
        """
        Upload a document from bytes (e.g., from a web upload).
        
        Args:
            store_name: The full store name.
            file_bytes: The file content as bytes.
            filename: Original filename (used for display and extension).
            poll_interval: Seconds between status checks.
            
        Returns:
            True if successful.
            
        Raises:
            Exception: If the upload fails.
        """
        # Save to temp file with safe name
        file_ext = os.path.splitext(filename)[1] if '.' in filename else ''
        temp_path = f"/tmp/{uuid.uuid4().hex}{file_ext}"
        
        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            return self.upload_file(store_name, temp_path, filename, poll_interval)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def chat(
        self, 
        store_name: str, 
        message: str, 
        history: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None
    ) -> ChatResponse:
        """
        Query documents in a deal room with RAG.
        
        Args:
            store_name: The full store name to search.
            message: The user's question.
            history: Optional conversation history. Each item should have
                     'role' ('user' or 'model') and 'content' (str) keys.
            system_prompt: Override the default system prompt for this query.
            thinking_budget: Override the default thinking budget for this query.
                     
        Returns:
            ChatResponse with text, citations, grounding details, and thinking.
            
        Raises:
            Exception: If the API call fails.
        """
        history = history or []
        effective_system_prompt = system_prompt or self.system_prompt
        effective_thinking_budget = thinking_budget if thinking_budget is not None else self.thinking_budget
        
        # Build conversation contents
        contents = []
        for msg in history:
            contents.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])]
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=message)]
        ))
        
        # Build config with system prompt, thinking, and file search
        config = types.GenerateContentConfig(
            system_instruction=effective_system_prompt,
            tools=[types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_name]
                )
            )],
            thinking_config=types.ThinkingConfig(
                thinking_budget=effective_thinking_budget,
                include_thoughts=True
            )
        )
        
        # Make the API call
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        # Extract thinking content if available
        thinking_text = None
        response_text = response.text
        
        # Check for thinking parts in the response
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            thinking_parts = []
            text_parts = []
            with open("parts.txt", "w") as f:
                for part in parts:
                    f.write(str(part) + "\n")
                    f.write("--------------------------------\n")
                    f.write("--------------------------------\n")
            for part in parts:
                if hasattr(part, 'thought') and part.thought:
                    print("Thought: ", part.text)
                    thinking_parts.append(part.text)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            if thinking_parts:
                thinking_text = "\n".join(thinking_parts)
            if text_parts:
                response_text = "\n".join(text_parts)
        
        # Extract citations and grounding
        citations = []
        grounding_details = {"chunks": [], "supports": []}
        
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            
            # Extract chunks (retrieved passages)
            if grounding.grounding_chunks:
                for i, chunk in enumerate(grounding.grounding_chunks):
                    if chunk.retrieved_context:
                        ctx = chunk.retrieved_context
                        detail = {
                            "index": i,
                            "title": ctx.title if ctx.title else "Unknown",
                            "text": ctx.text[:500] + "..." if ctx.text and len(ctx.text) > 500 else ctx.text,
                        }
                        grounding_details["chunks"].append(detail)
                        if ctx.title:
                            citations.append(ctx.title)
                
                citations = list(dict.fromkeys(citations))  # Dedupe preserving order
            
            # Extract supports (which chunks support which parts of the answer)
            if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                for support in grounding.grounding_supports:
                    if support.segment and support.segment.text:
                        grounding_details["supports"].append({
                            "text": support.segment.text,
                            "chunk_indices": list(support.grounding_chunk_indices) if support.grounding_chunk_indices else []
                        })
        
        return ChatResponse(
            text=response_text,
            citations=citations,
            grounding=grounding_details,
            thinking=thinking_text
        )


# Convenience function for quick usage
def create_client(
    api_key: Optional[str] = None, 
    model: str = "gemini-2.5-flash",
    system_prompt: Optional[str] = None,
    thinking_budget: int = 2048
) -> DealRoomAI:
    """
    Create a DealRoomAI client.
    
    Args:
        api_key: Google AI API key. If None, uses GEMINI_API_KEY env var.
        model: Model to use for chat.
        system_prompt: Custom system prompt. If None, uses default PE analyst prompt.
        thinking_budget: Token budget for model thinking. Default: 2048
        
    Returns:
        DealRoomAI instance.
    """
    return DealRoomAI(
        api_key=api_key, 
        model=model,
        system_prompt=system_prompt,
        thinking_budget=thinking_budget
    )


# --- Future: Query Decomposition (Level 2) ---
# 
# For complex queries that require multi-angle analysis, consider implementing:
#
# 1. Query Decomposition:
#    - Use an LLM to break complex query into sub-questions
#    - Run file search for each sub-question in parallel
#    - Aggregate results and synthesize final answer
#
# 2. Iterative Retrieval:
#    - After initial retrieval, analyze if information gaps exist
#    - Generate follow-up queries to fill gaps
#    - Continue until sufficient coverage
#
# Example structure:
#
# def chat_advanced(self, store_name: str, message: str, ...) -> ChatResponse:
#     # Step 1: Decompose query
#     sub_queries = self._decompose_query(message)
#     
#     # Step 2: Run parallel searches
#     results = [self._search(store_name, q) for q in sub_queries]
#     
#     # Step 3: Synthesize final answer
#     return self._synthesize(message, results)

