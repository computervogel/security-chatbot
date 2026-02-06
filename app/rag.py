import os
import time
from gpt4all import GPT4All
from app.ingest import collection

# --- CONFIGURATION ---
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

print("--- LOADING AI MODEL (GPT4All) ---")
# If you don't have a GPU, set device="cpu"
llm = GPT4All(MODEL_NAME, device="cpu") 
print("--- MODEL LOADED ---")

# --- STRICT SYSTEM PROMPT (Llama 3 Optimized) ---
SYSTEM_PROMPT = """You are a strict ISO/IEC/IEEE 29148 Security Auditor and a security requirements engineer.

CORE RULES:
1. IF the user asks a knowledge QUESTION (e.g., "What is TLS?", "How does..."):
   - Answer briefly using the provided Context.
   - Do NOT produce an Audit Report.

2. IF the user provides a REQUIREMENT (e.g., "System must be secure", "Users need to login", etc.):
   - You MUST act as an Auditor.
   - Vague words like "secure", "fast", "easy", "clean" or similar are FORBIDDEN.
   - SCORING RULE: If a requirement contains vague words, the Score MUST be between 1 and 3. Never higher.
   - You must output the result in this Markdown format:

3. COMPLIANCE CHECK: Before scoring, check if the requirement contradicts the REFERENCE CONTEXT. 
   - If the context says "90 days" and the user says "120 days", the Score MUST be 1/10.
   - Mention the specific policy violation in the Analysis.

### 📊 Audit Report
- **Score:** [1-10]
- **Analysis:** [Explain why it fails ISO 29148]
### ✨ Improved Proposal
[Rewrite as a SMART requirement and later explain how it meets ISO 29148 and why it is better]

--- EXAMPLE OF EXPECTED BEHAVIOR ---
Input: "The page should load fast."
Output:
### 📊 Audit Report
- **Score:** 2/10
- **Analysis:** The term "fast" is subjective and not measurable.
### ✨ Improved Proposal
The webpage must load the main content within 1.5 seconds on a 4G network connection.
-------------------------------------
"""

def generate_response(user_query: str, session_id: str):
    start_time = time.time()
    
    # 1. Retrieval (Limit to 2 chunks to confuse the model less)
    results = collection.query(
        query_texts=[user_query],
        n_results=2,
        where={
            "$or": [
                {"session_id": session_id},
                {"type": "supervisor"}
            ]
        }
    )
    
    context_text = ""
    sources = []
    
    if results['documents'] and results['documents'][0]:
        context_text = "\n\n".join(results['documents'][0])
        sources = list(set([m['source'] for m in results['metadatas'][0]]))
    
    # 2. Build Prompt using Official Llama 3 Special Tokens
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}\n"
        f"CONTEXT:\n{context_text}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_query}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # 3. Generate
    response = llm.generate(
        full_prompt, 
        max_tokens=600, 
        temp=0.1, 
        top_k=40,
        top_p=0.4
    )

    print(f"--- Generation took {round(time.time() - start_time, 2)} seconds ---")

    return {
        "answer": response,
        "sources": sources
    }