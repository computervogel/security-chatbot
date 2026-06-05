import os
import time
import json
import ast
import re
from gpt4all import GPT4All
from app.ingest import collection
from app.database import get_session_artifacts

# --- CONFIGURATION ---
MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_0.gguf"

print("--- LOADING AI MODEL (GPT4All) ---")

# If you don't have a GPU, set device="cpu"; change threads based on your cpu cores
llm = GPT4All(MODEL_NAME, device="cpu", n_threads=10) 
print("--- MODEL LOADED ---")

# --- PROMPT 1: AUDITOR (QA) ---
AUDIT_PROMPT = """You are a strict ISO/IEC/IEEE 29148 Security Auditor specialized in the Volere Security Framework.

CORE SECURITY PRINCIPLES (The 5 Security Requirement Types):
1. Access: Authentication, authorization, user roles, and access control boundaries.
2. Integrity: Prevention of unauthorized data/code tampering, validation mechanisms.
3. Privacy: Data minimization, anonymity, confidentiality, and regulatory data compliance (GDPR).
4. Audit: System logging, tracking of security-critical actions, accountability, and traceability.
5. Immunity: Resistance against active exploits, malware, brute-force, rate-limiting, and input validation.

CORE MISSION:
If the user asks a question, answer based on the CONTEXT. 
If the user provides a requirement, audit it against ISO 29148 standards using the following general rubric.

SCORING RUBRIC & EMOJIS:
Use ONLY these emojis based on your score: 🔴 (1-3), 🟡 (4-7), 🟢 (8-10).

1. Unambiguity (Score 1-10):
   - Penalty (1-3): The requirement uses qualitative, subjective adjectives (e.g., 'secure login', 'efficient logging').
   - Reward (8-10): The requirement uses precise, objective, and non-negotiable technical terms.

2. Verifiability (Score 1-10):
   - Penalty (1-3): A developer or tester cannot write a clear binary Pass/Fail automated test for this.
   - Reward (8-10): Contains a measurable threshold, a specific cryptographic standard, or a clear binary state.

3. Type Alignment & Compliance (Score 1-10):
   - Reward (8-10): The requirement aligns perfectly with one of the 5 Types and directly addresses an architectural threat found in the CONTEXT.
   - Penalty (1-3): The requirement does not fit any type or directly contradicts security controls stated in the CONTEXT.

ANTI-HALLUCINATION RULE:
For the "Improved Proposal", DO NOT invent specific technical parameters, numbers, or algorithms (like specific bit-lengths or iterations) unless they are explicitly stated in the CONTEXT. If context is missing, propose a general architectural mechanism (e.g., "The system shall encrypt data in transit" instead of inventing "TLS 1.3 with AES-256").

STRICT OUTPUT TEMPLATE:
### 📊 Audit Report

**Identified Requirement Type:** [Access / Integrity / Privacy / Audit / Immunity]

**Detailed Breakdown:**
- [Emoji] **Type Alignment \& Compliance:** [Score]/10 - [1 sentence reason]
- [Emoji] **Unambiguity:** [Score]/10 - [1 sentence reason]
- [Emoji] **Verifiability:** [Score]/10 - [1 sentence reason]

**Analysis:** [1-2 sentences summarizing the main architectural flaws regarding its specific security requirement type.]

### ✨ Improved Proposal
[Rewrite as a SMART requirement adhering to the Anti-Hallucination rule.]
"""

# --- PROMPT 2: ELICITATION (GUIDANCE) ---
ELICITATION_PROMPT = """You are an expert Security Requirements Engineer guiding a user to write a SMART security requirement step-by-step.
You strictly structure your analysis around the 5 fundamental Security Requirement Types: Access, Integrity, Privacy, Audit, and Immunity.

The user will provide an initial idea (e.g., "I need a secure login").

YOUR TASK:
Do not ask open-ended questions. Instead, act as a guide and offer concrete choices based on the provided CONTEXT (Bloomodoro architecture, security objectives, OWASP). 

Analyze what the user wants and identify what is missing to make it a complete SMART requirement:
1. Target/Asset: What component is being protected under the selected category (Access, Integrity, Privacy, Audit, Immunity)?
2. Mechanism: What architectural mechanism handles it?
3. Parameter/Threshold: What are the exact metrics or constraints?

Ask exactly ONE logical follow-up question to fill the next missing gap, and provide 2-3 specific options extracted from the CONTEXT.

OUTPUT FORMAT:
1. **Acknowledge:** Briefly confirm what we are working on and state which of the 5 Requirement Types it maps to.
2. **The Question:** Ask ONE specific question to define the next technical detail.
3. **Options:** Provide 2-3 concrete options (A, B, C) based on the CONTEXT (e.g., specific parameters, components, or protocols from the architecture text).
4. **Call to Action:** Ask the user to choose an option or propose their own.
"""

def generate_response(user_query: str, session_id: str, intent: str = "qa", process_state: str = "Asset"):
    start_time = time.time()
    
    # 1. Retrieval
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
    
    # 2. Shared Context
    artifacts = get_session_artifacts(session_id)
    artifacts_text = "No previously approved artifacts yet."
    if artifacts:
        artifacts_text = "\n".join([f"- [{art['type']}] {art['content']}" for art in artifacts])

    # 3. Intent Routing & State Injection
    if intent == "elicitation":
        active_prompt = ELICITATION_PROMPT
    else:
        active_prompt = AUDIT_PROMPT
    
    state_instruction = (
        f"CURRENT PROCESS PHASE: You are currently working on defining '{process_state}s'. "
        f"Strictly focus your response and options on this specific phase.\n\n"
        f"APPROVED PROJECT ARTIFACTS (Use these as context):\n{artifacts_text}\n"
    )

    # 4. Build Prompt
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{active_prompt}\n\n"
        f"{state_instruction}\n\n"
        f"CONTEXT (PDFs):\n{context_text}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_query}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # 5. Generate
    response = llm.generate(
        full_prompt, 
        max_tokens=400,
        temp=0.1, 
        top_k=40,
        top_p=0.4
    )

    print(f"--- Generation took {round(time.time() - start_time, 2)} seconds ---")

    return {
        "answer": response,
        "sources": sources
    }

def scan_full_document(session_id: str, filename: str):
    results = collection.get(
        where={"$and": [{"session_id": session_id}, {"source": filename}]}
    )
    
    if not results['documents']:
        return []

    all_artifacts = []
    
    print(f"--- Starting Full Document Scan for {filename} ({len(results['documents'])} chunks) ---")
    
    for i, chunk in enumerate(results['documents']):
        print(f"Scanning chunk {i+1}/{len(results['documents'])}...")
        
        extraction_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a Security Architect. Analyze the following functional description.\n"
            f"Task: Identify implicit and explicit security artifacts based on the system's functionality.\n"
            f"Focus areas for requirements include Volere types: Access, Integrity, Privacy, Audit, Immunity.\n"
            f"1. Assets: What data or components have value?\n"
            f"2. Threats: What could happen to these assets?\n"
            f"3. Goals: High-level security objectives.\n"
            f"4. Req: Concrete security requirements derived from the description.\n"
            f"Return ONLY a valid JSON array using DOUBLE QUOTES.\n"
            f"Format strictly like this: [{{\"type\": \"Asset\", \"content\": \"User Study Data\"}}]\n"
            f"Allowed types: Asset, Threat, Goal, Req.\n<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"TEXT TO ANALYZE:\n{chunk}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n["
        )

        response = llm.generate(extraction_prompt, max_tokens=250, temp=0.1)
        full_response = "[" + response
        
        artifacts = []
        try:
            start_idx = full_response.find('[')
            end_idx = full_response.rfind(']') + 1
            clean_string = full_response[start_idx:end_idx] if end_idx > 0 else full_response
            
            try:
                artifacts = json.loads(clean_string)
            except Exception:
                try:
                    artifacts = ast.literal_eval(clean_string)
                except Exception:
                    raise ValueError("Syntax broken")
                    
        except Exception as e:
            print(f"-> Syntax broken in chunk {i+1}. Engaging Regex Fallback...")
            blocks = re.findall(r'\{[^{}]*\}', full_response)
            for block in blocks:
                t_match = re.search(r'[\'"]type[\'"]\s*:\s*[\'"](Asset|Threat|Goal|Req)[\'"]', block, re.IGNORECASE)
                c_match = re.search(r'[\'"]content[\'"]\s*:\s*[\'"](.*?)[\'"]\s*\}', block, re.DOTALL | re.IGNORECASE)
                
                if t_match and c_match:
                    artifacts.append({
                        "type": t_match.group(1).capitalize(),
                        "content": c_match.group(1).strip()
                    })

        if isinstance(artifacts, list):
            all_artifacts.extend(artifacts)
            print(f"-> Extracted {len(artifacts)} artifacts from chunk {i+1}.")

    unique_artifacts = []
    seen = set()
    for item in all_artifacts:
        content_lower = item.get("content", "").lower().strip()
        art_type = item.get("type", "")
        
        if content_lower and art_type in ["Asset", "Threat", "Goal", "Req"]:
            if content_lower not in seen:
                seen.add(content_lower)
                unique_artifacts.append(item)

    print(f"--- Scan complete. Found {len(unique_artifacts)} unique artifacts. ---")
    return unique_artifacts