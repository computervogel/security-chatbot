from huggingface_hub import hf_hub_download
print("Starting download...")
hf_hub_download(repo_id="NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF", filename="Hermes-2-Pro-Llama-3-8B.Q4_0.gguf")
print("Download completed.")