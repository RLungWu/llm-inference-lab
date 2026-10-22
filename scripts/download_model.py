from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm import load, generate

repo_id = "mlx-community/Qwen3.5-0.8B-bf16"
target_dir = Path(__file__).resolve().parent.parent / "model"

print("Start Download")
model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir
)

print("Start Load Model")
model, tokenizer = load(model_path)

print("Start Generate")
response = generate(model, tokenizer, prompt="Who are you", verbose=True)
