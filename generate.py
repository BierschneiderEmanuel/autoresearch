import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from prepare import Tokenizer
from train import GPT, build_model_config


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.1, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    prompt_len = len(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids
    max_seq_len = model.cos.size(1)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            if generated.size(1) >= max_seq_len:
                break
            logits = model(generated)
            logits = logits[:, -1, :].float() / temperature
            # top-k filtering
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k).values
                logits[logits < topk_vals[:, -1:]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    # return only the newly generated tokens (skip prompt + BOS)
    new_ids = generated.squeeze().tolist()[prompt_len:]
    return tokenizer.decode(new_ids)


if __name__ == "__main__":
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = build_model_config(vocab_size)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    state_dict = torch.load("model_weights.pt")
    # weights saved from torch.compile'd model have "_orig_mod." prefix — strip it
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(torch.bfloat16)  # ensure uniform dtype
    prompt = "tell me a story"
    output = generate_text(model, tokenizer, prompt, max_new_tokens=32)
    print("Prompt:", prompt)
    print("Generated:", output)
