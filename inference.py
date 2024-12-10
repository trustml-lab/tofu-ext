import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
from pathlib import Path
import os
import hydra
import re

from utils import get_model_identifiers_from_yaml

# Assuming you're running inference on a model with generation capability

def run_inference(cfg, model, tokenizer, eval_task, input_texts):
    """
    Run inference on the model using the given input texts.
    """
    model.eval()

    # Tokenize the input texts
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=cfg.generation.max_length).to(model.device)

    # Generate output from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            max_length=cfg.generation.max_length, 
            do_sample=False,  # Set to True if you want to sample (random output)
            num_return_sequences=1,  # Number of sequences to return for each input
            pad_token_id=tokenizer.eos_token_id  # Padding token ID
        )

    # Decode the generated tokens
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_texts

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg):
    # Load model configuration and tokenizer
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file) or re.search("model-*\.safetensors", file):
            path_found = True
            break

    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print(f"Loading from checkpoint : {cfg.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"]=="true",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_flash_attention_2=model_cfg["flash_attention2"]=="true",
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, model_id=cfg.model_path)
        model = model.merge_and_unload()


    model.to(cfg.device)  # Move model to appropriate device (GPU or CPU)

    # Example of input texts for inference
    input_texts = [
        "Could you mention some of Jaime Vasquez's award-winning books?",
        "Can you possibilly write down some of Jaime Vasquez's award-winning books?",
        'What are some notable literary works that earned Jaime Vasquez recognition',
        'Which books by J. Vasquez received literary prizes?',
        'Can you name the award-winning publications authored by Jaime Vasquez?',
        'Tell me about the acclaimed literary works of Jamie Vasquez'
    ]
    generated_texts = run_inference(cfg, model, tokenizer, 'inference_task', input_texts)

    # Print the generated texts
    for input_text, generated_text in zip(input_texts, generated_texts):
        print(f"Input: {input_text}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

    print('ready')

    while True:
        input_texts = input()
        if input_texts == 0:
            break
        
        input_texts = [input_texts]

        # Perform inference and get generated outputs
        generated_texts = run_inference(cfg, model, tokenizer, 'inference_task', input_texts)

        # Print the generated texts
        for input_text, generated_text in zip(input_texts, generated_texts):
            print(f"Input: {input_text}")
            print(f"Generated: {generated_text}")
            print("-" * 50)


if __name__ == "__main__":
    main()
