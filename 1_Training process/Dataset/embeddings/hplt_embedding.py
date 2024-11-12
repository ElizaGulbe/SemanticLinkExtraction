import torch

import transformers

# see more information about HPLT embeddings here - https://huggingface.co/HPLT/hplt_bert_base_lv

transformers.logging.set_verbosity_error()

def get_embedding(text, tokenizer, model, device): 
    # Tokenize input text and move the input tensors to the GPU if available
    input_text = tokenizer(text, return_tensors="pt").to(device)

    # Get model output including hidden states
    with torch.no_grad():
        outputs = model(**input_text, output_hidden_states=True)

    # Extract the hidden states (the last layer)
    hidden_states = outputs.hidden_states

    # Ensure hidden_states is not None
    if hidden_states is None:
        raise ValueError("Hidden states are not available. Please check the model configuration.")

    # The last hidden state corresponds to the final layer embeddings for all tokens
    last_hidden_state = hidden_states[-1]

    # Pooling strategy: Take the mean of all token embeddings
    embedding = torch.mean(last_hidden_state, dim=1)  # [batch_size, 768]

    return embedding

# Check if a GPU is available