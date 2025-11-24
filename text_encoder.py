import torch
from transformers import AutoTokenizer, AutoModel

# Alternative: Using ClinicalBERT for medical text
class ClinicalTextEncoder:
    """ClinicalBERT encoder for medical text prompts"""

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.text_dim = 768  # BERT dimension

    def encode_text(self, text_prompts):
        """Encode text prompts using ClinicalBERT"""
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text_prompts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings
            outputs = self.model(**inputs)

            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings