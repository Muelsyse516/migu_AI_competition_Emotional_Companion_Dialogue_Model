"""Offline inference for a calibrated user-profile encoder."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ProfileEncoder:
    def __init__(self, path):
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, attn_implementation='sdpa',
        ).to('cuda').eval()
        self.selection = json.loads((path / 'selection.json').read_text())
        self.labels = json.loads((path / 'label_manifest.json').read_text())['labels']
        assert len(self.labels) == self.model.config.num_labels

    def predict(self, row):
        tok = self.tokenizer
        text = '\n'.join(t['content'] for t in row['history'] if t['role'] == 'user')
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > 510:
            ids = ids[:96] + ids[-414:]
        ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
        # Padding reproduces the development experiment's input construction.
        mask = [1] * len(ids) + [0] * (512 - len(ids))
        ids += [tok.pad_token_id] * (512 - len(ids))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(
                input_ids=torch.tensor([ids], device='cuda'),
                attention_mask=torch.tensor([mask], device='cuda'),
            ).logits
        probabilities = logits.float().sigmoid()[0].cpu().tolist()
        result = {field: [] for field in self.selection['thresholds']}
        for (field, label), probability in zip(self.labels, probabilities):
            if probability >= self.selection['thresholds'][field]['threshold']:
                result[field].append(label)
        return result

class EmotionEncoder(ProfileEncoder):
    def predict(self, row):
        tok = self.tokenizer
        text = '\n'.join(t['role'] + ': ' + t['content'] for t in row['history'][-5:])
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > 510:
            ids = ids[:96] + ids[-414:]
        ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
        mask = [1] * len(ids) + [0] * (512 - len(ids))
        ids += [tok.pad_token_id] * (512 - len(ids))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(
                input_ids=torch.tensor([ids], device='cuda'),
                attention_mask=torch.tensor([mask], device='cuda'),
            ).logits
        probabilities = logits.float().softmax(-1)[0].cpu()
        assert torch.isfinite(probabilities).all()
        index = int(probabilities.argmax())
        return {'emotion_label': self.labels[index][1], 'confidence': float(probabilities[index])}
