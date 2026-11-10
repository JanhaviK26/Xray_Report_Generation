"""IU-Xray captioner from the Colab notebook.

49 image tokens from a chest X-ray DenseNet, an MLP, then DistilGPT-2.
With k > 0 the decoder also sees K retrieved training reports.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

MAXLEN = 128


class Captioner(torch.nn.Module):
    def __init__(self, finetune_enc: bool = True):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained("distilgpt2")
        d = self.lm.config.n_embd
        self.enc_head = None
        if finetune_enc:
            import torchxrayvision as xrv

            dn = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.enc_head = copy.deepcopy(dn.features[-2:])  # denseblock4 + norm5
            del dn
        self.map = torch.nn.Sequential(
            torch.nn.Linear(1024, d),
            torch.nn.GELU(),
            torch.nn.Linear(d, d),
            torch.nn.LayerNorm(d),
        )
        self.map[-1].weight.data.fill_(self.lm.transformer.wte.weight.std().item())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.enc_head is not None:
            for m in self.enc_head.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        return self

    def tokens(self, feats: torch.Tensor) -> torch.Tensor:
        if self.enc_head is not None:  # [B, 512, 7, 7] -> [B, 49, 1024]
            feats = F.relu(self.enc_head(feats)).flatten(2).transpose(1, 2)
        return self.map(feats)


class RagCaptioner(Captioner):
    def __init__(self, k: int, finetune_enc: bool = True):
        super().__init__(finetune_enc=finetune_enc)
        self.k = k

    def prefix(self, feats: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        pre = self.tokens(feats)
        if self.k:
            pre = torch.cat([pre, self.lm.transformer.wte(ctx)], 1)
        return pre

    @torch.no_grad()
    def generate(self, feats, ctx, eos_id: int, beams: int = 1, ngram: int = 0):
        pre = self.prefix(feats, ctx)
        att = torch.ones(pre.shape[:2], dtype=torch.long, device=pre.device)
        kwargs = dict(
            inputs_embeds=pre,
            attention_mask=att,
            max_new_tokens=MAXLEN,
            num_beams=beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
        if ngram and ngram > 0:
            kwargs["no_repeat_ngram_size"] = ngram
        return self.lm.generate(**kwargs)
