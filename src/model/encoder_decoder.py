"""CNN encoder + GPT-2 decoder; image vector fused as first sequence position."""

from typing import Optional  # optional forward args

import torch  # tensors
import torch.nn as nn  # Module, Linear, Dropout
from torchvision.models import densenet121, resnet18  # backbones
from transformers import GPT2LMHeadModel  # causal LM with loss


class DenseNetEncoder(nn.Module):
    """DenseNet-121 -> GAP -> Linear(embed_dim)."""

    def __init__(self, pretrained: bool = True, embed_dim: int = 768):
        super().__init__()  # init nn.Module
        dnet = densenet121(weights="IMAGENET1K_V1" if pretrained else None)  # load backbone
        self.features = dnet.features  # convolutional trunk
        self.num_features = dnet.classifier.in_features  # channel count before classifier
        self.proj = nn.Linear(self.num_features, embed_dim)  # map to decoder dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)  # (B, C, H, W)
        out = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # (B, C, 1, 1)
        out = out.flatten(1)  # (B, C)
        return self.proj(out)  # (B, embed_dim)


class ResNet18Encoder(nn.Module):
    """ResNet-18 -> GAP -> Linear(embed_dim)."""

    def __init__(self, pretrained: bool = True, embed_dim: int = 768):
        super().__init__()
        rn = resnet18(weights="IMAGENET1K_V1" if pretrained else None)  # ImageNet weights
        self.conv1 = rn.conv1  # stem
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1  # residual stages
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        self.avgpool = rn.avgpool  # global pool
        self.num_features = 512  # ResNet-18 C before fc
        self.proj = nn.Linear(self.num_features, embed_dim)  # to decoder dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # (B,3,224,224) -> ...
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # (B,512,1,1)
        x = torch.flatten(x, 1)  # (B,512)
        return self.proj(x)  # (B, embed_dim)


def _make_encoder(encoder_name: str, pretrained: bool, embed_dim: int) -> nn.Module:
    if encoder_name and encoder_name.lower() in ("resnet18", "resnet_18"):  # config string
        return ResNet18Encoder(pretrained=pretrained, embed_dim=embed_dim)  # light encoder
    return DenseNetEncoder(pretrained=pretrained, embed_dim=embed_dim)  # heavier option


class ReportGenerator(nn.Module):
    """Encoder + GPT2LMHeadModel; image embedding prepended to text embeddings."""

    def __init__(
        self,
        tokenizer,  # used in generate() for ids
        encoder_embed_dim: int = 768,  # encoder output size
        decoder_name: str = "distilbert/distilgpt2",  # HF model id
        pretrained_encoder: bool = True,  # ImageNet for CNN
        dropout: float = 0.1,  # after projection
        encoder_name: str = "resnet18",  # which CNN
    ):
        super().__init__()
        self.tokenizer = tokenizer  # store ref
        self.encoder = _make_encoder(encoder_name, pretrained_encoder, encoder_embed_dim)  # CNN
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name)  # loads LM head
        self.encoder_embed_dim = encoder_embed_dim  # record
        self.decoder_embed_dim = self.decoder.config.n_embd  # GPT-2 d_model
        if encoder_embed_dim != self.decoder_embed_dim:  # need projection
            self.encoder_to_decoder = nn.Linear(encoder_embed_dim, self.decoder_embed_dim)  # align dims
        else:
            self.encoder_to_decoder = nn.Identity()  # no-op
        self.dropout = nn.Dropout(dropout)  # regularize image vector

    def forward(
        self,
        image: torch.Tensor,  # (B,3,H,W)
        input_ids: Optional[torch.Tensor] = None,  # token ids
        attention_mask: Optional[torch.Tensor] = None,  # 1/0 mask
        labels: Optional[torch.Tensor] = None,  # same shape as input_ids; -100 = ignore
    ):
        enc_out = self.encoder(image)  # (B, enc_dim)
        enc_out = self.dropout(self.encoder_to_decoder(enc_out))  # (B, dec_dim)
        image_emb = enc_out.unsqueeze(1)  # (B, 1, dec_dim) virtual first token
        if input_ids is not None:  # training / teacher forcing
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T, dec_dim)
            inputs_embeds = torch.cat([image_emb, inputs_embeds[:, :-1, :]], dim=1)  # align lengths
            outputs = self.decoder(  # HF computes loss vs labels
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return outputs  # CausalLMOutputWithPast
        return enc_out  # image-only embedding if no ids

    def generate(
        self,
        image: torch.Tensor,  # (B,3,H,W)
        max_length: int = 128,  # max tokens to generate
        num_beams: int = 4,  # >1 triggers beam search
        length_penalty: float = 1.0,  # alpha in length norm
        repetition_penalty: float = 1.2,  # penalize reuse
        temperature: float = 1.0,  # >1 sampling, 1 greedy
        top_p: float = 0.95,  # nucleus
        top_k: int = 0,  # 0 = disabled
        no_repeat_ngram_size: int = 0,  # 0 = disabled
        **kwargs,  # ignored extras
    ):
        enc_out = self.encoder(image)  # encode batch
        enc_out = self.dropout(self.encoder_to_decoder(enc_out))  # project
        image_emb = enc_out.unsqueeze(1)  # (B,1,D)
        return self._generate_with_image_embedding(  # delegate
            image_emb,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs,
        )

    def _generate_with_image_embedding(
        self,
        image_emb: torch.Tensor,  # (B,1,D) already projected
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.2,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        no_repeat_ngram_size: int = 0,
        **kwargs,
    ):
        if num_beams > 1:  # beam path
            return self._beam_search(
                image_emb,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        batch_size = image_emb.size(0)  # B
        device = image_emb.device  # cpu or cuda
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id  # start token
        if bos_id is None:
            bos_id = self.tokenizer.cls_token_id  # rare fallback
        start = torch.full(
            (batch_size, 1),
            bos_id if bos_id is not None else 0,
            dtype=torch.long,
            device=device,
        )  # (B,1)
        curr_emb = self.decoder.get_input_embeddings()(start)  # embed BOS
        curr_emb = torch.cat([image_emb, curr_emb], dim=1)  # fusion: [img, BOS]
        generated = start  # token ids generated so far
        use_sampling = temperature > 0 and temperature != 1.0  # stochastic?
        for _ in range(max_length - 1):  # autoregressive steps
            if curr_emb.size(1) >= max_length:  # safety cap
                break
            attn = torch.ones(curr_emb.size(0), curr_emb.size(1), device=device, dtype=torch.long)  # all valid
            out = self.decoder(inputs_embeds=curr_emb, attention_mask=attn)  # forward
            logits = out.logits[:, -1, :].float().clamp(-100.0, 100.0)  # next-token logits
            if repetition_penalty != 1.0 and generated.size(1) > 0:  # HF-style repetition penalty
                for b in range(batch_size):
                    for t in generated[b].tolist():
                        if logits[b, t] > 0:
                            logits[b, t] /= repetition_penalty
                        else:
                            logits[b, t] *= repetition_penalty
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:  # block repeated n-grams
                n = no_repeat_ngram_size
                for b in range(batch_size):
                    prev_ngram = tuple(generated[b, -(n - 1) :].tolist())
                    for i in range(0, generated.size(1) - (n - 1)):
                        if tuple(generated[b, i : i + n - 1].tolist()) == prev_ngram:
                            bad_token = generated[b, i + n - 1].item()
                            logits[b, bad_token] = -float("inf")
                            break
            if top_k > 0:  # mask low logits
                for b in range(batch_size):
                    k = min(top_k, logits.size(1))
                    _, top_indices = torch.topk(logits[b], k)
                    mask = torch.ones_like(logits[b], dtype=torch.bool)
                    mask[top_indices] = False
                    logits[b].masked_fill_(mask, -float("inf"))
            if temperature > 0 and temperature != 1.0:  # scale logits
                logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)  # distribution
            if top_p < 1.0 and top_p > 0:  # nucleus sampling filter
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8))
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            if use_sampling:  # draw from probs
                next_token = torch.multinomial(probs + 1e-10, 1)
            else:  # greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)  # append token
            if (next_token == self.tokenizer.eos_token_id).all():  # optional early exit
                break
            next_emb = self.decoder.get_input_embeddings()(next_token)  # embed new token
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)  # extend context
        return generated  # (B, seq_len)

    def _beam_search(
        self,
        image_emb: torch.Tensor,  # (B, 1, D)
        max_length: int,
        num_beams: int,
        length_penalty: float,
        repetition_penalty: float,
        top_k: int,
        no_repeat_ngram_size: int,
    ):
        B, _, D = image_emb.shape  # batch, 1, dim
        device = image_emb.device
        vocab_size = self.decoder.config.vocab_size  # vocab size
        eos_id = self.tokenizer.eos_token_id  # end token
        if eos_id is None:
            eos_id = -1  # disable eos checks
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if bos_id is None:
            bos_id = 0
        alpha = length_penalty  # length norm exponent

        image_emb = image_emb.repeat_interleave(num_beams, dim=0)  # (B*K, 1, D)
        beam_scores = torch.zeros(B * num_beams, device=device)  # log prob sums
        sequences = torch.full(
            (B * num_beams, 1),
            bos_id,
            dtype=torch.long,
            device=device,
        )  # start with BOS
        curr_emb = self.decoder.get_input_embeddings()(sequences)  # embed BOS
        curr_emb = torch.cat([image_emb, curr_emb], dim=1)  # [img, BOS]

        for step in range(1, max_length):  # beam steps
            attn = torch.ones(curr_emb.size(0), curr_emb.size(1), device=device, dtype=torch.long)
            out = self.decoder(inputs_embeds=curr_emb, attention_mask=attn)
            logits = out.logits[:, -1, :].float().clamp(-100.0, 100.0)

            if repetition_penalty != 1.0:
                for i in range(sequences.size(0)):
                    for t in sequences[i].tolist():
                        if t < 0 or t >= vocab_size:
                            continue
                        if logits[i, t] > 0:
                            logits[i, t] /= repetition_penalty
                        else:
                            logits[i, t] *= repetition_penalty

            if no_repeat_ngram_size > 0 and sequences.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for i in range(sequences.size(0)):
                    prev_ngram = tuple(sequences[i, -(n - 1) :].tolist())
                    for j in range(0, sequences.size(1) - (n - 1)):
                        if tuple(sequences[i, j : j + n - 1].tolist()) == prev_ngram:
                            bad = sequences[i, j + n - 1].item()
                            if 0 <= bad < vocab_size:
                                logits[i, bad] = -float("inf")
                            break

            if top_k > 0:
                k = min(top_k, logits.size(1))
                top_vals, _ = torch.topk(logits, k, dim=-1)
                thresh = top_vals[:, -1:].clamp(max=0)
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # log p(next|ctx)
            new_scores = beam_scores.unsqueeze(1) + log_probs  # accumulate log prob
            cur_len = step + 1  # generated length
            lp = ((5.0 + cur_len) ** alpha) / (6.0 ** alpha)  # length penalty divisor
            normalized = new_scores / lp  # score for ranking hypotheses

            K, V = num_beams, logits.size(1)  # K beams, V vocab
            normalized_flat = normalized.view(B, K * V)  # all extensions per image
            top_scores, top_idx = torch.topk(normalized_flat, num_beams, dim=1)  # keep top K
            prev_beams = top_idx // V  # which beam prev step
            next_tokens = top_idx % V  # which token
            base = torch.arange(B, device=device).unsqueeze(1) * K  # batch offsets
            beam_indices = (base + prev_beams).view(-1)  # global beam indices
            beam_scores = new_scores.view(B, K * V).gather(1, top_idx).view(-1)  # raw scores for survivors
            next_tokens_flat = next_tokens.view(-1)

            sequences = sequences[beam_indices]  # reorder beams
            sequences = torch.cat([sequences, next_tokens_flat.unsqueeze(1)], dim=1)  # append token
            curr_emb = curr_emb[beam_indices]  # reorder embeddings
            next_emb = self.decoder.get_input_embeddings()(next_tokens_flat.unsqueeze(1))
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)  # extend

            if eos_id >= 0 and (next_tokens_flat == eos_id).all():  # all beams ended (rare)
                break

        final_len = sequences.size(1)
        lp_final = ((5.0 + final_len) ** alpha) / (6.0 ** alpha)
        normalized_final = beam_scores / lp_final  # final normalized scores
        normalized_final = normalized_final.view(B, num_beams)
        best_beams = normalized_final.argmax(dim=1)  # best beam index per image
        best_indices = torch.arange(B, device=device) * num_beams + best_beams  # rows in flat tensor
        return sequences[best_indices]  # (B, seq_len) best hypothesis per image
