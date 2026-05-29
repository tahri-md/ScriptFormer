import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,pool_kernel_size:int | tuple[int, int] = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size,stride=pool_kernel_size)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self,hidden_size:int = 256,dropout:float=0.1,debug_shapes:bool=False):
        super().__init__()
        self.block1 = ConvBlock(1,32,pool_kernel_size=2)
        self.block2 = ConvBlock(32,64,pool_kernel_size=2)
        self.block3 = ConvBlock(64,128,pool_kernel_size=(2,1))
        self.block4 = ConvBlock(128,256,pool_kernel_size=(2,1))
        self.projection = nn.Linear(256*4,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.context_blend = 0.5
        self.context_kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32).view(1, 1, 3)
        self.debug_shapes = debug_shapes

    def _mix_sequence_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 3:
            return x

        sequence = x.transpose(1, 2)
        kernel = self.context_kernel.to(device=sequence.device, dtype=sequence.dtype).expand(sequence.size(1), -1, -1)
        context = F.conv1d(sequence, kernel, padding=1, groups=sequence.size(1))
        context = context.transpose(1, 2)
        return (1.0 - self.context_blend) * x + self.context_blend * context

    def forward(self,images:torch.Tensor)->torch.Tensor:
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if self.debug_shapes:
            print(x.shape)

        B,C,H,W = x.shape
        x = x.permute(0,3,1,2)
        x = x.reshape(B,W,C*H)
        x = self.projection(x)
        x = self._mix_sequence_context(x)
        x = self.dropout(x)
        x = self.norm(x)

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self,hidden_size:int,max_length:int=200,dropout:float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_length,hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(
                    torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
                )
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x+self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
class TransfomerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_size: int = 1024,
        max_length: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=hidden_size,
                    padding_idx=pad_id,
                )
        self.positional_encoding = PositionalEncoding(hidden_size, max_length, dropout)
        self.embed_scale = math.sqrt(hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)


    def _generate_causal_mask(self,seq_len:int,device:torch.device)->torch.Tensor:
        mask = torch.triu(torch.ones(seq_len,seq_len,device=device),diagonal=1).bool()
        return mask
    
    def _generate_padding_mask(self,token_ids:torch.Tensor)->torch.Tensor:
        return token_ids == self.pad_id
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, tgt_len = target_ids.shape
        device = target_ids.device

        x = self.token_embedding(target_ids) * self.embed_scale

        x = self.positional_encoding(x)

        causal_mask = self._generate_causal_mask(tgt_len, device)
        tgt_padding_mask = self._generate_padding_mask(target_ids)

        x = self.decoder(
            tgt=x,                             
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        x = self.norm(x)           
        logits = self.output_projection(x)
        return logits
    
class ScriptFormer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_hidden: int = 256,
        decoder_hidden: int = 256,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff: int = 1024,
        max_length: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

        self.encoder = CNNEncoder(
            hidden_size=encoder_hidden,
            dropout=dropout,
            debug_shapes=debug_shapes,
        )

        self.decoder = TransfomerDecoder(
            vocab_size=vocab_size,
            hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            feedforward_size=decoder_ff,
            max_length=max_length,
            dropout=dropout,
            pad_id=pad_id,
        )

    def forward(
        self,
        images: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder(images)
        logits = self.decoder(encoder_output, target_ids)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = None,
        temperature: float = 1.0,
        beam_size: int = 1,
    ) -> torch.Tensor:
        if max_length is None:
            max_length = self.max_length

        if beam_size is None or beam_size < 1:
            beam_size = 1

        B = images.shape[0]
        device = images.device

        encoder_output = self.encoder(images)

        if beam_size > 1:
            generated_sequences = [
                self._generate_beam_search_single(
                    encoder_output[i : i + 1],
                    max_length=max_length,
                    beam_size=beam_size,
                )
                for i in range(B)
            ]
            max_generated_length = max(sequence.size(0) for sequence in generated_sequences)
            padded = torch.full(
                (B, max_generated_length),
                self.pad_id,
                dtype=torch.long,
                device=device,
            )
            for idx, sequence in enumerate(generated_sequences):
                padded[idx, : sequence.size(0)] = sequence
            return padded

        generated = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.decoder(encoder_output, generated)
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1)
            next_token[finished] = self.pad_id
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.eos_id)

            if finished.all():
                break

        return generated

    def _generate_beam_search_single(
        self,
        encoder_output: torch.Tensor,
        max_length: int,
        beam_size: int,
    ) -> torch.Tensor:
        device = encoder_output.device
        beams = [
            {
                "tokens": torch.tensor([self.sos_id], dtype=torch.long, device=device),
                "score": 0.0,
                "finished": False,
            }
        ]

        for _ in range(max_length - 1):
            candidates = []

            for beam in beams:
                if beam["finished"]:
                    candidates.append(beam)
                    continue

                logits = self.decoder(encoder_output, beam["tokens"].unsqueeze(0))
                next_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_logits, dim=-1).squeeze(0)
                top_scores, top_tokens = torch.topk(log_probs, k=min(beam_size, log_probs.size(-1)))

                for score_delta, token_id in zip(top_scores.tolist(), top_tokens.tolist()):
                    next_tokens = torch.cat(
                        [beam["tokens"], torch.tensor([token_id], dtype=torch.long, device=device)]
                    )
                    candidates.append(
                        {
                            "tokens": next_tokens,
                            "score": beam["score"] + score_delta,
                            "finished": token_id == self.eos_id,
                        }
                    )

            candidates.sort(
                key=lambda item: item["score"] / max(1, item["tokens"].size(0) - 1),
                reverse=True,
            )
            beams = candidates[:beam_size]

            if all(beam["finished"] for beam in beams):
                break

        best_beam = max(
            beams,
            key=lambda item: item["score"] / max(1, item["tokens"].size(0) - 1),
        )
        return best_beam["tokens"]

    def count_parameters(self) -> dict:
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + decoder_params
        return {
            "encoder": f"{encoder_params:,}",
            "decoder": f"{decoder_params:,}",
            "total": f"{total:,}",
            "total_M": f"{total / 1e6:.1f}M",
        }