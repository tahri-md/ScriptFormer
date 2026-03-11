import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class CNNEncoder(nn.module):
    def __init__(self,hidden_size:int = 256,dropout:float=0.1):
        super().__init__()
        self.block1 = ConvBlock(1,32)
        self.block2 = ConvBlock(32,64)
        self.block3 = ConvBlock(64,128)
        self.block4 = ConvBlock(128,256)
        self.projection = nn.Linear(256*4,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self,images:torch.Tensor)->torch.Tensor:
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        B,C,H,W = x.shape
        x = x.permute(0,3,1,2)
        x = x.reshape(B,W,C*H)
        x = self.projection(x)
        x = self.dropout(x)
        x = self.norm(x)

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self,hidden_size:int,max_length:int=200,dropout:float = 0.1):
        super().__init__()
        self.dropout()= nn.dropout(dropout)
        pe = torch.zeros(max_length,hidden_size)
        position = torch.arange(0,max_length).unsqueeze1().float()
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
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

        self.encoder = CNNEncoder(
            hidden_size=encoder_hidden,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
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
    ) -> torch.Tensor:
        if max_length is None:
            max_length = self.max_length

        B = images.shape[0]
        device = images.device

        encoder_output = self.encoder(images)
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