import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocessing import ManuscriptPreprocessor
from .tokenizer import ArabicCharTokenizer

class ArabicOCRDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: ArabicCharTokenizer,
        preprocessor: ManuscriptPreprocessor,
        image_height: int = 64,
        image_width: int = 384,
        max_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.image_height = image_height
        self.image_width = image_width
        self.max_length = max_length
    
    def __len__(self)->int:
        return len(self.samples)
    
    def __getitem__(self,idx:int) -> tuple[torch.Tensor,torch.Tensor,str]:
        sample = self.samples[idx]
        raw_image = cv2.imread(sample["image_path"])

        if raw_image is None:
            raise FileNotFoundError(f"could not load image{sample['image_path']}")
        
        processed = self.preprocessor(raw_image,target_height=self.image_height,target_width=self.image_width)
        image_tensor = torch.from_numpy(processed).unsqueeze(0).float()
        token_ids = self.tokenizer.encode(
            sample["text"],
            add_special_tokens=True,
            max_length=self.max_length,
        )
        token_tensor = torch.tensor(token_ids,dtype=torch.long)
        return image_tensor,token_tensor,sample["text"]
    

def collate_fn(batch:list,pad_id:int = 0) ->dict :
    images,token_seqs,texts = zip(*batch)
    images = torch.stack(images,dim=0)
    max_len =max(seq.shape[0] for seq in token_seqs)
    padded_tokens = []
    attention_masks = []
        
    for seq in token_seqs:
        seq_len = seq.shape[0]
        pad_len = max_len - seq_len

        padded = torch.cat([seq, torch.full((pad_len,), pad_id, dtype=torch.long)])
        padded_tokens.append(padded)

        mask = torch.cat([torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long)])
        attention_masks.append(mask)

    return {
        "images": images,                                  
        "token_ids": torch.stack(padded_tokens, dim=0),    
        "attention_mask": torch.stack(attention_masks, dim=0), 
        "texts": list(texts),                                
    }

def create_dataloaders(
    train_samples: list[dict],
    val_samples: list[dict],
    tokenizer: ArabicCharTokenizer,
    preprocessor: ManuscriptPreprocessor,
    config: dict,
    ) -> tuple[DataLoader, DataLoader]:
    img_h = config["data"]["image"]["height"]
    img_w = config["data"]["image"]["width"]
    max_len = config["model"]["decoder"]["max_length"]
    batch_size = config["training"]["batch_size"]

    train_dataset = ArabicOCRDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        image_height=img_h,
        image_width=img_w,
        max_length=max_len,
    )
    val_dataset = ArabicOCRDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        image_height=img_h,
        image_width=img_w,
        max_length=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        num_workers=2,      
        pin_memory=True,      
        drop_last=True,        
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        num_workers=2,
        pin_memory=True,
    )

    print(f"  [DataLoader] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  [DataLoader] Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader
