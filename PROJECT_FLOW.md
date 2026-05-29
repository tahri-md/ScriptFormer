# ScriptFormer Flow

This file explains the full OCR pipeline in plain terms: what goes into each stage, what comes out, and why each stage exists. It is focused on the CNN encoder, transformer decoder, and the training flow used in this repo.

## One-line summary

ScriptFormer is an image-to-text model:

image -> preprocessing -> CNN encoder -> transformer decoder -> token ids -> text

During training, the model sees an image and the previous target tokens, then learns to predict the next token.

## Main pieces

| Part | Input | Output | Why it exists |
| --- | --- | --- | --- |
| `configs/default.yml` | None | Model, data, preprocessing, and training settings | Keeps all hyperparameters in one place |
| `data/labelparser.py` | KHATT CSV rows | Arabic text samples and image paths | Converts dataset annotations into usable training samples |
| `data/tokenizer.py` | Arabic training text | Character ids and vocabulary file | Converts text into model-friendly ids |
| `preprocessing/transforms.py` | Raw image from disk | Normalized grayscale image tensor | Makes image input consistent for the CNN |
| `data/dataset.py` | Parsed samples + tokenizer + preprocessor | Batched images and token ids | Feeds training and validation batches |
| `model/trocr.py` | Image tensor + token ids | Next-token logits | Core OCR model |
| `training/trainer.py` | Model + dataloaders + config | Checkpoints and metrics | Runs optimization and saves the trained model |
| `inference/pipeline.py` | Checkpoint + image path | Final text prediction | Easy prediction wrapper for testing and deployment |
| `postprocessing/arabic_text.py` | Raw decoded text | Cleaned Arabic text | Makes predictions easier to compare and read |

## Training flow

### 1. Config

File: `configs/default.yml`

Input: nothing, it is loaded first.

Output: the values that control image size, batch size, model size, learning rate, checkpoint path, and preprocessing settings.

Why: training and inference must use the same settings, otherwise the model and tokenizer can drift apart.

### 2. Parse the dataset

File: `data/labelparser.py`

Input: KHATT CSV files and the raw image folder.

Output: samples like `{"filename": ..., "image_path": ..., "text": ...}`.

Why: the model does not train on CSV codes directly; it needs real image paths and Arabic text.

### 3. Build the tokenizer

File: `data/tokenizer.py`

Input: all training and validation texts.

Output: `tokenizer.json` containing `char_to_id`, `id_to_char`, and special tokens like `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.

Why: the decoder predicts token ids, not raw characters. The tokenizer turns text into ids for training and ids back into text for inference.

### 4. Preprocess the images

File: `preprocessing/transforms.py`

Input: raw OpenCV image.

Output: a normalized grayscale tensor of shape roughly `(1, H, W)` after resize/padding and normalization.

Why: the CNN expects a stable image format. Preprocessing reduces noise, standardizes size, and makes training easier.

### 5. Build batches

Files: `data/dataset.py`, `training/trainer.py`, `scripts/train.py`

Input: samples, tokenizer, and preprocessor.

Output: batches with `images` and `token_ids`.

Why: training needs mini-batches so the model can learn efficiently.

## CNN encoder

Code: `CNNEncoder` in `model/trocr.py`

### Input

- A batch of preprocessed images with shape `(B, 1, H, W)`.

### What it does

- Applies 4 convolution blocks.
- Each block does convolution, batch norm, ReLU, and max pooling.
- After the last block, the spatial map is reshaped into a sequence.
- A linear layer projects features to the decoder hidden size.

### Output

- A sequence of visual features with shape roughly `(B, W', hidden_size)`.

### Why

- The encoder converts pixels into visual tokens.
- The transformer decoder cannot read raw images directly; it needs a feature sequence.

### Important detail

- The width dimension becomes the sequence length.
- Height is compressed by pooling.
- The encoder output is the “memory” the decoder attends to.

## Transformer decoder

Code: `TransfomerDecoder` in `model/trocr.py`

### Input

- Encoder output from the CNN.
- Previous target token ids during training.

### What it does

- Embeds token ids into vectors.
- Adds positional encoding.
- Applies a causal mask so the model can only look at previous target tokens.
- Uses transformer decoder layers to attend to the image features.
- Projects hidden states to vocabulary logits.

### Output

- Logits with shape `(B, T, vocab_size)`.

### Why

- This is the text generator.
- It learns to predict the next character one step at a time.

## ScriptFormer model

Code: `ScriptFormer` in `model/trocr.py`

### Forward pass during training

Input:

- Images: `(B, 1, H, W)`
- Decoder input ids: previous tokens shifted right

Output:

- Logits for every target position

Why:

- The model is trained with teacher forcing.
- That means the decoder gets the correct previous tokens, and learns to predict the next one.

### Generation during inference

Input:

- Images only

Output:

- Generated token ids

Why:

- At inference time we do not have the true answer, so the model must generate tokens by itself.
- The code supports greedy decoding and beam search.

## How training works step by step

File: `training/trainer.py`

1. Load a batch.
2. Split `token_ids` into:
   - `decoder_input = token_ids[:, :-1]`
   - `labels = token_ids[:, 1:]`
3. Run `logits = model(images, decoder_input)`.
4. Compute cross-entropy loss.
5. Ignore padding with `ignore_index=tokenizer.pad_id`.
6. Backpropagate.
7. Clip gradients.
8. Update optimizer and scheduler.
9. Validate on the val set.
10. Save the best checkpoint.

### Why this training setup works

- The shifted input/label setup teaches next-token prediction.
- Cross-entropy is the standard loss for token classification.
- Ignoring padding prevents blank tokens from affecting the loss.
- Gradient clipping helps stabilize training.

## What each training input means

### Image input

- Raw image -> preprocessed tensor.
- This is what the CNN learns from.

### Text input

- Ground-truth Arabic text -> tokenizer ids.
- These ids teach the decoder what sequence to produce.

### Special tokens

- `<SOS>` means start of sequence.
- `<EOS>` means end of sequence.
- `<PAD>` fills batch sequences to equal length.

## What the model outputs mean

### During training

- The model outputs logits over the vocabulary for each target position.
- These logits are compared with the correct next token ids.

### During inference

- The model outputs predicted token ids.
- Those ids are decoded back into text and then optionally postprocessed.

## Why predictions can still be bad even if the pipeline runs

The pipeline can be correct and the predictions can still be weak if:

- preprocessing differs from training,
- the checkpoint and tokenizer do not match perfectly,
- the model was undertrained,
- the decoder learned noisy patterns,
- the validation images are harder than the training images.

## Inference flow

File: `inference/pipeline.py`

1. Load checkpoint.
2. Load tokenizer.
3. Build the model using checkpoint config.
4. Preprocess the image.
5. Generate token ids.
6. Decode ids to text.
7. Apply Arabic postprocessing.

## Quick reference: input and output by part

### `labelparser`

- Input: CSV row
- Output: Arabic text + image path
- Why: converts dataset annotations into training data

### `tokenizer`

- Input: Arabic text
- Output: token ids
- Why: the model works with ids

### `preprocessor`

- Input: raw image
- Output: normalized tensor
- Why: standardizes images for the CNN

### `encoder`

- Input: image tensor
- Output: image feature sequence
- Why: extracts visual information

### `decoder`

- Input: image features + previous tokens
- Output: next-token logits
- Why: generates text

### `trainer`

- Input: batches of image tensors and token ids
- Output: trained weights and checkpoints
- Why: updates the model

### `inference pipeline`

- Input: image path
- Output: final Arabic text prediction
- Why: convenient end-to-end prediction

## Files to inspect first when debugging

1. `scripts/train.py` for how training is assembled.
2. `training/trainer.py` for the loss and update step.
3. `model/trocr.py` for encoder and decoder behavior.
4. `preprocessing/transforms.py` for image preparation.
5. `data/tokenizer.py` for text ids and special tokens.
6. `inference/pipeline.py` for actual prediction flow.

## Practical note from this repo

We already verified that:

- the tokenizer round-trip works,
- the code parses KHATT correctly,
- the checkpoint now loads even with the vocab-size mismatch,
- the model produces text, but the quality is still weak.

That means the next debugging targets are preprocessing, generation behavior, and how training matched the saved checkpoint.
