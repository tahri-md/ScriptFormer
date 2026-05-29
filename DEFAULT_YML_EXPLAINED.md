# `configs/default.yml` Explained

This file stores the main settings for ScriptFormer. It controls how data is loaded, how images are prepared, how the tokenizer works, how the model is built, and how training runs.

## Important note

The current code in `model/trocr.py` uses a CNN encoder plus a Transformer decoder.
The `model.encoder.*` section in this config looks like an older or future ViT setup and is not used by the current CNN-based code path.
The `model.decoder.*` section is the one that directly matches the current model code.

## `project`

- `name`: The project name used for labels, logs, or experiment naming.
- `seed`: Random seed for reproducibility. Using the same seed helps make runs more consistent.
- `device`: Which device to use for training or inference.
  - `cpu` means run on the processor.
  - `cuda` means run on an NVIDIA GPU if available.

## `data`

- `raw_dir`: Folder where the original KHATT data lives.
- `processed_dir`: Folder where preprocessed data would be stored if you save processed outputs.
- `annotations_file`: Path to a CSV file with image-to-text annotations.
- `train_ratio`: Fraction of the dataset used for training.
- `val_ratio`: Fraction used for validation.
- `test_ratio`: Fraction used for testing.

### `data.image`

- `height`: Final image height after preprocessing.
- `width`: Final image width after preprocessing.
- `channels`: Number of image channels.
  - `1` means grayscale.

These size values matter because the model expects all images to end up with the same shape.

## `preprocessing`

This controls how raw images are cleaned before the model sees them.

### `preprocessing.binarization`

- `method`: Which binarization method to use.
  - `sauvola` is adaptive and often works well for manuscripts.
  - `otsu` is a global thresholding method.
- `window_size`: Local neighborhood size used by Sauvola.
- `k`: Sensitivity value for Sauvola.

### `preprocessing.denoising`

- `enabled`: Turns denoising on or off.
- `method`: Denoising method to use.
  - `morphological`
  - `median`
  - `gaussian`
- `kernel_size`: Size of the denoising kernel.

### `preprocessing.augmentation`

These settings describe training-time image variation to make the model more robust.

- `enabled`: Turns augmentation on or off.
- `rotation_range`: Random rotation amount in degrees.
- `elastic_distortion`: Adds paper-like warping.
- `brightness_range`: Simulates lighter or darker ink.
- `noise_sigma`: Adds noise to simulate grain or scan artifacts.

## `tokenizer`

This section controls how Arabic text is turned into token ids.

- `type`: Tokenizer type.
  - `character` means each character is a token.
  - `bpe` would mean subword tokenization.
- `pad_token`: Token used to pad shorter sequences.
- `sos_token`: Start-of-sequence token.
- `eos_token`: End-of-sequence token.
- `unk_token`: Unknown token for characters not in the vocabulary.

### `tokenizer.normalization`

These flags control how text is normalized before tokenization.

- `normalize_whitespace`: Collapse or clean whitespace.
- `normalize_alef`: Normalize different alef forms to one form.
- `normalize_taa_marbuta`: Normalize taa marbuta if enabled.
- `normalize_alef_maqsura`: Normalize alef maqsura if enabled.
- `remove_diacritics`: Remove Arabic diacritics if enabled.
- `clean_punctuation`: Clean punctuation spacing or variants.
- `strip_non_arabic`: Remove non-Arabic characters if enabled.

## `model`

This section describes model architecture settings.

### `model.encoder`

- `type`: Encoder type described in the config.
- `pretrained`: Whether to start from pretrained weights.
- `pretrained_model`: Name of the pretrained model to use.
- `patch_size`: Patch size used by a ViT-style encoder.
- `hidden_size`: Feature size for each patch or representation.
- `num_layers`: Number of encoder layers.
- `num_heads`: Number of attention heads.
- `dropout`: Dropout rate.

Again, in the current codebase, these encoder keys are not used by the active CNN encoder path.

### `model.decoder`

These settings directly match the Transformer decoder in `model/trocr.py`.

- `type`: Decoder type.
- `hidden_size`: Size of the decoder hidden vectors.
- `num_layers`: Number of Transformer decoder layers.
- `num_heads`: Number of attention heads in each decoder layer.
- `feedforward_size`: Size of the feedforward network inside each layer.
- `dropout`: Dropout rate used in the decoder.
- `max_length`: Maximum token sequence length the decoder can generate or process.

## `training`

These settings control the training loop.

- `epochs`: Number of full passes over the training data.
- `batch_size`: Number of samples per batch.
- `learning_rate`: Step size for optimization.
- `weight_decay`: Regularization strength for AdamW.
- `warmup_steps`: Number of warmup steps before the learning rate schedule fully starts.
- `max_grad_norm`: Maximum gradient norm used for clipping.
- `optimizer`: Optimizer type.
- `scheduler`: Learning rate scheduler type.
- `checkpoint_dir`: Folder where checkpoints are saved.
- `save_every`: Save a checkpoint every N epochs.

### `training.early_stopping`

- `enabled`: Turn early stopping on or off.
- `patience`: How many epochs to wait without improvement before stopping.
- `metric`: Which metric to watch for early stopping.

## `evaluation`

This section controls inference-style evaluation.

- `metrics`: List of metrics to compute.
  - `cer` means Character Error Rate.
  - `wer` means Word Error Rate.
- `beam_size`: Beam search width used during prediction and evaluation.

## `logging`

These settings control logs and debugging output.

- `level`: Logging level such as `INFO`.
- `log_dir`: Folder where logs are stored.
- `tensorboard`: Whether to write TensorBoard logs.
- `log_every`: Print training progress every N steps.
- `preview_samples`: Number of validation examples to show during training.

## Quick summary

If you only remember a few things, remember these:

- `data.image.height` and `data.image.width` decide the final model input size.
- `tokenizer.*` decides how Arabic text becomes ids.
- `model.decoder.*` matches the current decoder architecture.
- `training.*` controls how training runs and where checkpoints are saved.
- `evaluation.beam_size` affects prediction quality and speed.
