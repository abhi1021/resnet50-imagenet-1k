# Upload CPU Checkpoint to HuggingFace Hub

## Quick Instructions

The CPU-optimized checkpoint (`best_model_cpu.pth`) needs to be uploaded to your HuggingFace model repository before deploying the Gradio app.

### Method 1: Using the Upload Script

```bash
# Activate your environment
pyenv activate erav4-sess9

# Upload using your HuggingFace token
python upload_cpu_checkpoint.py --hf-token YOUR_HF_TOKEN --hf-repo pandurangpatil/imagenet1k
```

### Method 2: Using HuggingFace CLI Login

```bash
# Login first (one-time setup)
huggingface-cli login
# Enter your token when prompted

# Then upload
python upload_cpu_checkpoint.py --hf-repo pandurangpatil/imagenet1k
```

### Method 3: Manual Upload via Web Interface

1. Go to https://huggingface.co/pandurangpatil/imagenet1k
2. Click "Files and versions"
3. Click "Add file" → "Upload file"
4. Upload `checkpoint_5/best_model_cpu.pth`
5. Commit with message: "Add CPU-optimized checkpoint (98MB)"

## Checkpoint Details

- **Original**: `checkpoint_5/best_model.pth` (195 MB)
- **CPU-Optimized**: `checkpoint_5/best_model_cpu.pth` (98 MB)
- **Reduction**: 50% size reduction (removed optimizer state)
- **Repository**: pandurangpatil/imagenet1k
- **Target Filename**: `best_model_cpu.pth`

## Verification

After upload, verify the file is available:
```bash
# Check if file exists on HuggingFace
huggingface-cli scan-cache | grep imagenet1k
```

Or visit: https://huggingface.co/pandurangpatil/imagenet1k/tree/main

You should see `best_model_cpu.pth` listed there.

## Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "imagenet1k-upload")
4. Select "Write" access
5. Copy the token

## Next Steps

After uploading the checkpoint:
1. Test the Gradio app locally (see `gradio_app/DEPLOYMENT_INSTRUCTIONS.md`)
2. Deploy to HuggingFace Spaces
3. Share your deployed app!
