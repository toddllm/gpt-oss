#!/usr/bin/env python3
"""
Download GPT-OSS-20B model files with progress tracking
"""

from huggingface_hub import snapshot_download
import os

model_id = "openai/gpt-oss-20b"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print("=" * 60)
print("Downloading GPT-OSS-20B Model Files")
print("=" * 60)
print(f"Model: {model_id}")
print(f"Cache directory: {cache_dir}")
print("\nThis will download approximately 40GB of data.")
print("The download will resume if interrupted.\n")

try:
    # Download with progress bars
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,  # Resume if interrupted
        local_files_only=False,
    )
    
    print(f"\n✓ Model downloaded successfully to: {local_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > 1:  # Only show files > 1MB
                print(f"  {file}: {size_mb:.1f} MB")
    
except KeyboardInterrupt:
    print("\n\nDownload interrupted. Run this script again to resume.")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Check disk space: df -h")
    print("3. Clear incomplete downloads: rm -rf ~/.cache/huggingface/hub/models--openai--gpt-oss-20b")
    print("4. Try again with: python download_model.py")