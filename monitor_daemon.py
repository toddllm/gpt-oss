#!/usr/bin/env python3
"""
Background daemon that monitors download and tests when complete
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open("monitor_daemon.log", "a") as f:
        f.write(log_message + "\n")

def get_download_status():
    """Check download status"""
    model_dir = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b"
    
    # Get size
    if model_dir.exists():
        result = subprocess.run(['du', '-sb', str(model_dir)], capture_output=True, text=True)
        size_bytes = int(result.stdout.split()[0])
        size_gb = size_bytes / (1024**3)
    else:
        size_gb = 0
    
    # Count incomplete files
    incomplete_files = list(model_dir.rglob("*.incomplete")) if model_dir.exists() else []
    
    # Check if download process is running
    pid_file = Path.home() / "gpt-oss/download.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            download_running = True
        except OSError:
            download_running = False
    else:
        download_running = False
    
    return {
        'size_gb': size_gb,
        'incomplete_count': len(incomplete_files),
        'download_running': download_running,
        'is_complete': size_gb >= 39 and len(incomplete_files) == 0
    }

def run_test():
    """Run the model test"""
    log("Running automatic model test...")
    
    test_code = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai/gpt-oss-20b"

print("Loading model for test...")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

print("Generating test response...")
inputs = tokenizer("Hello world", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Test response: {response}")

# If we get here, test passed
with open("model_ready.flag", "w") as f:
    f.write("READY")
print("✓ Model test successful!")
"""
    
    # Write and run test script
    with open("test_model.py", "w") as f:
        f.write(test_code)
    
    result = subprocess.run([sys.executable, "test_model.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        log("✅ TEST PASSED - Model is working!")
        log(result.stdout)
        return True
    else:
        log("❌ TEST FAILED")
        log(result.stderr)
        return False

def main():
    """Main monitoring loop"""
    log("="*50)
    log("GPT-OSS-20B Download Monitor Started")
    log("="*50)
    
    check_interval = 30  # seconds
    iteration = 0
    
    while True:
        iteration += 1
        status = get_download_status()
        
        log(f"Check #{iteration}: {status['size_gb']:.2f}GB, "
            f"{status['incomplete_count']} incomplete, "
            f"download={'running' if status['download_running'] else 'stopped'}")
        
        if status['is_complete']:
            log("🎉 DOWNLOAD COMPLETE!")
            
            # Wait a bit for files to settle
            time.sleep(5)
            
            # Run test
            if run_test():
                log("="*50)
                log("SUCCESS - Model ready for use!")
                log("Run: python run_after_download.py")
                log("="*50)
                
                # Create notification file
                with open("DOWNLOAD_COMPLETE.txt", "w") as f:
                    f.write(f"Download completed at {datetime.now()}\n")
                    f.write("Model tested and ready!\n")
                
                break
            else:
                log("Test failed but download is complete")
                break
        
        # Sleep before next check
        time.sleep(check_interval)
    
    log("Monitor exiting")

if __name__ == "__main__":
    # Fork to background
    if '--daemon' in sys.argv:
        pid = os.fork()
        if pid > 0:
            print(f"Monitor daemon started (PID: {pid})")
            sys.exit(0)
    
    main()