# Android Termux: GPT‑OSS Setup (Working Path + Usability Tweaks)

This doc captures the **exact path that worked on Android (Termux)** for running the **GPT‑OSS‑20B** GGUF model, plus quality‑of‑life tweaks (env vars, aliases, and a small runner script) so you don’t have to remember long commands.

> **Tested environment:** Termux (Android), aarch64 CPU, shared storage enabled (`termux-setup-storage`), `llama.cpp` built from source with **CMake** and **K‑quant** support.

---

## 0) Quick overview

* Build `llama.cpp` from source with **K‑quant** support.
* Ensure backend plug‑ins (`libggml-*.so`) are discoverable.
* Download the **Q4\_K\_M** GGUF file into **shared storage** (e.g., `~/storage/downloads`).
* Run via `llama-cli` (interactive) or `llama-run` (one‑shot) without needing to press Enter.
* Add `~/.bashrc` exports + aliases, and an optional wrapper script.

---

## 1) Prerequisites

```bash
pkg update && pkg upgrade
pkg install git build-essential cmake ninja ndk-sysroot aria2
termux-setup-storage
```

> `termux-setup-storage` creates the `~/storage` symlinks to shared folders (e.g., `~/storage/downloads` → `/storage/emulated/0/Download`).

---

## 2) Clone & build `llama.cpp` (CMake only, K‑quants ON)

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DLLAMA_NATIVE=ON -DLLAMA_K_QUANTS=ON
cmake --build build --config Release -j$(nproc)
```

**Make the backend plug‑ins discoverable** (pick one):

**Option A – Create the `../lib` path that the binaries expect**

```bash
mkdir -p build/lib
ln -s $PWD/build/bin/libggml-*.so $PWD/build/lib/
```

**Option B – Environment variables (per session or via `~/.bashrc`)**

```bash
export GGML_BACKEND_PATH=$HOME/llama.cpp/build/bin
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

> You can also run `cmake --install build --prefix $HOME/llama.cpp/build/install` and then execute from `build/install/bin/`, but ensure `bin/../lib` exists or use the env vars above.

---

## 3) Download the model to shared storage (resumable)

```bash
cd ~/storage/downloads
aria2c -x16 -s16 -c --file-allocation=none \
  -o gpt-oss-20b-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf
```

**Verify**

```bash
ls -lh gpt-oss-20b-Q4_K_M.gguf     # ~10.7G
file gpt-oss-20b-Q4_K_M.gguf       # should say: GGUF
```

> If space is tight, consider `Q3_K_L` (\~6G) or `Q4_0` (\~8G). K‑quants (Q**K**) usually give better quality/speed tradeoffs.

---

## 4) One‑shot generation (no Enter needed)

**Using `llama-run` (preferred for a single prompt):**

```bash
# Flags → Model → Prompt
export GGML_BACKEND_PATH=$HOME/llama.cpp/build/bin
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
~/llama.cpp/build/bin/llama-run \
  -n 256 --threads $(nproc) --context-size 4096 \
  /storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf \
  "Explain physics to a 7-year-old."
```

**Using `llama-cli` with prompt at launch (still one‑shot):**

```bash
export GGML_BACKEND_PATH=$HOME/llama.cpp/build/bin
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
~/llama.cpp/build/bin/llama-cli \
  -m /storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf \
  --ctx-size 4096 --threads $(nproc) \
  -p "Explain physics to a 7-year-old." \
  -n 256
```

> `-p` (or `--prompt`) avoids interactive typing. `-n` controls token budget; omit it to let the model generate until a stop token.

---

## 5) Make it easy: `~/.bashrc` snippet

Append to `~/.bashrc`:

```bash
# llama.cpp backends in this build directory
export GGML_BACKEND_PATH="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH"

# convenient aliases
alias llama-run-oss='~/llama.cpp/build/bin/llama-run \
  -n 256 --threads $(nproc) --context-size 4096 \
  /storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf'

alias llama-cli-oss='~/llama.cpp/build/bin/llama-cli \
  -m /storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf \
  --ctx-size 4096 --threads $(nproc)'
```

Reload the shell or run `source ~/.bashrc` once.

**Use it:**

```bash
llama-run-oss "Summarize this article in 5 bullet points."
llama-cli-oss -p "Write a haiku about rain." -n 128
```

---

## 6) Optional wrapper script (`~/bin/llama-oss`)

This script lets you pass a prompt without remembering flags.

**Create file:** `~/bin/llama-oss` (and `chmod +x ~/bin/llama-oss`)

```bash
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Defaults (override with env vars if desired)
MODEL_PATH="/storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf"
THREADS="$(nproc)"
CTX=4096
TOKENS=256

export GGML_BACKEND_PATH="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

PROMPT="$*"

if [ -z "$PROMPT" ]; then
  echo "Usage: llama-oss <your prompt>"
  exit 1
fi

exec "$HOME/llama.cpp/build/bin/llama-run" \
  -n "$TOKENS" --threads "$THREADS" --context-size "$CTX" \
  "$MODEL_PATH" \
  "$PROMPT"
```

**Use it:**

```bash
llama-oss "Explain quantum tunneling for kids."
```

If you prefer interactive sessions, make a `llama-chat` twin that invokes `llama-cli` with `-p "$PROMPT"`.

---

## 7) Common problems & fixes

### A) "no backends are loaded"

* Ensure the plug‑ins are visible:

  * `export GGML_BACKEND_PATH=$HOME/llama.cpp/build/bin`
  * `export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH`
* Or ensure `build/bin/libggml-*.so` also exists at `build/lib/` (the `bin/../lib` path).

### B) "invalid ggml type …" on Q**K** models

* Use a **fresh build** with `-DLLAMA_K_QUANTS=ON`. Older packaged binaries often lack K‑quant support.

### C) Download fails with `fallocate` or "No space left on device"

* Add `--file-allocation=none` to `aria2c`.
* Free up space or download to shared storage or an external USB drive.

### D) It stops mid‑answer and waits for Enter

* You likely used `llama-cli` interactively with a small `-n`.
* Fix: add `-p "…" -n 256` for one‑shot, or use `llama-run`.

### E) RAM tight / OOM

* Lower `--ctx-size` (e.g., 2048).
* Try a smaller quant (e.g., `Q3_K_L`).

---

## 8) Nice‑to‑have: PATH and model aliasing

Add to `~/.bashrc`:

```bash
export PATH="$HOME/llama.cpp/build/bin:$PATH"
MODEL_OSS="/storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf"
```

Then you can shorten commands to:

```bash
llama-run -n 256 --threads $(nproc) --context-size 4096 "$MODEL_OSS" "Write a bedtime story about robots."
```

---

## 9) Performance notes

* On recent flagship phones, Q4\_K\_M can yield \~3–5 tok/s.
* Keep the phone cool; long runs throttle. Consider fewer threads if it overheats.
* Bigger `--ctx-size` increases RAM usage.

---

## 10) Quick reference (copy‑paste)

```bash
# Build
pkg update && pkg upgrade
pkg install git build-essential cmake ninja ndk-sysroot aria2
termux-setup-storage
cd ~ && git clone https://github.com/ggerganov/llama.cpp.git
cd ~/llama.cpp && cmake -S . -B build -DLLAMA_NATIVE=ON -DLLAMA_K_QUANTS=ON
cmake --build build --config Release -j$(nproc)
mkdir -p build/lib && ln -s $PWD/build/bin/libggml-*.so $PWD/build/lib/

# Download
cd ~/storage/downloads
aria2c -x16 -s16 -c --file-allocation=none \
  -o gpt-oss-20b-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf

# Run (one‑shot)
export GGML_BACKEND_PATH=$HOME/llama.cpp/build/bin
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
~/llama.cpp/build/bin/llama-run -n 256 --threads $(nproc) --context-size 4096 \
  /storage/emulated/0/Download/gpt-oss-20b-Q4_K_M.gguf \
  "Explain physics to a 7-year-old."
```

---

**That’s it!** This is the working path you used, now with sensible defaults and shortcuts. We can iterate here—add screenshots, swap model names, or bake in different presets (coding, roleplay, etc.).