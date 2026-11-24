# ⚡ MATLAB Error-Fixer: Fine-Tuning Pipeline (0.5B–0.8B)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S-SaiTarun/matlab-fixer/blob/main/notebooks/train_matlab_fixer.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-500%20ChatML%20Examples-blue.svg)](file:///data/matlab_train.jsonl)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--0.5B--Instruct-purple.svg)](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B-Instruct)
[![Hardware](https://img.shields.io/badge/Target%20GPU-Google%20Colab%20T4-green.svg)](https://colab.research.google.com)
[![Quantization](https://img.shields.io/badge/Export-4--bit%20q4__k__m%20GGUF-orange.svg)](#)

A complete, production-ready pipeline for fine-tuning a lightweight, specialized **0.5B–0.8B MATLAB error-fixing LLM** on a free Google Colab T4 GPU, with zero local model weights footprint.

---

## 🔒 Critical Local Constraint

> **Zero Local Weight Download**: No model weights, Hugging Face checkpoints, safetensors, or GGUF files are downloaded to the local machine. All model downloads, 4-bit quantization, LoRA training, and GGUF conversions occur exclusively inside the remote Google Colab VM.

---

## 📁 Repository Structure

```text
matlab-fixer/
├── .env.example                  # Environment configuration template
├── .gitignore                    # Strictly ignores .env, weights, GGUFs, and caches
├── requirements.txt              # Local pipeline dependencies (google-genai, pydantic)
├── generate_dataset.py           # Dataset generator using google-genai SDK & Pydantic
├── validate_dataset.py           # Independent local validator for ChatML JSONL
├── data/
│   └── matlab_train.jsonl        # 500 validated MATLAB examples in ChatML format
└── notebooks/
    └── train_matlab_fixer.ipynb  # End-to-end Colab training & GGUF export notebook
```

---

## 📊 Dataset Breakdown (`data/matlab_train.jsonl`)

The dataset contains **500 high-quality, diverse MATLAB examples** formatted in standard ChatML schema (`messages`: `system`, `user`, `assistant`).

| Category | Description | Count | Share |
|---|---|:---:|:---:|
| **Operator Mismatch** | `*` vs `.*`, `^` vs `.^`, `/` vs `./`, `\` vs `.\` in DSP, plotting, kinematics | 100 | 20.0% |
| **Missing `end`** | Loops (`for`/`while`), conditionals (`if`/`else`), functions, switch blocks | 85 | 17.0% |
| **Indexing & Cell Arrays** | `0`-based & negative indexing, cell array `{}` vs `()` confusion, cell assignments | 85 | 17.0% |
| **Dimension Mismatches** | Inner dimensions in matrix multiplication, horizontal vs vertical concatenation | 90 | 18.0% |
| **Valid MATLAB Code** | Clean, bug-free idiomatic code resulting in **"No errors found"** | 75 | **15.0%** |
| **Common Traps & Syntax** | Variable shadowing built-in functions (`sum`, `mean`), `=` in `if`, unquoted chars | 65 | 13.0% |
| **Total** | | **500** | **100.0%** |

### ChatML Format Example

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert MATLAB debugger and code fixer..."
    },
    {
      "role": "user",
      "content": "The following MATLAB code has an issue...\n```matlab\nt = 0:0.1:10;\ny = 0.5 * 9.8 * t^2;\nplot(t, y);\n```"
    },
    {
      "role": "assistant",
      "content": "### Diagnosis\n**Error Type:** Incorrect use of matrix power `^` on vector `t`.\n\n**Cause:** The `^` operator is matrix exponentiation... Use `.^`.\n\n### Corrected Code\n```matlab\nt = 0:0.1:10;\ny = 0.5 * 9.8 * t.^2;\nplot(t, y);\n```"
    }
  ]
}
```

---

## 🛠️ Local Usage

### 1. Environment Setup

```bash
# Clone this repository
git clone https://github.com/S-SaiTarun/matlab-fixer.git
cd matlab-fixer

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional: Pre-generated 500 examples are already included!)

```bash
# Set your Gemini API key in .env
cp .env.example .env
# Edit .env and insert GEMINI_API_KEY

# Run dataset generation pipeline
python3 generate_dataset.py
```

### 3. Validate Dataset

```bash
python3 validate_dataset.py
```

---

## 🚀 One-Click Training on Google Colab

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S-SaiTarun/matlab-fixer/blob/main/notebooks/train_matlab_fixer.ipynb)

### Colab Notebook Execution Steps:

1. **Hardware Accelerator**: Ensure Runtime is set to **T4 GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
2. **Clone Repo**: Clones `matlab-fixer` into `/content/matlab-fixer` to immediately load `data/matlab_train.jsonl`.
3. **Download Model**: Loads `unsloth/Qwen2.5-Coder-0.5B-Instruct` in 4-bit mode directly inside Colab.
4. **LoRA Fine-Tuning**: Trains LoRA adapters (`r=16`, `lora_alpha=16`, 3 epochs, `max_seq_length=1024`) in ~3–5 minutes.
5. **Interactive Testing**: Runs inference on real MATLAB test prompts (operators, missing `end`, and valid code).
6. **Export GGUF**: Mounts Google Drive and saves the merged 4-bit `q4_k_m` GGUF directly to `/content/drive/MyDrive/matlab-fixer-model/`.
7. **(Optional) Hugging Face Hub**: Pushes the adapter and GGUF to Hugging Face if `HF_TOKEN` is present in Colab Secrets.

---

## 💻 Running the Exported GGUF Locally

Once exported to Google Drive, you can run the model locally using Ollama:

Create a `Modelfile`:
```dockerfile
FROM ./matlab-fixer-0.5B-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
SYSTEM "You are an expert MATLAB debugger and code fixer."
```

Create and run the model:
```bash
ollama create matlab-fixer -f Modelfile
ollama run matlab-fixer
```

---

## 📜 License
Apache-2.0 License.
