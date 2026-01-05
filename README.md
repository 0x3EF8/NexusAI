<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey?style=for-the-badge&logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/AI-Local%20LLM-purple?style=for-the-badge" alt="AI">
</p>

<h1 align="center">
  <br>
  🤖 NexusAI
  <br>
</h1>

<h4 align="center">A powerful, privacy-focused local AI programming assistant that runs entirely on your machine.</h4>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-models">Models</a> •
  <a href="#-commands">Commands</a> •
  <a href="#-configuration">Configuration</a>
</p>

---

## ⚠️ Disclaimer

> **This project was made for cheating purposes.** Use responsibly and at your own risk. The developer is not responsible for any misuse of this software.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **100% Local** | All processing happens on your machine - no data leaves your system |
| 🧠 **Multiple AI Models** | Switch between Phi-3, Phi-4, CodeLlama, and Mistral |
| 📁 **File References** | Include file contents with `@filename` syntax |
| 🖼️ **Image Support** | Attach images with `#image.png` syntax |
| 📋 **Smart Paste** | Automatically captures multi-line pasted text |
| 🔐 **Password Protected** | Secure your sessions with authentication |
| 🕵️ **Stealth Mode** | Disguise as a normal Windows terminal |
| ⏱️ **Session Timeout** | Auto-lock after inactivity |
| 📊 **Performance Monitoring** | Track response times and token usage |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Windows 10/11
- 8GB+ RAM (16GB recommended for larger models)
- GPU with CUDA support (optional, for faster inference)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/0x3EF8/NexusAI.git
cd NexusAI

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run NexusAI
python main.py
```

### Pre-built Executable

Download the latest release from [Releases](https://github.com/0x3EF8/NexusAI/releases) - no Python installation required!

---

## 🚀 Usage

### Starting NexusAI

```bash
python main.py
# or run the executable
NexusAI.exe
```

### Default Password

```
Hex
```

### Basic Interaction

```
You: Write a Python function to calculate fibonacci numbers

NexusAI: ```python
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
```

---

## 🤖 Models

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| **Phi-3 Mini** | ~2.4GB | Microsoft's efficient model | Quick responses, logic |
| **Phi-4** | ~9.1GB | State-of-the-art reasoning | Complex problems |
| **CodeLlama 13B** | ~7.4GB | Meta's code specialist | Python code generation |
| **Mistral 7B** | ~4.4GB | Versatile instruction model | Chat, explanations |

Models are downloaded automatically on first use.

---

## 💻 Commands

| Command | Description |
|---------|-------------|
| `help` | Show all available commands |
| `exit` | Safely terminate the program |
| `reset` | Clear conversation history |
| `clear` | Clear the screen |
| `cc` | Copy last code snippet to clipboard |
| `ca` | Copy last response to clipboard |
| `paste` / `v` | Send clipboard content as prompt |
| `model` | Switch AI model |
| `settings` | Edit runtime configuration |
| `stealth` | Toggle stealth mode |

### File References

Include file contents directly in your prompt:

```
You: Review this code @main.py and suggest improvements
You: Compare @file1.py with @file2.py
```

### Image References

Attach images to your prompt:

```
You: What's in this image? #screenshot.png
You: Analyze #"path with spaces.jpg"
```

---

## ⚙️ Configuration

### Runtime Settings

Access via `settings` command:

| Setting | Default | Description |
|---------|---------|-------------|
| Context Length | 4096 | Maximum context window |
| CPU Threads | 8 | Threads for inference |
| GPU Layers | 35 | Layers offloaded to GPU |
| Max Tokens | 4096 | Maximum response length |
| History Limit | Unlimited | Conversation memory |
| Monitoring | Disabled | Show performance stats |
| Stealth Mode | Disabled | Terminal disguise |
| Session Timeout | 5 min | Auto-lock timer |

### Model-Specific Settings

Each model has optimized settings:

```python
# Phi-3: Maximum logic precision
temperature: 0.0
max_tokens: 2048

# Phi-4: High-accuracy reasoning
temperature: 0.1
max_tokens: 8192

# CodeLlama: Code generation
temperature: 0.1
max_tokens: 4096

# Mistral: Conversational
temperature: 0.2
max_tokens: 8192
```

---

## 🏗️ Building from Source

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller build.spec --noconfirm

# Output: dist/NexusAI.exe
```

---

## 📁 Project Structure

```
NexusAI/
├── main.py              # Main application
├── build.spec           # PyInstaller configuration
├── requirements.txt     # Python dependencies
├── models/              # AI models (auto-downloaded)
│   ├── Phi-3-mini-4k-instruct-q4.gguf
│   ├── phi-4-Q4_K_M.gguf
│   ├── codellama-13b-python.Q4_K_S.gguf
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
└── dist/                # Built executable
    └── NexusAI.exe
```

---

## 🔧 Requirements

```
llama-cpp-python>=0.2.0
requests>=2.31.0
tqdm>=4.66.0
pyperclip>=1.8.0
colorama>=0.4.6
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**0x3ef8**

---

<p align="center">
  <i>⭐ Star this repo if you find it useful!</i>
</p>
