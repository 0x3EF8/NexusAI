#!/usr/bin/env python3
"""
NexusAI - Local AI Programming Assistant
=========================================
A powerful, privacy-focused AI assistant that runs entirely on your local machine.
No data leaves your system - complete privacy and security.

Developer: 0x3ef8
License: MIT
"""

__version__ = "1.0.0"
__author__ = "0x3ef8"
__description__ = "Local AI Programming Assistant"

# Standard Library
import os
import sys
import io
import re
import time
import base64
import hashlib
import getpass
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable

# Third Party
import requests
from tqdm import tqdm
from llama_cpp import Llama
import gc
import pyperclip
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Determine base path (works for both script and exe)
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Universal system prompt for all models
SYSTEM_PROMPT = """
Expert Programmer. Execute with these strict constraints:
1. CODE FIRST: Output complete, runnable code blocks before any text.
2. CLEAN IMPORTS: Place ALL required imports at the top. DO NOT import libraries that are not used in the code.
3. NO STUBS: Do not use placeholders, "TODO", or "# rest of code here."
4. ROBUST: Use direct variable names and professional try-except error handling.
5. BRIEF NOTES: Provide a max 2-sentence technical summary only after the code.
""".strip()

SYSTEM_PROMPTS = {
    "phi3": SYSTEM_PROMPT,
    "phi4": SYSTEM_PROMPT,
    "codellama": SYSTEM_PROMPT,
    "mistral": SYSTEM_PROMPT
}

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

PASSWORD_HASH = hashlib.sha256("Hex".encode()).hexdigest()
MAX_LOGIN_ATTEMPTS = 3

# =============================================================================
# COLOR SCHEME
# =============================================================================

class Colors:
    """Color scheme for NexusAI interface."""
    PRIMARY = Fore.CYAN
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    INFO = Fore.BLUE
    ACCENT = Fore.MAGENTA
    MUTED = Fore.LIGHTBLACK_EX
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS: Dict[str, Dict[str, Any]] = {
    "phi3": {
        "name": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Microsoft Phi-3 Mini - Optimized for maximum logic precision",
        "template": {
            "system": "<|system|>\n{system}<|end|>\n",
            "user": "<|user|>\n{prompt}<|end|>\n",
            "assistant": "<|assistant|>\n",
            "stop": ["<|end|>", "<|user|>", "<|endoftext|>"]
        },
        "settings": {
            "temperature": 0.0,      # Microsoft recommendation for 100% logic
            "top_p": 0.9,
            "repeat_penalty": 1.1,   # Prevents "stuttering" in code blocks
            "max_tokens": 2048       # Keep window tight for 4K context
        }
    },
    "phi4": {
        "name": "phi-4-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf",
        "description": "Phi-4 14B - State-of-the-art reasoning (9.1GB RAM required)",
        "template": {
            "system": "<|im_start|>system<|im_sep|>\n{system}<|im_end|>\n",
            "user": "<|im_start|>user<|im_sep|>\n{prompt}<|im_end|>\n",
            "assistant": "<|im_start|>assistant<|im_sep|>\n",
            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        },
        "settings": {
            "temperature": 0.1,      # Low temp for high-accuracy reasoning
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "max_tokens": 8192       # Phi-4 handles long context (16K native)
        }
    },
    "codellama": {
        "name": "codellama-13b-python.Q4_K_S.gguf",
        "url": "https://huggingface.co/TheBloke/CodeLlama-13B-Python-GGUF/resolve/main/codellama-13b-python.Q4_K_S.gguf",
        "description": "Meta CodeLlama 13B Python - Specialized for Python code generation",
        "template": {
            "system": "",
            "user": "{prompt}",
            "assistant": "",
            "stop": ["<EOT>", "</s>", "\n\n\n", "# Task:"]
        },
        "settings": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 4096
        }
    },
    "mistral": {
        "name": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B v0.2 - Instruct model for chat and teaching",
        "template": {
            "system": "",
            "user": "<s>[INST] {prompt} [/INST]",
            "assistant": "",
            "stop": ["[INST]", "</s>", "<s>"]
        },
        "settings": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "max_tokens": 8192
        }
    }
}

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

MODEL_DIRECTORY = os.path.join(BASE_DIR, "models")

# Performance Settings
MAX_SEQUENCE_LENGTH: int = 4096      # Maximum context length for the AI
NUM_THREADS: int = 8                  # Number of CPU threads allocated
GPU_LAYERS: int = 35                  # Layers offloaded to GPU
MAX_TOKENS: int = 4096                # Maximum tokens per response

# Feature Toggles
MAX_HISTORY_SIZE: int | bool = False  # Conversation history limit (False = unlimited)
ENABLE_MONITORING: bool = False       # Show response time and token stats
STEALTH_MODE: bool = False            # Disguise as normal terminal
SESSION_TIMEOUT: int | bool = 5       # Auto-lock timeout in minutes (False = disabled)

# =============================================================================
# NEXUSAI CORE CLASS
# =============================================================================

class NexusAI:
    """
    NexusAI Core Assistant Class.
    
    A local AI programming assistant that provides intelligent code assistance
    using locally-hosted language models. Supports multiple models, file/image
    references, and maintains conversation context.
    
    Attributes:
        history: Conversation history for context retention
        num_requests: Total number of requests processed
        total_tokens: Cumulative token count across all responses
        model_directory: Path to model storage directory
        current_model: Currently loaded model identifier
        model: Loaded Llama model instance
    """
    
    def __init__(self, model_directory: str) -> None:
        """Initialize NexusAI with the specified model directory."""
        self.history: List[str] = []
        self.num_requests: int = 0
        self.total_tokens: int = 0
        self.model_directory: str = model_directory
        self.current_model: Optional[str] = None
        self.model: Optional[Llama] = None
        self.stop_event: threading.Event = threading.Event()

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            self._print_error(f"Error reading image '{image_path}': {e}")
            return None

    def _get_image_mime_type(self, image_path: str) -> str:
        """Determine MIME type based on file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp"
        }
        return mime_types.get(ext, "image/jpeg")
    
    @staticmethod
    def _print_status(message: str) -> None:
        """Print a status message with appropriate formatting."""
        if STEALTH_MODE:
            print(message)
        else:
            print(f"{Colors.INFO}[NexusAI]{Colors.RESET} {message}")
    
    @staticmethod
    def _print_success(message: str) -> None:
        """Print a success message with green formatting."""
        if STEALTH_MODE:
            print(message)
        else:
            print(f"{Colors.SUCCESS}[✓]{Colors.RESET} {message}")
    
    @staticmethod
    def _print_error(message: str) -> None:
        """Print an error message with red formatting."""
        if STEALTH_MODE:
            print(f"Error: {message}")
        else:
            print(f"{Colors.ERROR}[✗]{Colors.RESET} {message}")
    
    @staticmethod
    def _print_warning(message: str) -> None:
        """Print a warning message with yellow formatting."""
        if STEALTH_MODE:
            print(message)
        else:
            print(f"{Colors.WARNING}[!]{Colors.RESET} {message}")

    def process_image_references(self, user_input: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process #image references in user input.
        
        Supports both direct references (#image.png) and quoted paths (#"my image.jpg").
        
        Args:
            user_input: Raw user input containing potential image references
            
        Returns:
            Tuple of (processed_input, list of image data dictionaries)
        """
        pattern = r'#"([^"]+)"|#([\w./\\:-]+\.(?:jpg|jpeg|png|gif|webp|bmp))'
        matches = re.findall(pattern, user_input, re.IGNORECASE)
        
        images: List[Dict[str, str]] = []
        processed_input = user_input
        
        for match in matches:
            filename = match[0] if match[0] else match[1]
            original_ref = f'#"{filename}"' if match[0] else f'#{filename}'
            
            file_path = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                image_data = self._encode_image(file_path)
                if image_data:
                    images.append({
                        "data": image_data,
                        "mime_type": self._get_image_mime_type(file_path),
                        "filename": filename
                    })
                    processed_input = processed_input.replace(original_ref, f"[Image: {filename}]")
                    self._print_status(f"Image loaded: {filename}")
            else:
                self._print_error(f"Image not found: {filename}")
        
        return processed_input, images

    def _ensure_model_exists(self, model_key: str) -> None:
        """
        Ensure the specified model exists locally, downloading if necessary.
        
        Args:
            model_key: Key identifier for the model in MODELS dict
        """
        model_info = MODELS[model_key]
        model_path = os.path.join(self.model_directory, model_info["name"])
        
        if os.path.exists(model_path):
            return

        if STEALTH_MODE:
            print("Downloading required components...")
        else:
            print(f"\n{Colors.WARNING}[!]{Colors.RESET} Model not found locally.")
            print(f"{Colors.INFO}[NexusAI]{Colors.RESET} Downloading: {Colors.ACCENT}{model_info['name']}{Colors.RESET}")
        
        os.makedirs(self.model_directory, exist_ok=True)

        try:
            headers = {}
            current_size = 0
            if os.path.exists(model_path):
                current_size = os.path.getsize(model_path)
                headers['Range'] = f"bytes={current_size}-"

            with requests.get(model_info["url"], stream=True, headers=headers, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0)) + current_size
                
                with open(model_path, "ab") as model_file:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, 
                        initial=current_size, desc="Progress",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
                    ) as progress_bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            model_file.write(chunk)
                            progress_bar.update(len(chunk))

            self._print_success(f"Model downloaded successfully.")
        except requests.RequestException as e:
            raise Exception(f"Download failed: {e}")

    def load_model(self, model_key: str) -> None:
        """
        Load an AI model into memory for inference.
        
        Args:
            model_key: Key identifier for the model to load
            
        Raises:
            ValueError: If model_key is not found in MODELS
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: '{model_key}'. Available: {list(MODELS.keys())}")
        
        # Unload previous model to free memory
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()  # Force garbage collection to free GPU/CPU memory
        
        self._ensure_model_exists(model_key)
        model_path = os.path.join(self.model_directory, MODELS[model_key]["name"])
        
        if not STEALTH_MODE:
            print(f"\n{Colors.INFO}[NexusAI]{Colors.RESET} Loading model...")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=MAX_SEQUENCE_LENGTH,
            n_threads=NUM_THREADS,
            n_gpu_layers=GPU_LAYERS,
            verbose=False
        )
        self.current_model = model_key
        
        if not STEALTH_MODE:
            self._print_success("Model loaded and ready.")

    def generate_response(self, prompt: str) -> str:
        """
        Generate an AI response to the given prompt.
        
        Uses model-specific prompting strategies:
        - Phi-3: Uses <|system|> tag with full system prompt
        - CodeLlama: Code completion style with embedded instructions
        - DeepSeek: No system prompt, instructions in user message, uses <think> tags
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Generated response text
        """
        if not self.model or not self.current_model:
            return "No model loaded. Please select a model first."

        self.num_requests += 1
        start_time = time.time()
        model_config = MODELS[self.current_model]
        template = model_config["template"]

        try:
            sys.stderr = io.StringIO()
            
            # Get model-specific system prompt
            system_prompt = SYSTEM_PROMPTS.get(self.current_model, SYSTEM_PROMPT)
            
            # Build prompt based on model type
            if not self.history:
                # First message - include system context appropriately per model
                if self.current_model == "phi3":
                    # Phi-3 uses explicit <|system|> tag
                    system_part = template.get("system", "").format(system=system_prompt)
                    combined_prompt = prompt
                    prefix = system_part
                elif self.current_model == "phi4":
                    # Phi-4 uses <|im_start|> format with system tag
                    system_part = template.get("system", "").format(system=system_prompt)
                    combined_prompt = prompt
                    prefix = system_part
                elif self.current_model == "codellama":
                    # CodeLlama-Python: completion style with docstring instructions
                    combined_prompt = f'"""\n{system_prompt}\n"""\n{prompt}'
                    prefix = ""
                elif self.current_model == "mistral":
                    # Mistral: embed system prompt in first user message
                    combined_prompt = f"{system_prompt}\n\n{prompt}"
                    prefix = ""
                else:
                    combined_prompt = f"[System: {system_prompt}]\n\n{prompt}"
                    prefix = ""
            else:
                combined_prompt = prompt
                prefix = ""
            
            user_message = template["user"].format(prompt=combined_prompt)
            self.history.append(user_message)

            # Manage history size
            if MAX_HISTORY_SIZE and len(self.history) > MAX_HISTORY_SIZE * 2:
                self.history = self.history[-(MAX_HISTORY_SIZE * 2):]

            # Show loading animation
            self.stop_event.clear()
            spinner_thread = threading.Thread(target=self._show_loading_animation, daemon=True)
            if not STEALTH_MODE:
                spinner_thread.start()

            # Build prompt and generate
            if prefix and self.history and not self.history[0].startswith(prefix):
                full_prompt = f"{prefix}{''.join(self.history)}{template['assistant']}"
            elif prefix and not self.history:
                full_prompt = f"{prefix}{template['assistant']}"
            else:
                full_prompt = f"{''.join(self.history)}{template['assistant']}"
            stop_tokens = template["stop"]
            
            # Get model-specific settings
            settings = model_config.get("settings", {})
            
            response = self.model(
                full_prompt,
                max_tokens=settings.get("max_tokens", MAX_TOKENS),
                stop=stop_tokens,
                echo=False,
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.9),
                repeat_penalty=settings.get("repeat_penalty", 1.1),
                top_k=settings.get("top_k", 40)
            )

            response_text = response['choices'][0]['text'].strip()
            self.total_tokens += len(response_text.split())
            self.history.append(f"{template['assistant']}{response_text}")

            # Stop spinner
            self.stop_event.set()
            if not STEALTH_MODE and spinner_thread.is_alive():
                spinner_thread.join(timeout=1.0)

            end_time = time.time()
            response_time = end_time - start_time

            # Display response
            if STEALTH_MODE:
                print(f"\n{response_text}\n")
            else:
                print(f"\n{Colors.PRIMARY}{Colors.BOLD}NexusAI:{Colors.RESET} {response_text}\n")

            # Show monitoring stats if enabled
            if ENABLE_MONITORING and not STEALTH_MODE:
                print(f"{Colors.MUTED}─" * 50)
                print(f"  Response Time: {response_time:.2f}s | Requests: {self.num_requests} | Tokens: {self.total_tokens}")
                print(f"─" * 50 + f"{Colors.RESET}")

            return response_text

        except Exception as e:
            self.stop_event.set()
            self._print_error(f"Generation failed: {e}")
            return f"Error: {e}"

        finally:
            sys.stderr = sys.__stderr__

    def reset_conversation(self) -> str:
        """Reset conversation history and clear the screen."""
        os.system("cls" if os.name == "nt" else "clear")
        self.history = []
        
        if STEALTH_MODE:
            show_stealth_banner()
            return ""
        else:
            show_ascii(self.current_model)
            self._print_success("Conversation history cleared.")
            return ""

    def show_help(self) -> str:
        """Display help information with available commands."""
        if STEALTH_MODE:
            return (
                "\nCommands: exit, reset, clear, cc, ca, paste/v, model, settings, stealth\n"
                "File: @filename | Image: #image.png\n"
            )
        
        timeout_info = f"{SESSION_TIMEOUT} min" if SESSION_TIMEOUT else "Disabled"
        
        return f"""
{Colors.PRIMARY}{Colors.BOLD}╔{'═' * 48}╗
║{'NexusAI Help Center':^48}║
╚{'═' * 48}╝{Colors.RESET}

{Colors.ACCENT}Commands:{Colors.RESET}
  {Colors.SUCCESS}exit{Colors.RESET}      Safely terminate the program
  {Colors.SUCCESS}reset{Colors.RESET}     Clear conversation history
  {Colors.SUCCESS}clear{Colors.RESET}     Clear the screen
  {Colors.SUCCESS}cc{Colors.RESET}        Copy last code snippet to clipboard
  {Colors.SUCCESS}ca{Colors.RESET}        Copy last response to clipboard
  {Colors.SUCCESS}paste/v{Colors.RESET}   Send clipboard content as prompt (for multi-line text)
  {Colors.SUCCESS}model{Colors.RESET}     Switch AI model
  {Colors.SUCCESS}settings{Colors.RESET}  Edit runtime configuration
  {Colors.SUCCESS}stealth{Colors.RESET}   Toggle stealth mode

{Colors.ACCENT}File Input (@):{Colors.RESET}
  Include file contents: {Colors.MUTED}@main.py{Colors.RESET} or {Colors.MUTED}@"path with spaces.py"{Colors.RESET}

{Colors.ACCENT}Image Input (#):{Colors.RESET}
  Attach images: {Colors.MUTED}#photo.png{Colors.RESET} or {Colors.MUTED}#"my image.jpg"{Colors.RESET}

{Colors.MUTED}────────────────────────────────────────────────
Session Timeout: {timeout_info} | Version: {__version__}
Developed by: {__author__}{Colors.RESET}
"""

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def process_command(self, command: str) -> None:
        """Handle clipboard commands (cc/ca/paste)."""
        cmd = command.lower()
        
        if cmd == "paste":
            # Read from clipboard and return as prompt
            try:
                clipboard_content = pyperclip.paste()
                if clipboard_content:
                    # Replace newlines with spaces for single-line processing
                    clean_content = ' '.join(clipboard_content.strip().split())
                    self._print_status(f"Clipboard content ({len(clean_content)} chars) loaded.")
                    return clean_content
                else:
                    self._print_warning("Clipboard is empty.")
                    return None
            except Exception as e:
                self._print_error(f"Failed to read clipboard: {e}")
                return None
        
        if not self.history:
            self._print_warning("No conversation history available.")
            return None
        
        if cmd == "cc":
            code = self._extract_code(self.history[-1])
            if code:
                pyperclip.copy(code)
                self._print_success("Code snippet copied to clipboard.")
            else:
                self._print_warning("No code block found in the latest response.")
        elif cmd == "ca":
            pyperclip.copy(self.history[-1])
            self._print_success("Response copied to clipboard.")
        
        return None

    def _extract_code(self, response: str) -> str:
        """Extract code blocks from AI response."""
        # Try to find fenced code blocks first (```python or ```)
        match = re.search(r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for indented code blocks (4 spaces or tab)
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line[4:] if line.startswith('    ') else line[1:])
                in_code = True
            elif in_code and line.strip() == '':
                code_lines.append('')
            elif in_code:
                break
        
        return '\n'.join(code_lines).strip() if code_lines else ""

    def _show_loading_animation(self) -> None:
        """Display a spinner animation while processing."""
        spinner = ["⠋", "⠙", "⠸", "⠰", "⠴", "⠦", "⠇", "⠏"]
        idx = 0
        while not self.stop_event.is_set():
            print(f"\r{Colors.PRIMARY}Processing {spinner[idx]}{Colors.RESET}", end="", flush=True)
            idx = (idx + 1) % len(spinner)
            time.sleep(0.1)
        print("\r" + " " * 20 + "\r", end="", flush=True)

    def process_file_references(self, user_input: str) -> str:
        """
        Process @filename references in user input and include file contents.
        
        Supports both direct references (@file.py) and quoted paths (@"my file.py").
        
        Args:
            user_input: Raw user input containing potential file references
            
        Returns:
            Processed input with file contents embedded
        """
        pattern = r'@"([^"]+)"|@([\w./\\:-]+)'
        matches = re.findall(pattern, user_input)
        
        if not matches:
            return user_input
        
        processed_input = user_input
        
        for match in matches:
            filename = match[0] if match[0] else match[1]
            original_ref = f'@"{filename}"' if match[0] else f'@{filename}'
            file_path = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    ext = os.path.splitext(filename)[1].lstrip('.') or 'text'
                    file_block = f"\n```{ext}\n# File: {filename}\n{content}\n```\n"
                    processed_input = processed_input.replace(original_ref, file_block)
                    self._print_status(f"File loaded: {filename} ({len(content):,} bytes)")
                except Exception as e:
                    self._print_error(f"Error reading '{filename}': {e}")
            else:
                self._print_error(f"File not found: {filename}")
        
        return processed_input

# =============================================================================
# UI FUNCTIONS
# =============================================================================

def get_fake_path() -> str:
    """Return current working directory for stealth mode prompt."""
    return os.getcwd()


def show_stealth_banner() -> None:
    """Display a fake Windows command prompt banner."""
    print("Microsoft Windows [Version 10.0.22631.4460]")
    print("(c) Microsoft Corporation. All rights reserved.")
    print()

def show_ascii(model_name: Optional[str] = None) -> None:
    """Display the NexusAI ASCII art banner."""
    if STEALTH_MODE:
        show_stealth_banner()
        return
    
    # Large ASCII art banner
    ascii_banner = f"""{Colors.PRIMARY}{Colors.BOLD}
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗     █████╗ ██╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██╔══██╗██║
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ███████║██║
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██╔══██║██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ██║  ██║██║
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝
{Colors.RESET}{Colors.MUTED}═══════════════════════════════════════════════════════════
 {Colors.ACCENT}Version:{Colors.RESET} {__version__}                              {Colors.ACCENT}Dev:{Colors.RESET} {__author__}{Colors.MUTED}"""
    
    print(ascii_banner)
    
    if model_name and model_name in MODELS:
        model_info = MODELS[model_name]
        desc = model_info.get('description', model_info['name'])[:50]
        print(f" {Colors.ACCENT}Model:{Colors.RESET} {Colors.SUCCESS}{model_name.upper()}{Colors.RESET} - {Colors.MUTED}{desc}{Colors.RESET}")
    
    print(f"{Colors.MUTED}═══════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"\n  Type {Colors.SUCCESS}'help'{Colors.RESET} for commands or {Colors.SUCCESS}'exit'{Colors.RESET} to quit.\n")

# =============================================================================
# SETTINGS MANAGEMENT
# =============================================================================

def edit_model_settings(model_key: str) -> None:
    """Edit settings for a specific model."""
    if model_key not in MODELS:
        return
    
    model = MODELS[model_key]
    settings = model.get("settings", {})
    
    model_settings = {
        "1": ("temperature", "Temperature", float, 0.0, 2.0),
        "2": ("top_p", "Top P", float, 0.0, 1.0),
        "3": ("top_k", "Top K", int, 1, 100),
        "4": ("repeat_penalty", "Repeat Penalty", float, 1.0, 2.0),
        "5": ("max_tokens", "Max Tokens", int, 256, 8192),
    }
    
    while True:
        if STEALTH_MODE:
            print(f"\n--- {model_key} Settings ---")
        else:
            print(f"\n{Colors.PRIMARY}╔{'═' * 48}╗")
            print(f"║{Colors.BOLD}{f'{model_key.upper()} Model Settings':^48}{Colors.RESET}{Colors.PRIMARY}║")
            print(f"╚{'═' * 48}╝{Colors.RESET}")
        
        for key, (setting_name, desc, val_type, min_val, max_val) in model_settings.items():
            current = settings.get(setting_name, "default")
            if STEALTH_MODE:
                print(f"  [{key}] {desc}: {current}")
            else:
                print(f"  {Colors.SUCCESS}{key}{Colors.RESET}. {desc:<18} {Colors.ACCENT}{current}{Colors.RESET}")
        
        print(f"\n  {Colors.MUTED}0{Colors.RESET}. Back" if not STEALTH_MODE else "  [0] Back")
        print()
        
        choice = input(f"{Colors.INFO}Select [1-5]:{Colors.RESET} " if not STEALTH_MODE else "Edit: ").strip()
        
        if choice == "0" or choice.lower() in ["back", "exit", "q", "b"]:
            break
        
        if choice not in model_settings:
            print(f"{Colors.ERROR}Invalid selection.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")
            continue
        
        setting_name, desc, val_type, min_val, max_val = model_settings[choice]
        current = settings.get(setting_name, "N/A")
        
        if not STEALTH_MODE:
            print(f"\n  Current: {Colors.ACCENT}{current}{Colors.RESET}")
            print(f"  Range: {Colors.MUTED}{min_val} - {max_val}{Colors.RESET}")
        else:
            print(f"Current: {current} (Range: {min_val}-{max_val})")
        
        inp = input(f"  New value ({Colors.MUTED}'back' to cancel{Colors.RESET}): " if not STEALTH_MODE else "New value: ").strip()
        
        if inp.lower() in ["back", "b", "cancel", "c", "q", "0"]:
            continue
        
        try:
            new_val = val_type(inp)
            if new_val < min_val or new_val > max_val:
                print(f"  {Colors.ERROR}Value must be between {min_val} and {max_val}.{Colors.RESET}" if not STEALTH_MODE else f"Must be {min_val}-{max_val}.")
                continue
            
            settings[setting_name] = new_val
            model["settings"] = settings
            print(f"  {Colors.SUCCESS}✓{Colors.RESET} {desc} updated to: {Colors.ACCENT}{new_val}{Colors.RESET}" if not STEALTH_MODE else f"Updated: {new_val}")
        except ValueError:
            print(f"  {Colors.ERROR}Invalid value.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")


def edit_settings() -> None:
    """Interactive configuration editor for runtime settings."""
    global MAX_SEQUENCE_LENGTH, NUM_THREADS, GPU_LAYERS, MAX_TOKENS
    global MAX_HISTORY_SIZE, ENABLE_MONITORING, STEALTH_MODE, SESSION_TIMEOUT
    
    while True:
        # Display main settings menu
        if STEALTH_MODE:
            print("\n" + "-" * 40)
            print("Settings")
            print("-" * 40)
        else:
            print(f"\n{Colors.PRIMARY}╔{'═' * 48}╗")
            print(f"║{Colors.BOLD}{'Settings Menu':^48}{Colors.RESET}{Colors.PRIMARY}║")
            print(f"╚{'═' * 48}╝{Colors.RESET}")
        
        if STEALTH_MODE:
            print("  [1] Global Settings")
            print("  [2] Model Settings")
            print("  [0] Back")
        else:
            print(f"  {Colors.SUCCESS}1{Colors.RESET}. Global Settings")
            print(f"  {Colors.SUCCESS}2{Colors.RESET}. Model Settings")
            print(f"\n  {Colors.MUTED}0{Colors.RESET}. Back to chat")
        
        print()
        choice = input(f"{Colors.INFO}Select [1-2]:{Colors.RESET} " if not STEALTH_MODE else "Select: ").strip()
        
        if choice == "0" or choice.lower() in ["back", "exit", "q"]:
            break
        elif choice == "1":
            edit_global_settings()
        elif choice == "2":
            edit_model_settings_menu()
        else:
            print(f"{Colors.ERROR}Invalid selection.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")


def edit_model_settings_menu() -> None:
    """Menu to select which model's settings to edit."""
    while True:
        if STEALTH_MODE:
            print("\n--- Select Model ---")
        else:
            print(f"\n{Colors.PRIMARY}╔{'═' * 48}╗")
            print(f"║{Colors.BOLD}{'Select Model to Configure':^48}{Colors.RESET}{Colors.PRIMARY}║")
            print(f"╚{'═' * 48}╝{Colors.RESET}")
        
        model_keys = list(MODELS.keys())
        for i, key in enumerate(model_keys, 1):
            model_info = MODELS[key]
            desc = model_info.get('description', model_info['name'][:35])
            if STEALTH_MODE:
                print(f"  [{i}] {key}")
            else:
                print(f"  {Colors.SUCCESS}{i}{Colors.RESET}. {Colors.ACCENT}{key.upper():<12}{Colors.RESET}")
                print(f"     {Colors.MUTED}{desc}{Colors.RESET}")
        
        print(f"\n  {Colors.MUTED}0{Colors.RESET}. Back" if not STEALTH_MODE else "  [0] Back")
        print()
        
        choice = input(f"{Colors.INFO}Select [1-{len(model_keys)}]:{Colors.RESET} " if not STEALTH_MODE else "Select: ").strip()
        
        if choice == "0" or choice.lower() in ["back", "exit", "q", "b"]:
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_keys):
                edit_model_settings(model_keys[idx])
            else:
                print(f"{Colors.ERROR}Invalid selection.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")
        except ValueError:
            print(f"{Colors.ERROR}Enter a number.{Colors.RESET}" if not STEALTH_MODE else "Enter a number.")


def edit_global_settings() -> None:
    """Edit global runtime settings."""
    global MAX_SEQUENCE_LENGTH, NUM_THREADS, GPU_LAYERS, MAX_TOKENS
    global MAX_HISTORY_SIZE, ENABLE_MONITORING, STEALTH_MODE, SESSION_TIMEOUT
    
    settings: Dict[str, Tuple[str, str, Callable, Any]] = {
        "1": ("MAX_SEQUENCE_LENGTH", "Context length", lambda: MAX_SEQUENCE_LENGTH, int),
        "2": ("NUM_THREADS", "CPU threads", lambda: NUM_THREADS, int),
        "3": ("GPU_LAYERS", "GPU layers", lambda: GPU_LAYERS, int),
        "4": ("MAX_TOKENS", "Max tokens", lambda: MAX_TOKENS, int),
        "5": ("MAX_HISTORY_SIZE", "History limit", lambda: MAX_HISTORY_SIZE, "bool_or_int"),
        "6": ("ENABLE_MONITORING", "Monitoring", lambda: ENABLE_MONITORING, bool),
        "7": ("STEALTH_MODE", "Stealth mode", lambda: STEALTH_MODE, bool),
        "8": ("SESSION_TIMEOUT", "Session timeout", lambda: SESSION_TIMEOUT, "bool_or_int"),
    }
    
    while True:
        # Display settings menu
        if STEALTH_MODE:
            print("\n" + "-" * 40)
            print("Global Settings")
            print("-" * 40)
        else:
            print(f"\n{Colors.PRIMARY}╔{'═' * 48}╗")
            print(f"║{Colors.BOLD}{'Global Settings':^48}{Colors.RESET}{Colors.PRIMARY}║")
            print(f"╚{'═' * 48}╝{Colors.RESET}")
        
        for key, (name, desc, getter, val_type) in settings.items():
            val = getter()
            
            # Format value display
            if val is False:
                val_display = f"{Colors.MUTED}Disabled{Colors.RESET}" if not STEALTH_MODE else "Disabled"
            elif val is True:
                val_display = f"{Colors.SUCCESS}Enabled{Colors.RESET}" if not STEALTH_MODE else "Enabled"
            else:
                val_display = f"{Colors.ACCENT}{val}{Colors.RESET}" if not STEALTH_MODE else str(val)
            
            if STEALTH_MODE:
                print(f"  [{key}] {desc}: {val_display}")
            else:
                print(f"  {Colors.SUCCESS}{key}{Colors.RESET}. {desc:<20} {val_display}")
        
        print(f"\n  {Colors.MUTED}0{Colors.RESET}. Back" if not STEALTH_MODE else "  [0] Back")
        print()
        
        choice = input(f"{Colors.INFO}Select [1-8]:{Colors.RESET} " if not STEALTH_MODE else "Edit: ").strip()
        
        if choice == "0" or choice.lower() in ["back", "exit", "q"]:
            break
        
        if choice not in settings:
            print(f"{Colors.ERROR}Invalid selection.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")
            continue
        
        name, desc, getter, val_type = settings[choice]
        current = getter()
        
        if val_type == bool:
            # Toggle boolean settings
            new_val = not current
            status = f"{Colors.SUCCESS}Enabled{Colors.RESET}" if new_val else f"{Colors.MUTED}Disabled{Colors.RESET}"
            print(f"  {desc}: {status}" if not STEALTH_MODE else f"{desc}: {'Enabled' if new_val else 'Disabled'}")
        elif val_type == "bool_or_int":
            # Display current status
            if not STEALTH_MODE:
                print(f"\n{Colors.MUTED}─" * 40 + Colors.RESET)
            
            if "History" in desc:
                if current is False:
                    print(f"  Current: {Colors.MUTED}UNLIMITED{Colors.RESET}" if not STEALTH_MODE else "  Currently: UNLIMITED")
                    print(f"  {Colors.INFO}•{Colors.RESET} Enter a number to set limit" if not STEALTH_MODE else "  • Enter number to set limit")
                else:
                    print(f"  Current: {Colors.ACCENT}{current}{Colors.RESET} exchanges" if not STEALTH_MODE else f"  Currently: {current} exchanges")
                    print(f"  {Colors.INFO}•{Colors.RESET} Enter number to change | 'false' to unlimited" if not STEALTH_MODE else "  • Enter number or 'false'")
            elif "timeout" in desc.lower():
                if current is False:
                    print(f"  Current: {Colors.MUTED}DISABLED{Colors.RESET}" if not STEALTH_MODE else "  Currently: DISABLED")
                    print(f"  {Colors.INFO}•{Colors.RESET} Enter minutes to enable auto-lock" if not STEALTH_MODE else "  • Enter minutes to enable")
                else:
                    print(f"  Current: {Colors.ACCENT}{current}{Colors.RESET} minutes" if not STEALTH_MODE else f"  Currently: {current} minutes")
                    print(f"  {Colors.INFO}•{Colors.RESET} Enter number to change | 'false' to disable" if not STEALTH_MODE else "  • Enter number or 'false'")
            
            inp = input(f"  {Colors.INFO}>{Colors.RESET} " if not STEALTH_MODE else "> ").strip().lower()
            
            if inp in ["back", "b", "cancel", "c", "q"]:
                continue
            
            if inp in ["false", "f", "off"]:
                if current is False:
                    print(f"  {Colors.MUTED}Already disabled.{Colors.RESET}" if not STEALTH_MODE else "Already disabled.")
                    continue
                new_val = False
            elif inp in ["true", "t", "on", "yes"]:
                print(f"  {Colors.WARNING}Enter a NUMBER to enable (e.g., 5 or 10).{Colors.RESET}" if not STEALTH_MODE else "Enter a number to enable.")
                continue
            else:
                try:
                    new_val = int(inp)
                    if new_val <= 0:
                        print(f"  {Colors.ERROR}Must be a positive number.{Colors.RESET}" if not STEALTH_MODE else "Must be positive.")
                        continue
                except ValueError:
                    print(f"  {Colors.ERROR}Invalid. Enter a number.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")
                    continue
        else:
            # Integer settings
            if not STEALTH_MODE:
                print(f"\n  Current: {Colors.ACCENT}{current}{Colors.RESET}")
            else:
                print(f"Current: {current}")
            
            inp = input(f"  New value ({Colors.MUTED}'back' to cancel{Colors.RESET}): " if not STEALTH_MODE else "New value: ").strip()
            
            if inp.lower() in ["back", "b", "cancel", "c", "q", "0"]:
                continue
            
            try:
                new_val = val_type(inp)
                if new_val <= 0:
                    print(f"  {Colors.ERROR}Must be positive.{Colors.RESET}" if not STEALTH_MODE else "Must be positive.")
                    continue
            except ValueError:
                print(f"  {Colors.ERROR}Invalid value.{Colors.RESET}" if not STEALTH_MODE else "Invalid.")
                continue
        
        # Apply the setting
        if choice == "1":
            MAX_SEQUENCE_LENGTH = new_val
        elif choice == "2":
            NUM_THREADS = new_val
        elif choice == "3":
            GPU_LAYERS = new_val
        elif choice == "4":
            MAX_TOKENS = new_val
        elif choice == "5":
            MAX_HISTORY_SIZE = new_val
        elif choice == "6":
            ENABLE_MONITORING = new_val
        elif choice == "7":
            STEALTH_MODE = new_val
        elif choice == "8":
            SESSION_TIMEOUT = new_val
        
        if val_type not in [bool]:
            display_val = "Disabled" if new_val is False else new_val
            print(f"  {Colors.SUCCESS}✓{Colors.RESET} {desc} updated to: {Colors.ACCENT}{display_val}{Colors.RESET}" if not STEALTH_MODE else f"Updated: {display_val}")

# =============================================================================
# MODEL SELECTION
# =============================================================================

def select_model(stealth: bool = False) -> str:
    """Display model selection menu and return chosen model key."""
    if stealth:
        print("\nSelect mode:")
        for i, key in enumerate(MODELS.keys(), 1):
            print(f"  [{i}] Mode {i}")
    else:
        print(f"\n{Colors.ACCENT}Select AI Model:{Colors.RESET}\n")
        
        for i, (key, value) in enumerate(MODELS.items(), 1):
            name_display = key.upper()
            desc = value.get('description', value['name'])
            settings = value.get('settings', {})
            temp = settings.get('temperature', 0.7)
            max_tok = settings.get('max_tokens', 2048)
            print(f"  {Colors.SUCCESS}{i}{Colors.RESET}. {Colors.BOLD}{name_display}{Colors.RESET}")
            print(f"     {Colors.MUTED}{desc}{Colors.RESET}")
            print(f"     {Colors.MUTED}Temp: {temp} | Max Tokens: {max_tok}{Colors.RESET}\n")
    
    while True:
        try:
            prompt = f"\n{Colors.INFO}Select [1-{len(MODELS)}]:{Colors.RESET} " if not stealth else "\nSelect: "
            choice = int(input(prompt))
            if 1 <= choice <= len(MODELS):
                return list(MODELS.keys())[choice - 1]
            print(f"{Colors.ERROR}Invalid choice.{Colors.RESET}" if not stealth else "Invalid.")
        except ValueError:
            print(f"{Colors.ERROR}Enter a number.{Colors.RESET}" if not stealth else "Enter a number.")

# =============================================================================
# AUTHENTICATION
# =============================================================================

def verify_password() -> bool:
    """Verify user password before granting access."""
    os.system("cls" if os.name == "nt" else "clear")
    
    if STEALTH_MODE:
        print("Microsoft Windows [Version 10.0.22631.4460]")
        print("(c) Microsoft Corporation. All rights reserved.\n")
        print("Access code required.")
    else:
        print(f"\n{Colors.PRIMARY}╔{'═' * 48}╗")
        print(f"║{Colors.BOLD}{'NexusAI Security Gateway':^48}{Colors.RESET}{Colors.PRIMARY}║")
        print(f"╚{'═' * 48}╝{Colors.RESET}")
        print(f"\n  {Colors.WARNING}Authentication required to continue.{Colors.RESET}\n")
    
    for attempt in range(MAX_LOGIN_ATTEMPTS):
        remaining = MAX_LOGIN_ATTEMPTS - attempt
        
        try:
            if STEALTH_MODE:
                password = getpass.getpass("Access code: ")
            else:
                password = getpass.getpass(f"  {Colors.INFO}Password{Colors.RESET} ({remaining} {'attempt' if remaining == 1 else 'attempts'} left): ")
        except KeyboardInterrupt:
            return False
        
        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
            if not STEALTH_MODE:
                print(f"\n  {Colors.SUCCESS}✓ Access granted. Welcome to NexusAI.{Colors.RESET}\n")
            time.sleep(0.5)
            return True
        else:
            print(f"  {Colors.ERROR}✗ Incorrect password.{Colors.RESET}" if not STEALTH_MODE else "Access denied.")
    
    print(f"\n  {Colors.ERROR}Maximum attempts exceeded. System locked.{Colors.RESET}" if not STEALTH_MODE else "\nLocked.")
    return False

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def check_session_timeout(last_activity: float) -> bool:
    """Check if session has exceeded timeout threshold."""
    if not SESSION_TIMEOUT:
        return False
    elapsed_minutes = (time.time() - last_activity) / 60
    return elapsed_minutes >= SESSION_TIMEOUT

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for NexusAI."""
    global STEALTH_MODE
    
    # Authentication
    if not verify_password():
        sys.exit(1)
    
    try:
        assistant = NexusAI(MODEL_DIRECTORY)
        assistant.clear_screen()
        show_ascii()

        # Model selection
        if not STEALTH_MODE:
            print(f"{Colors.INFO}[NexusAI]{Colors.RESET} Please select an AI model to begin.")
        model_key = select_model(STEALTH_MODE)
        assistant.load_model(model_key)
        assistant.clear_screen()
        show_ascii(model_key)
        
        last_activity = time.time()

        # Main interaction loop
        while True:
            # Session timeout check
            if SESSION_TIMEOUT and check_session_timeout(last_activity):
                assistant.clear_screen()
                print(f"\n{Colors.WARNING}Session expired due to inactivity.{Colors.RESET}\n" if not STEALTH_MODE else "Session expired.\n")
                
                if not verify_password():
                    if not STEALTH_MODE:
                        print(f"{Colors.ERROR}Authentication failed.{Colors.RESET}")
                    sys.exit(1)
                
                assistant.clear_screen()
                show_ascii(assistant.current_model)
                last_activity = time.time()
                continue
            
            # Build prompt
            if STEALTH_MODE:
                prompt = f"{get_fake_path()}> "
            else:
                prompt = f"{Colors.SUCCESS}You:{Colors.RESET} "
            
            try:
                # Get user input
                user_input = input(prompt).strip()
                
                # Smart paste detection: if input matches first line of clipboard,
                # assume user pasted multi-line text and use full clipboard
                if user_input:
                    try:
                        clipboard = pyperclip.paste()
                        if clipboard:
                            clipboard_lines = clipboard.strip().split('\n')
                            first_clipboard_line = clipboard_lines[0].strip()
                            
                            # If input matches first line of clipboard and clipboard has multiple lines
                            if (len(clipboard_lines) > 1 and 
                                user_input == first_clipboard_line):
                                # Use full clipboard content (join with spaces)
                                user_input = ' '.join(line.strip() for line in clipboard_lines if line.strip())
                                
                                # IMPORTANT: Drain the stdin buffer to discard remaining pasted lines
                                if os.name == 'nt':
                                    import msvcrt
                                    time.sleep(0.05)  # Give buffer time to fill
                                    while msvcrt.kbhit():
                                        msvcrt.getwch()  # Discard each character
                                
                                if not STEALTH_MODE:
                                    assistant._print_status(f"Multi-line paste detected - captured all {len(clipboard_lines)} lines.")
                                    # Show the captured content
                                    print(f"\n{Colors.MUTED}┌─ Captured Input {'─' * 30}{Colors.RESET}")
                                    for line in clipboard_lines:
                                        if line.strip():
                                            print(f"{Colors.MUTED}│ {line.strip()}{Colors.RESET}")
                                    print(f"{Colors.MUTED}└{'─' * 48}{Colors.RESET}\n")
                    except:
                        pass  # Clipboard access failed, use original input
                
                last_activity = time.time()
            except (EOFError, KeyboardInterrupt):
                if not STEALTH_MODE:
                    print(f"\n{Colors.INFO}[NexusAI]{Colors.RESET} Session ended. Goodbye!")
                break

            # Handle commands
            cmd = user_input.lower()
            
            if cmd == "exit":
                if not STEALTH_MODE:
                    print(f"\n{Colors.INFO}[NexusAI]{Colors.RESET} Thank you for using NexusAI. Goodbye!")
                break

            if cmd == "stealth":
                STEALTH_MODE = not STEALTH_MODE
                assistant.clear_screen()
                show_ascii(assistant.current_model)
                if not STEALTH_MODE:
                    assistant._print_success("Stealth mode disabled.")
                continue

            if cmd == "reset":
                assistant.reset_conversation()
                continue

            if cmd == "help":
                print(assistant.show_help())
                continue

            if cmd in ["clear", "cls"]:
                assistant.clear_screen()
                show_ascii(assistant.current_model)
                continue

            if cmd in ["cc", "ca"]:
                assistant.process_command(cmd)
                continue

            # Paste command - reads from clipboard (solves multi-line paste issue)
            if cmd in ["paste", "v", "p"]:
                clipboard_content = assistant.process_command("paste")
                if clipboard_content:
                    user_input = clipboard_content
                    # Fall through to process as normal input
                else:
                    continue

            if cmd == "model":
                model_key = select_model(STEALTH_MODE)
                assistant.load_model(model_key)
                assistant.clear_screen()
                show_ascii(model_key)
                continue

            if cmd == "settings":
                edit_settings()
                assistant.clear_screen()
                show_ascii(assistant.current_model)
                continue

            # Process user message
            if user_input:
                processed_input = assistant.process_file_references(user_input)
                processed_input, images = assistant.process_image_references(processed_input)
                
                if images:
                    assistant._print_warning("Images loaded but current model is text-only.")
                
                assistant.generate_response(processed_input)

    except KeyboardInterrupt:
        if not STEALTH_MODE:
            print(f"\n{Colors.INFO}[NexusAI]{Colors.RESET} Session interrupted. Goodbye!")
    except Exception as e:
        print(f"{Colors.ERROR}Critical error: {e}{Colors.RESET}" if not STEALTH_MODE else f"Error: {e}")

if __name__ == "__main__":
    main()

