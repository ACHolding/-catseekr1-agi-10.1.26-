#!/usr/bin/env python3
"""
CatSeek R1 1.x
Single-file Tkinter local agent whose reasoning stack mirrors DeepSeek-V4.1-Flash
architecture in pure Python (no cloud, no checkpoint download).

Architecture mirror (toy-scale, offline):
- Causal Encoder–Decoder (CED): 20 encoder + 20 decoder layers
- Compressed Sparse Attention 2 (CSA2): Full / Reindex / Reuse modes
- DeepSeekMoE-style routing: 1 shared + 384 routed experts, top-6 active
- Engram conditional memory (token-keyed sparse lookup)
- FP4-style global KV cache (~890 bytes/token metaphor)
- Single-Pass mHC residual mixing
- DSpark speculative draft + confidence verification
- Controllable reasoning effort (1–100)

FILES = OFF (agent tools):
- No file read/write tools are exposed to the agent.
- No shell execution.
- No arbitrary Python execution.

App memory:
- On first OS launch / detect of CatSeek R1, a persistent memory file is created
  under ~/.catseek-r1/memory.json and survives app close/reopen.
- Use Clear Memory to wipe that store.
"""

import json
import os
import queue
import re
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

APP_NAME = "CatSeek R1 1.x [c] Kondo Solutions 1999-2026"
MODEL_NAME = "CatSeek R1 1.x"
MODEL_ARCH = "CED-MoE · V4.1-Flash mirror"
FILES_OFF = True
LOCAL_ONLY = True
PERSISTENT_MEMORY = True
MEMORY_DIR = Path.home() / ".catseek-r1"
MEMORY_FILE = MEMORY_DIR / "memory.json"
MEMORY_VERSION = 1
MAX_LOG_ENTRIES = 500

# DeepSeek-V4.1-Flash scale factors, kept as named constants for the toy runtime.
N_LAYERS = 40
N_ENCODER = 20
N_DECODER = 20
N_ROUTED_EXPERTS = 384
N_SHARED_EXPERTS = 1
N_ACTIVE_ROUTED = 6
KV_BYTES_PER_TOKEN = 890
CONTEXT_WINDOW = 1_000_000
PREFILL_ACTIVE_B = 8
DECODE_ACTIVE_B = 16
ENGRAM_PARAMS_B = 196
BACKBONE_PARAMS_B = 552
DEFAULT_REASONING_EFFORT = 64

SYSTEM_PROMPT = """You are CatSeek R1 1.x, a goal-oriented AI agent.
Your stack mirrors DeepSeek-V4.1-Flash: Causal Encoder–Decoder, CSA2, MoE,
Engram memory, FP4 KV, Single-Pass mHC, and DSpark decode.
Work toward the user's objective one step at a time.

You have NO file access, NO shell, and NO arbitrary code execution.
Available actions:
- THINK: reason about the next useful step.
- ASK: ask the user for missing information.
- ANSWER: provide useful progress or a result.
- DONE: provide the final result and finish the run.

Return ONLY valid JSON in this exact shape:
{"thought":"brief private planning summary","action":"THINK|ASK|ANSWER|DONE","content":"message or useful result"}

Rules:
- Prefer concrete progress over repetitive planning.
- Do not claim to have accessed files, programs, websites, or devices.
- Do not invent tool results.
- Keep the thought short.
- Use DONE once the objective has been adequately answered.
"""


# ---------------------------------------------------------------------------
# DeepSeek-V4.1-Flash architectural components (pure-Python toy mirror)
# ---------------------------------------------------------------------------

class FP4KVCache:
    """FP4 (E2M1 + E4M3 scale/16-ch) global KV cache — ~890 bytes/token."""

    def __init__(self, bytes_per_token=KV_BYTES_PER_TOKEN):
        self.bytes_per_token = bytes_per_token
        self.tokens = []
        self._hbm = []

    def write(self, token_id, payload):
        # Store a compact fingerprint instead of full activations.
        entry = (token_id, hash(payload) & 0xFFFFFFFF, len(payload))
        self.tokens.append(entry)
        self._hbm.append(entry)
        return entry

    def footprint(self):
        return len(self.tokens) * self.bytes_per_token

    def project_from_encoder(self, encoder_hidden):
        """CED: decoder global KV is projected from final encoder states."""
        projected = []
        for i, h in enumerate(encoder_hidden):
            projected.append(self.write(i, h))
        return projected


class CSA2Attention:
    """Compressed Sparse Attention 2 with Full / Reindex / Reuse modes."""

    MODES = ("Full", "Reindex", "Reuse")

    def __init__(self):
        # Encoder: first 2 SWA-only; remaining CSA2 with alternating modes.
        # Decoder: Hierarchical Sparse Indexer over Full → Reindex → Reuse.
        self.encoder_modes = ["SWA", "SWA"] + [
            self.MODES[i % 3] for i in range(N_ENCODER - 2)
        ]
        self.decoder_modes = [self.MODES[i % 3] for i in range(N_DECODER)]
        self.last_full_kv = None
        self.last_topk = None
        self.candidate_pool = []

    def _topk(self, scores, k=512):
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[: min(k, len(ranked))]

    def attend(self, layer, side, query, kv_store):
        modes = self.encoder_modes if side == "encoder" else self.decoder_modes
        mode = modes[layer]
        scores = [((query ^ (e[1] if isinstance(e, tuple) else hash(e))) & 0xFFFF) for e in kv_store] or [1]

        if mode == "SWA":
            window = kv_store[-64:] if len(kv_store) > 64 else kv_store
            return {"mode": mode, "window": len(window), "selected": list(range(len(window)))}

        if mode == "Full":
            top = self._topk(scores)
            self.last_full_kv = kv_store
            self.last_topk = top
            # Hierarchical Sparse Indexer: 2048 blocks × 8 → candidate pool.
            blocks = max(1, len(scores) // 8)
            block_scores = [
                max(scores[b * 8 : (b + 1) * 8] or [0]) for b in range(blocks)
            ]
            best_blocks = self._topk(block_scores, k=min(2048, len(block_scores)))
            pool = []
            for b in best_blocks:
                pool.extend(range(b * 8, min((b + 1) * 8, len(scores))))
            self.candidate_pool = pool[:16384]
            return {"mode": mode, "selected": top, "pool": len(self.candidate_pool)}

        if mode == "Reindex":
            domain = self.candidate_pool or list(range(len(scores)))
            local = [scores[i] if i < len(scores) else 0 for i in domain]
            local_top = self._topk(local)
            top = [domain[i] for i in local_top if i < len(domain)]
            self.last_topk = top
            return {"mode": mode, "selected": top, "from_pool": True}

        # Reuse
        return {
            "mode": "Reuse",
            "selected": self.last_topk or [],
            "kv_from": "prior Full",
        }


class SinglePassMHC:
    """Single-Pass multi-stream Hyper-Connection residual mixer."""

    def __init__(self, n_streams=4):
        self.n_streams = n_streams

    def mix(self, streams, block_out):
        # Xl+1 = B·Xl + C·F(A·Xl) — toy scalar mix of residual streams.
        if not streams:
            return [block_out] * self.n_streams
        a = sum(streams) / len(streams)
        mixed = []
        for i, s in enumerate(streams):
            mixed.append((s + a + block_out * (i + 1)) % (2**31 - 1))
        return mixed


class BitNetCoder:
    """
    Offline code synthesizer (BitNet coding expert).

    Parses write/implement/print requests and emits real source for many
    languages — not just baked prose tips.
    """

    LANG_ALIASES = {
        "c": "c", "c language": "c", "clang": "c",
        "c++": "cpp", "cpp": "cpp", "cplusplus": "cpp",
        "c#": "csharp", "csharp": "csharp", "cs": "csharp",
        "python": "python", "py": "python",
        "javascript": "javascript", "js": "javascript", "node": "javascript",
        "typescript": "typescript", "ts": "typescript",
        "java": "java",
        "go": "go", "golang": "go",
        "rust": "rust", "rs": "rust",
        "ruby": "ruby", "rb": "ruby",
        "php": "php",
        "swift": "swift",
        "kotlin": "kotlin", "kt": "kotlin",
        "bash": "bash", "shell": "bash", "sh": "bash",
        "lua": "lua",
        "r": "r",
        "perl": "perl",
        "haskell": "haskell", "hs": "haskell",
        "scala": "scala",
        "dart": "dart",
        "sql": "sql",
        "html": "html",
        "assembly": "asm", "asm": "asm", "x86": "asm",
    }

    CODE_VERBS = (
        "write", "implement", "code", "program", "create", "make", "build",
        "generate", "craft", "author", "print", "show", "emit", "compile",
        "fix", "debug", "refactor", "convert", "translate",
    )

    @classmethod
    def is_coding_request(cls, text):
        q = text.lower().strip()
        if not q:
            return False
        if re.search(r"\b(chip-?8|emulator)\b", q):
            return True
        if any(re.search(rf"\b{re.escape(v)}\b", q) for v in cls.CODE_VERBS):
            return True
        if re.search(
            r"\b(function|class|method|script|program|source|snippet|"
            r"algorithm|fizzbuzz|factorial|fibonacci|hello\s*world|"
            r"\.c\b|\.py\b|\.js\b|\.rs\b|\.go\b|\.java\b)\b",
            q,
        ):
            return True
        # "… in <lang>" / "<lang> code"
        for alias in cls.LANG_ALIASES:
            if re.search(rf"\bin\s+{re.escape(alias)}\b", q) or re.search(
                rf"\b{re.escape(alias)}\s+code\b", q
            ):
                return True
        return False

    @classmethod
    def detect_lang(cls, text):
        q = text.lower()
        # Prefer explicit "in <lang>" / "<lang>:" patterns (longest alias first).
        for alias in sorted(cls.LANG_ALIASES, key=len, reverse=True):
            if re.search(rf"\bin\s+{re.escape(alias)}\b", q):
                return cls.LANG_ALIASES[alias]
            if re.search(rf"\b{re.escape(alias)}\s+code\b", q):
                return cls.LANG_ALIASES[alias]
            if re.search(rf"\b{re.escape(alias)}\b", q) and alias not in {"r", "c", "go", "sh"}:
                return cls.LANG_ALIASES[alias]
        # Standalone short aliases only when clearly coding.
        for alias in ("c", "c++", "go", "r", "sh"):
            if re.search(rf"\bin\s+{re.escape(alias)}\b", q):
                return cls.LANG_ALIASES[alias]
        return "python"

    @classmethod
    def _extract_message(cls, text):
        """Pull the string the user wants printed (e.g. hello cat)."""
        q = text.strip()
        # Quoted payload wins.
        m = re.search(r'["“](.+?)["”]', q)
        if m:
            return m.group(1).strip()
        m = re.search(r"'(.+?)'", q)
        if m:
            return m.group(1).strip()

        low = q.lower()
        # write/print/say <message> in <lang>
        m = re.search(
            r"\b(?:write|print|say|echo|output|display)\s+"
            r"(?:a\s+|an\s+|the\s+|program\s+(?:that\s+prints?\s+|to\s+print\s+)?)?"
            r"(.+?)\s+in\s+\w[\w+#++]*\s*$",
            low,
            re.I,
        )
        if m:
            msg = m.group(1).strip()
            msg = re.sub(
                r"^(a\s+|an\s+|the\s+)?(program|script|code|function|app)\s+"
                r"(that\s+)?(prints?\s+|outputs?\s+|says?\s+)?",
                "",
                msg,
                flags=re.I,
            ).strip()
            if msg:
                return msg

        # hello world / hello <words>
        m = re.search(r"\b(hello(?:\s+[\w'-]+){0,4})\b", low)
        if m:
            return m.group(1).strip()

        # strip coding verbs / language tails → leftover as message
        cleaned = re.sub(
            r"\b(" + "|".join(map(re.escape, cls.CODE_VERBS)) + r")\b",
            " ",
            low,
            flags=re.I,
        )
        cleaned = re.sub(r"\bin\s+[\w+#++]+\s*$", " ", cleaned, flags=re.I)
        cleaned = re.sub(
            r"\b(a|an|the|program|script|code|function|app|that|prints?|outputs?)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
        return cleaned or "hello"

    @classmethod
    def _escape(cls, lang, msg):
        if lang in {"c", "cpp", "csharp", "java", "javascript", "typescript", "go", "rust", "swift", "kotlin", "dart"}:
            return msg.replace("\\", "\\\\").replace('"', '\\"')
        if lang == "python":
            return msg.replace("\\", "\\\\").replace("'", "\\'")
        return msg.replace("'", "'\\''") if lang == "bash" else msg

    @classmethod
    def print_program(cls, lang, message):
        msg = cls._escape(lang, message)
        fence = {
            "c": "c", "cpp": "cpp", "csharp": "csharp", "python": "python",
            "javascript": "javascript", "typescript": "typescript", "java": "java",
            "go": "go", "rust": "rust", "ruby": "ruby", "php": "php",
            "swift": "swift", "kotlin": "kotlin", "bash": "bash", "lua": "lua",
            "r": "r", "perl": "perl", "haskell": "haskell", "scala": "scala",
            "dart": "dart", "sql": "sql", "html": "html", "asm": "asm",
        }.get(lang, lang)

        bodies = {
            "c": (
                f'#include <stdio.h>\n\n'
                f'int main(void) {{\n'
                f'    printf("{msg}\\n");\n'
                f'    return 0;\n'
                f'}}\n'
            ),
            "cpp": (
                f'#include <iostream>\n\n'
                f'int main() {{\n'
                f'    std::cout << "{msg}" << std::endl;\n'
                f'    return 0;\n'
                f'}}\n'
            ),
            "csharp": (
                f'using System;\n\n'
                f'class Program {{\n'
                f'    static void Main() {{\n'
                f'        Console.WriteLine("{msg}");\n'
                f'    }}\n'
                f'}}\n'
            ),
            "python": f"print('{msg}')\n",
            "javascript": f'console.log("{msg}");\n',
            "typescript": f'console.log("{msg}");\n',
            "java": (
                f'public class Main {{\n'
                f'    public static void main(String[] args) {{\n'
                f'        System.out.println("{msg}");\n'
                f'    }}\n'
                f'}}\n'
            ),
            "go": (
                f'package main\n\n'
                f'import "fmt"\n\n'
                f'func main() {{\n'
                f'    fmt.Println("{msg}")\n'
                f'}}\n'
            ),
            "rust": (
                f'fn main() {{\n'
                f'    println!("{msg}");\n'
                f'}}\n'
            ),
            "ruby": f'puts "{msg}"\n',
            "php": f'<?php\necho "{msg}\\n";\n',
            "swift": f'print("{msg}")\n',
            "kotlin": f'fun main() {{\n    println("{msg}")\n}}\n',
            "bash": f"echo '{message}'\n",
            "lua": f'print("{msg}")\n',
            "r": f'cat("{msg}\\n")\n',
            "perl": f'print "{msg}\\n";\n',
            "haskell": f'main :: IO ()\nmain = putStrLn "{msg}"\n',
            "scala": f'@main def hello() = println("{msg}")\n',
            "dart": f'void main() {{\n  print("{msg}");\n}}\n',
            "sql": f"SELECT '{msg}' AS greeting;\n",
            "html": (
                f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
                f"  <meta charset=\"utf-8\">\n  <title>{message}</title>\n"
                f"</head>\n<body>\n  <p>{message}</p>\n</body>\n</html>\n"
            ),
            "asm": (
                f"; nasm -f elf64 hello.asm && ld hello.o -o hello\n"
                f"section .data\n"
                f'    msg db "{msg}", 10\n'
                f"    len equ $ - msg\n\n"
                f"section .text\n"
                f"    global _start\n"
                f"_start:\n"
                f"    mov rax, 1\n"
                f"    mov rdi, 1\n"
                f"    mov rsi, msg\n"
                f"    mov rdx, len\n"
                f"    syscall\n"
                f"    mov rax, 60\n"
                f"    xor rdi, rdi\n"
                f"    syscall\n"
            ),
        }
        code = bodies.get(lang, bodies["python"])
        return (
            f"Here is a complete {lang} program that prints `{message}`:\n\n"
            f"```{fence}\n{code}```\n"
            f"Save it, then compile/run with your usual {lang} toolchain."
        )

    @classmethod
    def special_program(cls, lang, text):
        q = text.lower()

        if "fizzbuzz" in q:
            templates = {
                "python": (
                    "for n in range(1, 101):\n"
                    "    if n % 15 == 0:\n        print('FizzBuzz')\n"
                    "    elif n % 3 == 0:\n        print('Fizz')\n"
                    "    elif n % 5 == 0:\n        print('Buzz')\n"
                    "    else:\n        print(n)\n"
                ),
                "c": (
                    "#include <stdio.h>\n\n"
                    "int main(void) {\n"
                    "    for (int n = 1; n <= 100; n++) {\n"
                    "        if (n % 15 == 0) puts(\"FizzBuzz\");\n"
                    "        else if (n % 3 == 0) puts(\"Fizz\");\n"
                    "        else if (n % 5 == 0) puts(\"Buzz\");\n"
                    "        else printf(\"%d\\n\", n);\n"
                    "    }\n    return 0;\n}\n"
                ),
                "javascript": (
                    "for (let n = 1; n <= 100; n++) {\n"
                    "  if (n % 15 === 0) console.log('FizzBuzz');\n"
                    "  else if (n % 3 === 0) console.log('Fizz');\n"
                    "  else if (n % 5 === 0) console.log('Buzz');\n"
                    "  else console.log(n);\n"
                    "}\n"
                ),
                "go": (
                    "package main\n\nimport \"fmt\"\n\n"
                    "func main() {\n"
                    "    for n := 1; n <= 100; n++ {\n"
                    "        switch {\n"
                    "        case n%15 == 0:\n            fmt.Println(\"FizzBuzz\")\n"
                    "        case n%3 == 0:\n            fmt.Println(\"Fizz\")\n"
                    "        case n%5 == 0:\n            fmt.Println(\"Buzz\")\n"
                    "        default:\n            fmt.Println(n)\n"
                    "        }\n    }\n}\n"
                ),
                "rust": (
                    "fn main() {\n"
                    "    for n in 1..=100 {\n"
                    "        match (n % 3, n % 5) {\n"
                    "            (0, 0) => println!(\"FizzBuzz\"),\n"
                    "            (0, _) => println!(\"Fizz\"),\n"
                    "            (_, 0) => println!(\"Buzz\"),\n"
                    "            _ => println!(\"{n}\"),\n"
                    "        }\n    }\n}\n"
                ),
            }
            code = templates.get(lang, templates["python"])
            fence = "python" if lang not in templates else lang
            if lang not in templates:
                fence = "python"
                lang = "python"
            return f"FizzBuzz in {lang}:\n\n```{fence}\n{code}```"

        if "factorial" in q:
            templates = {
                "python": (
                    "def factorial(n):\n"
                    "    if n < 0:\n        raise ValueError('n must be >= 0')\n"
                    "    out = 1\n"
                    "    for i in range(2, n + 1):\n        out *= i\n"
                    "    return out\n\n"
                    "if __name__ == '__main__':\n"
                    "    print(factorial(5))\n"
                ),
                "c": (
                    "#include <stdio.h>\n\n"
                    "long factorial(int n) {\n"
                    "    long out = 1;\n"
                    "    for (int i = 2; i <= n; i++) out *= i;\n"
                    "    return out;\n}\n\n"
                    "int main(void) {\n"
                    "    printf(\"%ld\\n\", factorial(5));\n"
                    "    return 0;\n}\n"
                ),
                "go": (
                    "package main\n\nimport \"fmt\"\n\n"
                    "func factorial(n int) int {\n"
                    "    out := 1\n"
                    "    for i := 2; i <= n; i++ {\n        out *= i\n    }\n"
                    "    return out\n}\n\n"
                    "func main() {\n    fmt.Println(factorial(5))\n}\n"
                ),
                "javascript": (
                    "function factorial(n) {\n"
                    "  let out = 1;\n"
                    "  for (let i = 2; i <= n; i++) out *= i;\n"
                    "  return out;\n"
                    "}\n\nconsole.log(factorial(5));\n"
                ),
                "rust": (
                    "fn factorial(n: u64) -> u64 {\n"
                    "    (1..=n).product()\n"
                    "}\n\n"
                    "fn main() {\n    println!(\"{}\", factorial(5));\n}\n"
                ),
            }
            code = templates.get(lang, templates["python"])
            fence = lang if lang in templates else "python"
            shown = lang if lang in templates else "python"
            return f"Factorial in {shown}:\n\n```{fence}\n{code}```"

        if "fibonacci" in q or "fibonnaci" in q:
            templates = {
                "python": (
                    "def fib(n):\n"
                    "    a, b = 0, 1\n"
                    "    for _ in range(n):\n        a, b = b, a + b\n"
                    "    return a\n\n"
                    "print([fib(i) for i in range(10)])\n"
                ),
                "c": (
                    "#include <stdio.h>\n\n"
                    "long fib(int n) {\n"
                    "    long a = 0, b = 1;\n"
                    "    for (int i = 0; i < n; i++) {\n"
                    "        long t = a + b; a = b; b = t;\n"
                    "    }\n    return a;\n}\n\n"
                    "int main(void) {\n"
                    "    for (int i = 0; i < 10; i++)\n"
                    "        printf(\"%ld%s\", fib(i), i == 9 ? \"\\n\" : \" \");\n"
                    "    return 0;\n}\n"
                ),
                "go": (
                    "package main\n\nimport \"fmt\"\n\n"
                    "func fib(n int) int {\n"
                    "    a, b := 0, 1\n"
                    "    for i := 0; i < n; i++ {\n        a, b = b, a+b\n    }\n"
                    "    return a\n}\n\n"
                    "func main() {\n"
                    "    for i := 0; i < 10; i++ {\n        fmt.Print(fib(i), \" \")\n    }\n"
                    "    fmt.Println()\n}\n"
                ),
                "rust": (
                    "fn fib(n: u32) -> u64 {\n"
                    "    let (mut a, mut b) = (0u64, 1u64);\n"
                    "    for _ in 0..n {\n        let t = a + b; a = b; b = t;\n    }\n"
                    "    a\n}\n\n"
                    "fn main() {\n"
                    "    for i in 0..10 {\n        print!(\"{} \", fib(i));\n    }\n"
                    "    println!();\n}\n"
                ),
            }
            code = templates.get(lang, templates["python"])
            fence = lang if lang in templates else "python"
            shown = lang if lang in templates else "python"
            return f"Fibonacci in {shown}:\n\n```{fence}\n{code}```"

        return None

    @classmethod
    def scaffold(cls, lang, text):
        """Generic runnable scaffold when no specialized template matches."""
        title = re.sub(r"\s+", " ", text.strip())[:72] or "program"
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", title).strip("_").lower() or "main"
        if lang == "c":
            return (
                f"Starter C program for: {title}\n\n```c\n"
                f"#include <stdio.h>\n"
                f"#include <stdlib.h>\n\n"
                f"/* TODO: flesh out logic for: {title} */\n"
                f"int main(int argc, char **argv) {{\n"
                f"    (void)argc; (void)argv;\n"
                f"    puts(\"{cls._escape('c', title)}\");\n"
                f"    return 0;\n"
                f"}}\n```\n"
                f"Build with: `cc -o {safe} {safe}.c && ./{safe}`"
            )
        if lang == "python":
            return (
                f"Starter Python program for: {title}\n\n```python\n"
                f"def main():\n"
                f"    # TODO: implement — {title}\n"
                f"    print({title!r})\n\n"
                f"if __name__ == '__main__':\n"
                f"    main()\n```"
            )
        if lang == "javascript":
            return (
                f"Starter JavaScript for: {title}\n\n```javascript\n"
                f"function main() {{\n"
                f"  // TODO: implement — {title}\n"
                f"  console.log({json.dumps(title)});\n"
                f"}}\n\nmain();\n```"
            )
        if lang == "go":
            return (
                f"Starter Go program for: {title}\n\n```go\n"
                f"package main\n\nimport \"fmt\"\n\n"
                f"func main() {{\n"
                f"    // TODO: implement — {title}\n"
                f"    fmt.Println(\"{cls._escape('go', title)}\")\n"
                f"}}\n```"
            )
        if lang == "rust":
            return (
                f"Starter Rust program for: {title}\n\n```rust\n"
                f"fn main() {{\n"
                f"    // TODO: implement — {title}\n"
                f"    println!(\"{cls._escape('rust', title)}\");\n"
                f"}}\n```"
            )
        # Fall back to a print program using a cleaned objective as message.
        message = cls._extract_message(text)
        return cls.print_program(lang, message)

    @classmethod
    def synthesize(cls, text):
        lang = cls.detect_lang(text)
        special = cls.special_program(lang, text)
        if special:
            return {"lang": lang, "kind": "special", "content": special}

        q = text.lower()
        # Prefer print synthesis for hello/print/say/write-message style asks.
        if re.search(r"\b(hello|print|say|echo|output|display)\b", q) or re.search(
            r"\bwrite\b.+\bin\b", q
        ):
            message = cls._extract_message(text)
            return {
                "lang": lang,
                "kind": "print",
                "content": cls.print_program(lang, message),
            }

        return {
            "lang": lang,
            "kind": "scaffold",
            "content": cls.scaffold(lang, text),
        }


class EngramMemory:
    """Engram conditional memory — sparsely accessed via token-based lookup."""

    def __init__(self):
        # Baked conditional memory tables (stand-in for 196B Engram params).
        self.tables = {
            "chip8": (
                "A tiny CHIP-8 emulator needs only a small virtual machine and a "
                "fetch/decode/execute loop.\n\n"
                "1. Create 4 KB of RAM. Programs normally begin at address 0x200.\n"
                "2. Add sixteen 8-bit registers V0-VF. VF is also used as a flag.\n"
                "3. Add the 16-bit index register I, program counter PC, a stack and SP.\n"
                "4. Create a 64x32 monochrome framebuffer and a 16-key keypad state.\n"
                "5. Add delay_timer and sound_timer; decrement them at about 60 Hz.\n"
                "6. Load the CHIP-8 font sprites into low memory.\n"
                "7. Fetch each opcode with (memory[PC] << 8) | memory[PC+1], then PC += 2.\n"
                "8. Decode the opcode using masks such as opcode & 0xF000.\n"
                "9. Implement the instruction families: 00E0/00EE, 1NNN, 2NNN, "
                "3XNN-5XY0, 6XNN, 7XNN, 8XY*, 9XY0, ANNN, BNNN, CXNN, DXYN, "
                "EX9E/EXA1 and FX**.\n"
                "10. For DXYN, XOR sprite pixels into the framebuffer and set VF when "
                "a lit pixel is erased by the XOR collision.\n"
                "11. Map sixteen host keys to CHIP-8 keys 0-F.\n"
                "12. Run CPU instructions faster than 60 Hz while updating timers/display "
                "on their own 60 Hz schedule.\n\n"
                "Minimal CPU skeleton:\n\n"
                "    opcode = (memory[pc] << 8) | memory[pc + 1]\n"
                "    pc += 2\n"
                "    top = opcode & 0xF000\n"
                "    x = (opcode >> 8) & 0xF\n"
                "    y = (opcode >> 4) & 0xF\n"
                "    nn = opcode & 0xFF\n"
                "    nnn = opcode & 0xFFF\n\n"
                "Start with 00E0, 1NNN, 6XNN, 7XNN and DXYN, then add the remaining "
                "instructions. Keep the CPU independent from Tkinter so the emulator "
                "core is easy to test."
            ),
            "python": (
                "For a small Python project, separate the state from the update loop. "
                "Represent state with lists/bytearrays, write small deterministic step "
                "functions, and keep the UI as a thin layer around the core."
            ),
            "coding": (
                "CatSeek R1 coding expert is online. Ask me to write, implement, or "
                "print programs in C, C++, Python, JavaScript, Go, Rust, Java, and more — "
                "I synthesize runnable source, not just advice."
            ),
            "math": (
                "For math, identify the known values and requested unknown, choose the relevant "
                "formula or operation, substitute carefully, compute in small steps, and verify "
                "the result using units, bounds, or a reverse calculation."
            ),
            "summary": (
                "To summarize text, preserve the main claim, the strongest supporting points, "
                "important numbers or constraints, and the conclusion. Remove repetition and "
                "minor examples without changing the author's meaning."
            ),
            "planning": (
                "Turn the goal into a short sequence of concrete milestones. Start with the "
                "smallest reversible step, identify dependencies, define what 'done' means for "
                "each milestone, then test or review before moving to the next."
            ),
            "chat": (
                "I'm CatSeek R1 1.x — a local Causal Encoder–Decoder MoE agent mirroring "
                "DeepSeek-V4.1-Flash (CED, CSA2, Engram, FP4 KV, DSpark). I can chat, "
                "organize ideas, explain baked concepts, help structure code and plans, "
                "and keep the current conversation in RAM."
            ),
            "architecture": (
                "CatSeek R1 1.x mirrors DeepSeek-V4.1-Flash:\n"
                f"- {N_LAYERS}-layer CED ({N_ENCODER} encoder + {N_DECODER} decoder)\n"
                f"- MoE: {N_SHARED_EXPERTS} shared + {N_ROUTED_EXPERTS} routed, "
                f"top-{N_ACTIVE_ROUTED} active\n"
                f"- Prefill activates ~{PREFILL_ACTIVE_B}B; decode ~{DECODE_ACTIVE_B}B "
                f"(backbone {BACKBONE_PARAMS_B}B + Engram {ENGRAM_PARAMS_B}B metaphor)\n"
                f"- CSA2 Full/Reindex/Reuse + Hierarchical Sparse Indexer\n"
                f"- FP4 global KV ≈ {KV_BYTES_PER_TOKEN} bytes/token; context up to "
                f"{CONTEXT_WINDOW:,} tokens\n"
                "- Single-Pass mHC residuals · DSpark speculative decode\n"
                "- Controllable reasoning effort 1–100\n"
                "This build is a pure-Python structural mirror — no cloud checkpoint."
            ),
            "generic": (
                "Break the objective into state, inputs, processing, outputs, and a main loop. "
                "Build the smallest working version first, test each component, then add features. "
                "Because this compact offline build has no pretrained language-model checkpoint, "
                "its open-domain knowledge is intentionally limited to Engram tables baked into Python."
            ),
        }
        # Map 384 routed expert slots onto Engram keys (cycled).
        keys = list(self.tables.keys())
        self.expert_map = {i: keys[i % len(keys)] for i in range(N_ROUTED_EXPERTS)}

    def lookup(self, key):
        return self.tables.get(key, self.tables["generic"])


class DeepSeekMoE:
    """1 shared + 384 routed experts; activate 6 routed experts per token."""

    def __init__(self, engram: EngramMemory):
        self.engram = engram
        self.coder = BitNetCoder()
        # Ternary-style gate vectors (BitNet b1.58 flavour retained inside MoE).
        self.gates = {
            "chip8": (1, 1, 1, 0, -1, 1, 1, 1),
            "python": (1, 0, 1, 1, 1, 0, -1, 1),
            "coding": (1, 1, 0, 1, 1, 0, 1, 0),
            "math": (0, 1, 1, -1, 1, 1, 0, 1),
            "summary": (1, 0, 0, 1, 1, 1, 0, 1),
            "planning": (0, 1, 1, 1, 0, 1, 1, 0),
            "chat": (1, 1, 0, 0, 1, 0, 1, 1),
            "architecture": (1, 1, 1, 1, 0, 1, 0, 1),
            "explain": (1, 1, 0, 1, 0, 1, 1, 0),
            "generic": (0, 1, 1, 0, 1, 0, 1, 1),
        }

    @staticmethod
    def _features(text):
        t = text.lower()
        return (
            1 if "chip" in t else -1,
            1 if "emulat" in t else -1,
            1 if "python" in t else 0,
            1 if any(w in t for w in ("build", "make", "create", "write", "code")) else 0,
            1 if any(w in t for w in ("deepseek", "v4.1", "flash", "ced", "moe", "csa2", "engram", "architecture", "bitnet")) else 0,
            1 if any(w in t for w in ("explain", "how", "what")) else 0,
            1 if len(t.split()) > 4 else 0,
            1,
        )

    @staticmethod
    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _is_greeting_only(text):
        """True only for bare greetings — not 'write hello cat in c'."""
        q = text.lower().strip()
        if BitNetCoder.is_coding_request(q):
            return False
        return bool(
            re.fullmatch(
                r"(hi|hey|hello|yo|sup|hiya|howdy)([!.?]|\s+(there|catseek|friend))?|"
                r"who are you\??|what are you\??|chat",
                q,
            )
        )

    def route(self, text):
        f = self._features(text)
        q = text.lower()

        # Coding intents beat greetings (fixes "hello cat" → chat false positive).
        if "chip-8" in q or "chip8" in q:
            primary = "chip8"
        elif BitNetCoder.is_coding_request(text):
            primary = "coding"
        elif any(w in q for w in ("deepseek", "v4.1", "flash", "ced", "csa2", "engram", "architecture", "moe", "bitnet")) and not BitNetCoder.is_coding_request(text):
            # "catseek" alone is often just the product name in a coding ask.
            if "catseek" in q and BitNetCoder.is_coding_request(text):
                primary = "coding"
            else:
                primary = "architecture"
        elif any(w in q for w in ("calculate", "math", "equation", "multiply", "divide", "percent")):
            primary = "math"
        elif any(w in q for w in ("summarize", "summary", "shorten")):
            primary = "summary"
        elif any(w in q for w in ("plan", "roadmap", "steps", "schedule")) and not BitNetCoder.is_coding_request(text):
            primary = "planning"
        elif self._is_greeting_only(text) or q in {"hello", "hi", "hey"}:
            primary = "chat"
        else:
            scores = {k: self._dot(f, w) for k, w in self.gates.items()}
            primary = max(scores, key=scores.get)

        # Select top-6 routed experts around the primary key.
        keys = list(self.engram.tables.keys())
        base = keys.index(primary) if primary in keys else 0
        routed = []
        for i in range(N_ACTIVE_ROUTED):
            eid = (base * 47 + i * 61) % N_ROUTED_EXPERTS
            routed.append((eid, self.engram.expert_map[eid]))
        shared = primary
        return shared, routed

    def forward(self, text):
        shared, routed = self.route(text)
        # Shared expert always contributes; routed experts vote by key frequency.
        votes = {shared: 2}
        for _eid, key in routed:
            votes[key] = votes.get(key, 0) + 1
        winner = max(votes, key=votes.get)

        # Coding expert synthesizes real source instead of Engram prose.
        if winner in {"coding", "python"} or BitNetCoder.is_coding_request(text):
            synth = self.coder.synthesize(text)
            content = synth["content"]
            winner = "coding"
            shared = "coding"
        else:
            content = self.engram.lookup(winner)

        return {
            "shared": shared,
            "routed": routed,
            "winner": winner,
            "content": content,
        }


class DSparkDecoder:
    """Semi-autoregressive draft generation with confidence-scheduled verification."""

    def draft(self, content, effort):
        # Higher effort → longer / more structured draft verification trail.
        depth = max(1, min(8, effort // 12))
        drafts = []
        for i in range(depth):
            drafts.append(f"draft[{i+1}/{depth}] confidence={0.55 + i * 0.05:.2f}")
        return drafts

    def verify(self, drafts, content):
        if not drafts:
            return content, 0.5
        conf = 0.55 + 0.05 * (len(drafts) - 1)
        return content, min(0.99, conf)


class CausalEncoderDecoder:
    """20-layer causal encoder + 20-layer decoder (CED)."""

    def __init__(self):
        self.kv = FP4KVCache()
        self.csa2 = CSA2Attention()
        self.mhc = SinglePassMHC()
        self.engram = EngramMemory()
        self.moe = DeepSeekMoE(self.engram)
        self.dspark = DSparkDecoder()

    def _tokenize(self, text):
        # Extremely small tokenizer for the toy runtime.
        parts = text.lower().replace("\n", " ").split()
        return parts[:4096] or ["<bos>"]

    def prefill(self, tokens):
        """Prefill activates encoder only (~8B metaphor); projects decoder KV."""
        streams = [0] * self.mhc.n_streams
        encoder_hidden = []
        for t_i, tok in enumerate(tokens):
            h = hash(tok) & 0x7FFFFFFF
            for layer in range(N_ENCODER):
                attn = self.csa2.attend(layer, "encoder", h, encoder_hidden or [(0, h, 0)])
                h = (h + len(attn.get("selected", [])) * (layer + 1)) & 0x7FFFFFFF
                streams = self.mhc.mix(streams, h)
            encoder_hidden.append(f"{tok}:{h}")
            self.kv.write(t_i, encoder_hidden[-1])
        # CED: project decoder global KV from final encoder hidden states.
        projected = self.kv.project_from_encoder(encoder_hidden)
        return {
            "phase": "prefill",
            "active_params_b": PREFILL_ACTIVE_B,
            "encoder_layers": N_ENCODER,
            "tokens": len(tokens),
            "kv_bytes": self.kv.footprint(),
            "projected_kv": len(projected),
            "streams": streams,
            "encoder_hidden": encoder_hidden,
        }

    def decode(self, prompt, prefill_state, effort):
        """Decode activates encoder+decoder path (~16B metaphor) + MoE + Engram."""
        moe_out = self.moe.forward(prompt)
        h = hash(moe_out["winner"]) & 0x7FFFFFFF
        streams = prefill_state["streams"]
        attn_trace = []
        for layer in range(N_DECODER):
            attn = self.csa2.attend(layer, "decoder", h, self.kv._hbm)
            attn_trace.append(attn["mode"])
            h = (h ^ (len(attn.get("selected", [])) << (layer % 8))) & 0x7FFFFFFF
            streams = self.mhc.mix(streams, h)

        drafts = self.dspark.draft(moe_out["content"], effort)
        content, confidence = self.dspark.verify(drafts, moe_out["content"])
        return {
            "phase": "decode",
            "active_params_b": DECODE_ACTIVE_B,
            "decoder_layers": N_DECODER,
            "moe": moe_out,
            "attn_modes": attn_trace,
            "drafts": drafts,
            "confidence": confidence,
            "content": content,
            "kv_bytes": self.kv.footprint(),
        }


class CatSeekR1Engine:
    """
    CatSeek R1 1.x — pure-Python structural mirror of DeepSeek-V4.1-Flash.

    Not a weight-compatible checkpoint. Ships CED / CSA2 / MoE / Engram /
    FP4-KV / mHC / DSpark orchestration and baked Engram knowledge inside
    this .py file. No model download, API, files, shell, numpy, or external
    package is required.
    """

    def __init__(self, reasoning_effort=DEFAULT_REASONING_EFFORT):
        self.name = MODEL_NAME
        self.arch = MODEL_ARCH
        self.reasoning_effort = max(1, min(100, int(reasoning_effort)))
        self.ced = CausalEncoderDecoder()

    def set_effort(self, effort):
        self.reasoning_effort = max(1, min(100, int(effort)))

    def infer(self, objective, history, step, max_steps, user_note=""):
        prompt = (user_note or objective).strip()
        tokens = self.ced._tokenize(prompt)

        # --- Prefill (encoder, ~8B active) ---
        prefill = self.ced.prefill(tokens)

        # --- Decode (decoder + MoE + Engram + DSpark, ~16B active) ---
        decode = self.ced.decode(prompt, prefill, self.reasoning_effort)

        if user_note:
            prefix = f"Regarding your note, “{user_note}”:\n\n"
        else:
            prefix = ""

        routed_ids = ",".join(str(e) for e, _ in decode["moe"]["routed"][:6])
        thought = (
            f"{self.name}: CED prefill={prefill['active_params_b']}B/"
            f"{prefill['encoder_layers']}L → decode={decode['active_params_b']}B/"
            f"{decode['decoder_layers']}L; MoE shared={decode['moe']['shared']} "
            f"routed=[{routed_ids}] → {decode['moe']['winner']}; "
            f"CSA2={decode['attn_modes'][0]}…{decode['attn_modes'][-1]}; "
            f"KV={decode['kv_bytes']}B; DSpark conf={decode['confidence']:.2f}; "
            f"effort={self.reasoning_effort}"
        )

        return {
            "thought": thought,
            "action": "DONE",
            "content": prefix + decode["content"],
        }


# Keep legacy alias so older call sites still resolve.
BitNetEngine = CatSeekR1Engine


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PersistentMemory:
    """
    App-owned conversation memory that survives close.

    Created the first time the OS launches / detects CatSeek R1.
    Agent tools remain FILES OFF — this store is only for the GUI shell.
    """

    def __init__(self, path=MEMORY_FILE):
        self.path = Path(path)
        self.first_boot = False
        self.data = {
            "version": MEMORY_VERSION,
            "model": MODEL_NAME,
            "first_seen": None,
            "updated": None,
            "objective": "Explain how to build a tiny CHIP-8 emulator.",
            "effort": DEFAULT_REASONING_EFFORT,
            "log": [],
            "agent_history": [],
        }
        self.load_or_create()

    def load_or_create(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.first_boot = True
            self.data["first_seen"] = _utc_now()
            self.data["updated"] = self.data["first_seen"]
            self.save()
            return self.data

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data.update(raw)
                self.data["version"] = MEMORY_VERSION
                self.data["model"] = MODEL_NAME
                if not self.data.get("first_seen"):
                    self.data["first_seen"] = _utc_now()
                    self.first_boot = True
            else:
                self.first_boot = True
                self.data["first_seen"] = _utc_now()
        except (OSError, json.JSONDecodeError, TypeError):
            self.first_boot = True
            self.data["first_seen"] = _utc_now()
            self.data["updated"] = self.data["first_seen"]
        self.save()
        return self.data

    def save(self):
        self.data["updated"] = _utc_now()
        self.data["log"] = list(self.data.get("log") or [])[-MAX_LOG_ENTRIES:]
        self.data["agent_history"] = list(self.data.get("agent_history") or [])[-MAX_LOG_ENTRIES:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def append_log(self, who, text):
        self.data.setdefault("log", []).append({"who": who, "text": text.strip()})
        self.save()

    def set_objective(self, objective):
        self.data["objective"] = objective
        self.save()

    def set_effort(self, effort):
        self.data["effort"] = int(effort)
        self.save()

    def set_agent_history(self, history):
        self.data["agent_history"] = list(history or [])
        self.save()

    def clear(self):
        first_seen = self.data.get("first_seen") or _utc_now()
        self.data = {
            "version": MEMORY_VERSION,
            "model": MODEL_NAME,
            "first_seen": first_seen,
            "updated": _utc_now(),
            "objective": "Explain how to build a tiny CHIP-8 emulator.",
            "effort": DEFAULT_REASONING_EFFORT,
            "log": [],
            "agent_history": [],
            "cleared_at": _utc_now(),
        }
        self.save()

    def summary(self):
        n = len(self.data.get("log") or [])
        return f"{self.path} · {n} entries · first_seen={self.data.get('first_seen')}"


class CatSeekAgent:
    def __init__(self, engine):
        self.engine = engine
        self.objective = ""
        self.history = []
        self.step = 0
        self.max_steps = 12

    def reset(self, objective, history=None):
        self.objective = objective.strip()
        self.history = list(history or [])
        self.step = 0  # steps in this run; history may already hold prior turns

    def context(self):
        if not self.history:
            return "(no previous steps)"
        return "\n".join(
            f"{i+1}. {item['action']}: {item['content']}"
            for i, item in enumerate(self.history[-10:])
        )

    @staticmethod
    def parse_action(raw):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {
                "thought": "The model response was not valid action JSON.",
                "action": "ANSWER",
                "content": raw.strip(),
            }

        action = str(data.get("action", "ANSWER")).upper()
        if action not in {"THINK", "ASK", "ANSWER", "DONE"}:
            action = "ANSWER"

        return {
            "thought": str(data.get("thought", "")).strip(),
            "action": action,
            "content": str(data.get("content", "")).strip(),
        }

    def next_step(self, user_note=""):
        self.step += 1
        result = self.engine.infer(
            self.objective, self.history, self.step, self.max_steps, user_note
        )
        self.history.append(result)
        return result


class CatSeekGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("920x680")
        self.minsize(720, 520)

        self.events = queue.Queue()
        self.running = False
        self.agent = None
        self.memory = PersistentMemory()
        self._restoring = False

        self.status = tk.StringVar(
            value=f"Ready · {MODEL_NAME} · {MODEL_ARCH} · LOCAL · PERSISTENT MEMORY · FILES OFF"
        )

        self._build_ui()
        self._restore_memory()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    def _ready_status(self):
        return f"Ready · {MODEL_NAME} · {MODEL_ARCH} · LOCAL · PERSISTENT MEMORY · FILES OFF"

    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        title_row = ttk.Frame(outer)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="CatSeek R1 1.x", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(title_row, text="[c] Kondo Solutions 1999-2026 · FILES OFF").pack(side="right")

        ttk.Label(
            outer,
            text="DeepSeek-V4.1-Flash mirror · CED 20+20 · MoE 384e/top-6 · CSA2 · Engram · FP4 KV · DSpark · persistent memory",
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(outer, text="Objective").pack(anchor="w", pady=(12, 3))
        self.objective = tk.Text(outer, height=3, wrap="word")
        self.objective.pack(fill="x")
        self.objective.insert("1.0", self.memory.data.get("objective") or "Explain how to build a tiny CHIP-8 emulator.")
        self.objective.bind("<<Modified>>", self._on_objective_modified)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(buttons, text="Run Agent", command=self.start_agent)
        self.run_btn.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.stop_agent).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear Log", command=self.clear_log).pack(side="left")
        ttk.Button(buttons, text="Clear Memory", command=self.clear_memory).pack(side="left", padx=6)
        ttk.Label(buttons, text="R1 1.x · CED-MoE · LOCAL · DISK").pack(side="right")

        effort_row = ttk.Frame(outer)
        effort_row.pack(fill="x", pady=(0, 6))
        ttk.Label(effort_row, text="Reasoning effort").pack(side="left")
        start_effort = int(self.memory.data.get("effort") or DEFAULT_REASONING_EFFORT)
        self.effort = tk.IntVar(value=start_effort)
        self.effort_scale = ttk.Scale(
            effort_row, from_=1, to=100, orient="horizontal",
            variable=self.effort, command=self._on_effort,
        )
        self.effort_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.effort_label = ttk.Label(effort_row, text=str(start_effort))
        self.effort_label.pack(side="left")

        log_frame = ttk.Frame(outer)
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        input_row = ttk.Frame(outer)
        input_row.pack(fill="x", pady=(8, 0))
        self.user_input = ttk.Entry(input_row)
        self.user_input.pack(side="left", fill="x", expand=True)
        self.user_input.bind("<Return>", lambda _e: self.send_note())
        ttk.Button(input_row, text="Send Note", command=self.send_note).pack(side="left", padx=(6, 0))

        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            outer,
            text="Chat · coding · explanations · math · summaries · planning · architecture · memory survives close",
        ).pack(anchor="w", pady=(2, 0))

    def _restore_memory(self):
        self._restoring = True
        saved_log = list(self.memory.data.get("log") or [])
        pending_persist = []

        if self.memory.first_boot:
            msg = (
                f"First OS detect of {MODEL_NAME}. Persistent memory initialized at "
                f"{self.memory.path}. Conversation will survive app close. "
                f"Agent tools remain FILES OFF (no shell / arbitrary file access)."
            )
            self._append("SYSTEM", msg, persist=False)
            pending_persist.append(("SYSTEM", msg))
        elif saved_log:
            for entry in saved_log:
                who = entry.get("who", "SYSTEM")
                text = entry.get("text", "")
                self._append(who, text, persist=False)
            msg = (
                f"Restored persistent memory ({len(saved_log)} entries) from "
                f"{self.memory.path}."
            )
            self._append("SYSTEM", msg, persist=False)
            pending_persist.append(("SYSTEM", msg))
        else:
            msg = (
                f"{MODEL_NAME} ready ({MODEL_ARCH}). Persistent memory at "
                f"{self.memory.path}. "
                f"CED {N_ENCODER}+{N_DECODER}, MoE {N_ROUTED_EXPERTS}e/top-{N_ACTIVE_ROUTED}, "
                f"FP4 KV {KV_BYTES_PER_TOKEN} B/tok. Agent FILES OFF."
            )
            self._append("SYSTEM", msg, persist=False)
            pending_persist.append(("SYSTEM", msg))

        self.status.set(self._ready_status())
        self._restoring = False
        for who, text in pending_persist:
            self.memory.append_log(who, text)

    def _on_objective_modified(self, _event=None):
        if self._restoring:
            self.objective.edit_modified(False)
            return
        if self.objective.edit_modified():
            self.memory.set_objective(self.objective.get("1.0", "end").strip())
            self.objective.edit_modified(False)

    def _on_effort(self, _value=None):
        v = int(float(self.effort.get()))
        self.effort_label.configure(text=str(v))
        if not self._restoring:
            self.memory.set_effort(v)

    def _append(self, who, text, persist=True):
        body = text.strip()
        self.log.configure(state="normal")
        self.log.insert("end", f"\n[{who}]\n{body}\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        if persist and PERSISTENT_MEMORY and not self._restoring:
            self.memory.append_log(who, body)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._append("SYSTEM", "Log view cleared (persistent memory kept on disk).", persist=True)

    def clear_memory(self):
        self.running = False
        self.agent = None
        self.memory.clear()
        self._restoring = True
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.objective.delete("1.0", "end")
        self.objective.insert("1.0", self.memory.data["objective"])
        self.effort.set(int(self.memory.data["effort"]))
        self.effort_label.configure(text=str(int(self.memory.data["effort"])))
        self._restoring = False
        self.run_btn.configure(state="normal")
        self.status.set(self._ready_status())
        self._append(
            "SYSTEM",
            f"Persistent memory cleared. Store reset at {self.memory.path}.",
            persist=True,
        )

    def make_agent(self):
        return CatSeekAgent(CatSeekR1Engine(reasoning_effort=int(float(self.effort.get()))))

    def start_agent(self):
        if self.running:
            return
        objective = self.objective.get("1.0", "end").strip()
        if not objective:
            self._append("SYSTEM", "Enter an objective first.")
            return

        self.memory.set_objective(objective)
        self.agent = self.make_agent()
        # Keep prior agent turns in the session so memory compounds across runs.
        prior = list(self.memory.data.get("agent_history") or [])
        self.agent.reset(objective, history=prior)
        self.running = True
        self.run_btn.configure(state="disabled")
        self.status.set(f"Agent running… effort={int(float(self.effort.get()))}")
        self._append("OBJECTIVE", objective)
        threading.Thread(target=self._agent_loop, daemon=True).start()

    def stop_agent(self):
        self.running = False
        if self.agent:
            self.memory.set_agent_history(self.agent.history)
        self.status.set(f"Stopped · {MODEL_NAME} · PERSISTENT MEMORY · FILES OFF")
        self.run_btn.configure(state="normal")

    def send_note(self):
        note = self.user_input.get().strip()
        if not note:
            return
        self.user_input.delete(0, "end")
        self._append("YOU", note)
        if self.agent:
            self.events.put(("note", note))

    def _agent_loop(self):
        pending_note = ""
        try:
            while self.running and self.agent.step < self.agent.max_steps:
                try:
                    while True:
                        kind, value = self.events.get_nowait()
                        if kind == "note":
                            pending_note = value
                except queue.Empty:
                    pass

                result = self.agent.next_step(pending_note)
                pending_note = ""
                self.events.put(("result", result))

                if result["action"] in {"ASK", "DONE"}:
                    break

            self.events.put(("finished", None))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()

                if kind == "result":
                    action = value["action"]
                    thought = value["thought"]
                    content = value["content"]
                    if thought:
                        self._append("PLAN", thought)
                    self._append(action, content or "(no content)")
                    if self.agent:
                        self.memory.set_agent_history(self.agent.history)

                elif kind == "error":
                    self._append("ERROR", value)
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.status.set("CatSeek R1 1.x engine error")

                elif kind == "finished":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    if self.agent:
                        self.memory.set_agent_history(self.agent.history)
                    self.status.set(self._ready_status())

                elif kind == "note":
                    self.events.put((kind, value))
                    break
        except queue.Empty:
            pass

        self.after(100, self._poll_events)

    def _on_close(self):
        try:
            self.memory.set_objective(self.objective.get("1.0", "end").strip())
            self.memory.set_effort(int(float(self.effort.get())))
            if self.agent:
                self.memory.set_agent_history(self.agent.history)
            self.memory.save()
        except OSError:
            pass
        self.destroy()


if __name__ == "__main__":
    CatSeekGUI().mainloop()
