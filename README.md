# plog-ai

> AI-powered logical analysis tool for detecting contradictions, inconsistencies, and fallacies in natural language texts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**plog-ai** translates natural language documents into formal logic statements, then uses SAT solving to detect contradictions, fallacies, and logical inconsistencies. It combines the power of LLMs for semantic understanding with the rigor of formal verification.

## Features

- **Automated Text Analysis** — Feed documents to an AI assistant that converts prose into formal logic
- **Formal Verification** — Uses Z3 SAT solver to detect logical contradictions
- **Fallacy Detection** — Identifies and reports logical fallacies using tagged formulas (`FALLACY_*` atoms, `f_FALLACY_*` formulas)
- **Plain English Reports** — Human-readable analysis output explaining what's wrong and why
- **Interactive Simulation** — Explore "what-if" scenarios by toggling facts, formulas, and fallacy detection
- **Multiple Logic Systems** — Supports both propositional and predicate (first-order) logic

## Use Cases

- Validating requirements documents and user stories for contradictions
- Analyzing political speeches and public statements for logical consistency
- Reviewing research papers and technical documentation
- Exploring hypothetical scenarios in complex arguments

## Quick Start

### Prerequisites

- Python 3.10+
- A chat-based AI with tool execution capabilities (Claude, ChatGPT, etc.)

### Installation

```bash
git clone https://github.com/yourusername/plog-ai.git
cd plog-ai/python

pip install PyYAML lark z3-solver simple-term-menu
```

Verify the installation:

```bash
python plog.py
```

## Usage

### Text Analysis with AI

1. Open your AI chat interface (claude.ai, chatgpt.com, etc.)
2. Start a new conversation with this prompt:

```
Follow the attached instructions and process the attached document named <your document name here>
```

3. Attach the following files:
- Your document to analyze (`.md` format recommended)
- `python/plog-grammar.txt` — Grammar specification
- `python/plog.py` — Validation script
- `prompts/instructions.md` - Analysis process instructions

The AI will produce:
- `logic-results.plog` — Your text translated to formal logic
- `logic-analysis.md` — Validation results
- A plain-English interpretation of any inconsistencies found

### Direct Analysis (Non-Interactive)

Analyze a `.plog` file and get a plain English report:

```bash
python plog.py document.plog
```

The tool will output:
1. **Fallacies detected** (if any) — Listed first for immediate visibility
2. **Facts stated** — What the text claims as true
3. **Logical connections** — The reasoning/inferences made
4. **Contradictions** (if any) — Which statements conflict
5. **Summary** — Plain English bottom line

Exit codes: `0` = consistent, `1` = inconsistencies or fallacies found

### Interactive Simulation

Explore "what-if" scenarios by toggling facts and formulas:

```bash
python plog.py --interactive document.plog
# or
python plog.py -i document.plog
```

**Interactive features:**
- **Toggle Facts** — Switch between TRUE/FALSE to test scenarios
- **Toggle Formulas** — Turn reasoning rules ON/OFF
- **Toggle Fallacy Detection** — Mark fallacies as ON (consider) or OFF (ignore)
- **Evaluate** — Run Z3 analysis on current configuration
- **Detailed Analysis** — View full configuration and results
- **Reset** — Return to original state

### Example Output (Non-Interactive)

```
======================================================================
 🔍 TEXT LOGIC CHECKER - ANALYSIS REPORT
======================================================================

 ⚠️ FALLACIES DETECTED: The reasoning contains logical fallacies
----------------------------------------------------------------------

======================================================================
 🚩 LOGICAL FALLACIES DETECTED
======================================================================

 1. 🚩 Appeal to popularity - 'Everyone knows' does not establish truth
 2. 🚩 Slippery slope - No logical necessity that recycling leads to banning cars
 3. 🚩 False dilemma - Presents only two options when others exist

----------------------------------------------------------------------

📋 WHAT THE TEXT STATES AS FACTS:
----------------------------------------------------------------------
 1. "Everyone knows the recycling plan will bankrupt our town"
 2. "There is a new recycling plan"
 ...

🔗 THE LOGIC PRESENT IN THE TEXT:
----------------------------------------------------------------------
 1. IF "Everyone knows X", THEN "X is true"
 2. IF "We allow recycling", THEN "They will ban cars"
 ...

================================================================================
 📝 SUMMARY
======================================================================

⚠️ BOTTOM LINE: This text contains 3 logical fallacy(ies). 
The author uses flawed reasoning techniques to persuade.

======================================================================
```

### Example Interactive Session

```
══════════════════════════════════════════════════════════════════════
 🔬 WHAT-IF SIMULATOR - Interactive Mode
══════════════════════════════════════════════════════════════════════

📄 Loaded: argument.plog

══════════════════════════════════════════════════════════════════════
 📋 FACTS (4 items)
══════════════════════════════════════════════════════════════════════
 1    ✅ TRUE     "The defendant was at the scene"
 2    ❌ FALSE    "The defendant has an alibi"        ← CHANGED
 ...

══════════════════════════════════════════════════════════════════════
 🚩 FALLACY DETECTION (2 items)
══════════════════════════════════════════════════════════════════════
 T1   🚩 ON       Appeal to authority
 T2   ⚪ OFF      Ad hominem                          ← NOT A FALLACY

══════════════════════════════════════════════════════════════════════
 ⚡ STATUS: ✅ CONSISTENT
══════════════════════════════════════════════════════════════════════
```

## Project Structure

```
plog-ai/
├── plog/                # Example .plog files
│   ├── mlk-war-pred.plog
│   ├── test-*.plog
│   ├── un-speech-pred.plog
│   └── ...
├── prompts/
│   ├── instructions.md   # AI instructions prompt 
├── python/
│   ├── plog.py  # Main tool: parser, Z3 translator, analyzer, simulator
│   └── plog-grammar.txt          # Grammar specification for .plog format
└── texts/               # Example source documents
   ├── mlk-war.md
   ├── un-speech.md
   └── ...
```

## The `.plog` Format

A simple declarative format for logic statements:

```plog
# Atoms represent facts (assumed TRUE by default)
ATOM socrates_is_man: "Socrates is a man"
ATOM socrates_is_mortal: "Socrates is mortal"
ATOM men_are_mortal: "All men are mortal"

# Formulas express logical relationships
FORMULA f_mortality: (socrates_is_man & men_are_mortal) -> socrates_is_mortal
```

### Fallacy Tagging Convention

To enable fallacy detection, use these naming conventions:

```plog
# FALLACY_* atoms are hidden metadata (descriptions of fallacy types)
ATOM FALLACY_appeal_to_popularity: "FALLACY: Appeal to popularity - 'Everyone knows' does not establish truth"

# f_FALLACY_* formulas trigger fallacy detection
FORMULA f_FALLACY_appeal_to_popularity: everyone_knows_X -> FALLACY_appeal_to_popularity
```

**Supported operators:**
| Operator | Symbols | Description |
|----------|---------|-------------|
| NOT | `~` `¬` `!` | Negation |
| AND | `&` `∧` | Conjunction |
| OR | `\|` `∨` | Disjunction |
| XOR | `^` `⊕` | Exclusive or |
| NAND | `!&` `↑` | Not-and |
| NOR | `!\|` `↓` | Not-or |
| IMPLIES | `->` `→` | Implication |
| IFF | `<->` `↔` | Biconditional |

## How It Works

1. **Translation** — An LLM reads your document and converts statements into `.plog` format
2. **Parsing** — Lark parser validates syntax against `plog-grammar.txt` grammar
3. **Categorization** — Items are separated into:
  - Regular atoms (facts) — assumed TRUE
  - Regular formulas (reasoning) — logical connections
  - Fallacy atoms (`FALLACY_*`) — hidden metadata
  - Fallacy tag formulas (`f_FALLACY_*`) — detection rules
4. **Fallacy Detection** — Tag formulas are evaluated to find active fallacies
5. **Consistency Check** — Z3 SAT solver verifies logical consistency:
    - **SAT** → No contradictions detected
    - **UNSAT** → Contradiction found; minimal conflict set reported
6. **Reporting** — Results presented in plain English with:
  - Fallacies listed first (most actionable)
  - Contradictions explained with involved formulas
  - Human-readable summary

## Examples

The `plog/` directory contains analyzed examples:

| File | Description |
|------|-------------|
| `fallacies-example.plog` | Common logical fallacies demonstration |
| `mlk-war.plog` | MLK's "Declaration of Independence from the War in Vietnam" |
| `sojourner-aint-woman.plog` | Sojourner Truth's "Ain't I a Woman?" |
| `test-*.plog` | Different test cases |
| `un-speech-pred.plog` | UN General Assembly speech analysis |
| `wilson-war.plog` | Woodrow Wilson, "War Message" |


## Limitations

- Reports the **minimal conflict set**, not necessarily all contradictions
- Requires manual review of AI-generated translations for accuracy
- Best suited for argumentative/declarative texts
- Fallacy detection requires proper tagging in the `.plog` file

## Roadmap / WIP

- [ ] Improve explanation of **why** specific facts/formulas cause contradictions
- [ ] Add support for analyzing multiple files
- [ ] Create SKILL integration for AI agents (live document review)
- [ ] Add more fallacy detection templates
- [ ] Web interface for non-technical users

> **Note:** This tool was developed with assistance from Claude and ChatGPT. 
> Code continues to be refined for clarity and maintainability.

## Contributing

This is an experimental project exploring the intersection of LLMs and formal verification. Contributions, ideas, and feedback are welcome!

## License

MIT

## Acknowledgments

- [Lark](https://github.com/lark-parser/lark) — Parsing library
- [Z3](https://github.com/Z3Prover/z3) — Theorem prover
- [Simple Terminal Menu](https://github.com/IngoMeyer441/simple-term-menu) - Interactive terminal menus
- Claude and ChatGPT — Code generation assistance