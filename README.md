# plog-ai

> AI-powered logical analysis tool for detecting inconsistencies and fallacies in natural language texts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**plog-ai** translates natural language documents into formal logic statements, then uses SAT solving to detect contradictions, fallacies, and logical inconsistencies. It combines the power of LLMs for semantic understanding with the rigor of formal verification.

## Features

- **Automated Text Analysis** — Feed documents to an AI assistant that converts prose into formal logic
- **Formal Verification** — Uses Z3 theorem prover to detect logical contradictions
- **Interactive Simulation** — Explore "what-if" scenarios by toggling facts and observing effects
- **Multiple Logic Systems** — Supports both propositional and predicate (first-order) logic
- **Fallacy Detection** — Identifies logical fallacies in arguments and reasoning

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
git clone https://github.com/anteloc/plog-ai.git
cd plog-ai/python

pip install PyYAML lark z3-solver
```

Verify the installation:

```bash
python plog.py --help
```

## Usage

### Text Analysis with AI

1. Open your AI chat interface (claude.ai, chatgpt.com, etc.)
2. Start a new conversation with this prompt:
   ```
   Follow the attached instructions and process the attached document.
   ```
3. Attach the following files:
   - Your document to analyze (`.md` format recommended)
   - `python/plog.lark` — Grammar specification
   - `python/plog.py` — Validation script
   - One of the instruction files from `prompts/`:
     - `instructions-pred-logic.md` for predicate logic analysis
     - `instructions-prop-logic.md` for propositional logic analysis

The AI will produce:
- `logic-results.plog` — Your text translated to formal logic
- `logic-analysis.json` — Validation results
- A plain-English interpretation of any inconsistencies found

### Interactive Simulation

Run the simulator to explore "what-if" scenarios on any `.plog` file:

```bash
python plog.py -i logic-results.plog
```

**Simulation commands:**
| Command | Action |
|---------|--------|
| `[tag]` | Toggle negation on a fact or formula (e.g., `a2`) |
| `e` | Evaluate current state for contradictions |
| `q` | Quit simulation |

### Example Session

```
================================================================================
FACTS
================================================================================
    [ a1] plato_is_man       : TRUE  - "Plato is a man"
    [ a2] plato_is_not_animal: TRUE  - "Plato is not an animal"

================================================================================
FORMULAE
================================================================================
    [ f1] f_plato_is_man_not_animal:
         (a1) plato_is_man -> (a2) plato_is_not_animal

================================================================================
> e

EVALUATION RESULT: SAT (No contradictions found)
```

Toggle `a2` to FALSE and re-evaluate:

```
> a2
> e

CONFLICTING_FACTS:
- (a1) plato_is_man: AFFIRMING "Plato is a man"
- (a2 ~) plato_is_not_animal: DENYING "Plato is not an animal"

CONFLICTING_FORMULAE:
- (f1) UNSATISFIABLE: plato_is_man -> ~plato_is_not_animal
```

## Project Structure

```
plog-ai/
├── python/
│   ├── plog.py          # Main tool: parser, Z3 translator, simulator
│   └── plog.lark        # Grammar specification for .plog format
├── prompts/
│   ├── instructions-pred-logic.md   # AI instructions (predicate logic)
│   └── instructions-prop-logic.md   # AI instructions (propositional logic)
├── plog/                # Example .plog files
│   ├── mlk-war-pred.plog
│   ├── un-speech-pred.plog
│   └── ...
└── texts/               # Example source documents
    ├── mlk-war.md
    ├── un-speech.md
    └── ...
```

## The `.plog` Format

A simple declarative format for logic statements:

```plog
# Atoms represent facts (assumed TRUE by default)
ATOM plato_is_man: "Plato is a man"
ATOM plato_is_mortal: "Plato is mortal"
ATOM men_are_mortal: "All men are mortal"

# Formulas express logical relationships
FORMULA f_mortality: (plato_is_man & men_are_mortal) -> plato_is_mortal
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

1. **Translation** — An LLM reads your document and converts statements into `.plog` format atoms and formulas
2. **Parsing** — The Lark parser validates syntax against the `plog.lark` grammar
3. **Transformation** — Parse trees are converted to Z3 boolean expressions
4. **Solving** — Z3 checks satisfiability:
   - **SAT** → No contradictions detected
   - **UNSAT** → Contradiction found; conflicting facts/formulas reported

## Examples

The `plog/` directory contains analyzed examples:

| File | Description |
|------|-------------|
| `mlk-war-pred.plog` | MLK's "Declaration of Independence from the War in Vietnam" |
| `un-speech-pred.plog` | UN General Assembly speech analysis |
| `sojourner-aint-woman-pred.plog` | Sojourner Truth's "Ain't I a Woman?" |
| `fallacies-example-pred.plog` | Common logical fallacies demonstration |

## Limitations

- Reports only the **first** inconsistency found, not all of them
- Requires manual review of AI-generated translations for accuracy
- Best suited for argumentative/declarative texts

## Contributing

This is an experimental project exploring the intersection of LLMs and formal verification. Contributions, ideas, and feedback are welcome!

## License

MIT

## Acknowledgments

- [Lark](https://github.com/lark-parser/lark) — Parsing library
- [Z3](https://github.com/Z3Prover/z3) — Theorem prover
- Claude and ChatGPT — Code generation assistance