"""
Plog What-If Simulator - Interactive Mode with Fallacy Detection

Allows users to interactively toggle facts (TRUE/FALSE), formulas (ON/OFF),
and fallacy detection (ON/OFF) to explore "what-if" scenarios and understand 
logical contradictions and fallacious reasoning.

Usage:
    python plog.py <plog_file>
    python plog.py --interactive <plog_file>

Dependencies:
    pip install lark z3-solver simple-term-menu

Conventions:
    - FALLACY_* atoms: Hidden metadata marking fallacy types
    - f_FALLACY_* formulas: Fallacy detection rules (toggleable)
"""

import argparse
import sys
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from lark import Lark, Transformer, v_args
from z3 import (
    Bool, And, Or, Not, Xor, Implies, BoolVal,
    Solver, sat, unsat, BoolRef
)

# Conditional import for simple-term-menu (graceful fallback)
try:
    from simple_term_menu import TerminalMenu
    HAS_TERMINAL_MENU = True
except ImportError:
    HAS_TERMINAL_MENU = False


# =============================================================================
# Constants - Fallacy Detection Prefixes
# =============================================================================

FALLACY_ATOM_PREFIX = "FALLACY_"
FALLACY_TAG_FORMULA_PREFIX = "f_FALLACY_"


# =============================================================================
# Data Classes
# =============================================================================

class FormulaState(Enum):
    ON = "on"      # Use formula as-is
    OFF = "off"    # Ignore formula (don't add to solver)


@dataclass
class SimulationState:
    """Holds the current state of the simulation."""
    fact_values: dict = field(default_factory=dict)
    formula_states: dict = field(default_factory=dict)
    fallacy_tag_states: dict = field(default_factory=dict)
    last_evaluation: Optional[dict] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a scenario."""
    is_consistent: bool
    conflicting_formulas: list
    details: str
    active_fallacies: list = field(default_factory=list)


@dataclass
class DetectedFallacy:
    """Represents a detected fallacy."""
    tag_formula_id: str
    fallacy_atom_name: str
    fallacy_description: str
    trigger_description: str


@dataclass 
class PlainEnglishReport:
    """A report designed for non-experts."""
    is_consistent: bool
    verdict: str
    facts_summary: str
    reasoning_summary: str
    fallacies_detected: list
    problems_found: list[str]
    detailed_explanations: list[str]
    bottom_line: str


# =============================================================================
# Lark Transformer: Parse Tree -> Z3 Expressions
# =============================================================================

class PlogToZ3(Transformer):
    """Transforms parse tree into Z3 expressions."""

    def __init__(self):
        super().__init__()
        self.atoms = {}
        self.formulas = {}

    def start(self, items):
        return {'atoms': self.atoms, 'formulas': self.formulas}

    def statement(self, items):
        return items[0] if items else None

    @v_args(inline=True)
    def atom_decl(self, name, description):
        atom_name = str(name)
        self.atoms[atom_name] = {'z3_var': Bool(atom_name), 'description': description}
        return ('atom', atom_name)

    @v_args(inline=True)
    def formula_decl(self, formula_id, expr):
        fid = str(formula_id)
        self.formulas[fid] = expr
        return ('formula', fid, expr)

    def desc(self, items):
        return items[0]

    def ESCAPED_STRING(self, token):
        return str(token)[1:-1]

    def TRIPLE_STRING(self, token):
        return str(token)[3:-3]

    @v_args(inline=True)
    def atom_ref(self, name):
        atom_name = str(name)
        if atom_name not in self.atoms:
            self.atoms[atom_name] = {'z3_var': Bool(atom_name), 'description': None}
        return self.atoms[atom_name]['z3_var']

    def true(self, items):
        return BoolVal(True)

    def false(self, items):
        return BoolVal(False)

    def neg(self, items):
        return Not(items[-1])

    def andlike(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            if op in ('&', '∧'):
                result = And(result, right)
            elif op in ('!&', '↑'):
                result = Not(And(result, right))
            i += 2
        return result

    def orlike(self, items):
        if len(items) == 1:
            return items[0]
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            if op in ('|', '∨'):
                result = Or(result, right)
            elif op in ('!|', '↓'):
                result = Not(Or(result, right))
            elif op in ('^', '⊕'):
                result = Xor(result, right)
            i += 2
        return result

    def implies(self, items):
        if len(items) == 1:
            return items[0]
        operands = [item for item in items if not hasattr(item, 'type')]
        if len(operands) == 1:
            return operands[0]
        result = operands[-1]
        for i in range(len(operands) - 2, -1, -1):
            result = Implies(operands[i], result)
        return result

    def iff(self, items):
        if len(items) == 1:
            return items[0]
        operands = [item for item in items if not hasattr(item, 'type')]
        if len(operands) == 1:
            return operands[0]
        result = operands[0]
        for i in range(1, len(operands)):
            result = And(Implies(result, operands[i]), Implies(operands[i], result))
        return result


# =============================================================================
# Parser
# =============================================================================

class PlogParser:
    """Parser for .plog files."""

    def __init__(self, grammar_path: str = None):
        if grammar_path is None:
            grammar_path = Path(__file__).parent / 'plog.lark'
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        self.lark_parser = Lark(grammar, parser='earley', start='start')

    def parse_string(self, text: str) -> dict:
        tree = self.lark_parser.parse(text)
        transformer = PlogToZ3()
        return transformer.transform(tree)

    def parse_file(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            return self.parse_string(f.read())


# =============================================================================
# Parsed Result Categorizer
# =============================================================================

class CategorizedPlog:
    """Categorizes parsed plog items into regular items and fallacy-related items."""
    
    def __init__(self, parsed_result: dict):
        self.all_atoms = parsed_result['atoms']
        self.all_formulas = parsed_result['formulas']
        
        self.regular_atoms = {}
        self.fallacy_atoms = {}
        for name, info in self.all_atoms.items():
            if name.startswith(FALLACY_ATOM_PREFIX):
                self.fallacy_atoms[name] = info
            else:
                self.regular_atoms[name] = info
        
        self.regular_formulas = {}
        self.fallacy_tag_formulas = {}
        for fid, expr in self.all_formulas.items():
            if fid.startswith(FALLACY_TAG_FORMULA_PREFIX):
                self.fallacy_tag_formulas[fid] = expr
            else:
                self.regular_formulas[fid] = expr
    
    def get_fallacy_description(self, fallacy_atom_name: str) -> str:
        if fallacy_atom_name in self.fallacy_atoms:
            desc = self.fallacy_atoms[fallacy_atom_name].get('description', '')
            if desc:
                return desc
        name = fallacy_atom_name.replace(FALLACY_ATOM_PREFIX, '')
        return name.replace('_', ' ').title()
    
    def get_fallacy_atom_for_tag(self, tag_formula_id: str) -> Optional[str]:
        if tag_formula_id not in self.fallacy_tag_formulas:
            return None
        expr = self.fallacy_tag_formulas[tag_formula_id]
        
        def find_fallacy_atoms(e) -> list[str]:
            found = []
            if isinstance(e, BoolRef):
                if e.num_args() == 0:
                    name = str(e)
                    if name.startswith(FALLACY_ATOM_PREFIX):
                        found.append(name)
                else:
                    for i in range(e.num_args()):
                        found.extend(find_fallacy_atoms(e.arg(i)))
            return found
        
        fallacy_atoms = find_fallacy_atoms(expr)
        return fallacy_atoms[0] if fallacy_atoms else None


# =============================================================================
# Plain English Converter
# =============================================================================

class PlainEnglishConverter:
    """Converts Z3 expressions to plain English."""

    def __init__(self, atoms: dict):
        self.atoms = atoms

    def get_fact_description(self, atom_name: str) -> str:
        if atom_name in self.atoms and self.atoms[atom_name]['description']:
            return f'"{self.atoms[atom_name]["description"]}"'
        return f'"{atom_name.replace("_", " ")}"'

    def expr_to_plain_english(self, expr, depth=0) -> str:
        expr_str = str(expr)

        if expr.num_args() == 0:
            if expr_str == 'True':
                return "always true"
            elif expr_str == 'False':
                return "always false"
            else:
                return self.get_fact_description(expr_str)

        decl_name = expr.decl().name()

        if decl_name == 'not':
            inner = self.expr_to_plain_english(expr.arg(0), depth + 1)
            if inner.startswith('"') and inner.endswith('"'):
                return f"NOT {inner}"
            return f"NOT ({inner})"

        elif decl_name == 'and':
            parts = [self.expr_to_plain_english(expr.arg(i), depth + 1)
                     for i in range(expr.num_args())]
            if len(parts) == 2:
                return f"{parts[0]} AND {parts[1]}"
            return f"ALL OF: {', '.join(parts)}"

        elif decl_name == 'or':
            parts = [self.expr_to_plain_english(expr.arg(i), depth + 1)
                     for i in range(expr.num_args())]
            if len(parts) == 2:
                return f"{parts[0]} OR {parts[1]}"
            return f"ANY OF: {', '.join(parts)}"

        elif decl_name == '=>':
            left = self.expr_to_plain_english(expr.arg(0), depth + 1)
            right = self.expr_to_plain_english(expr.arg(1), depth + 1)
            return f"IF {left}, THEN {right}"

        elif decl_name == 'xor':
            left = self.expr_to_plain_english(expr.arg(0), depth + 1)
            right = self.expr_to_plain_english(expr.arg(1), depth + 1)
            return f"EITHER {left} OR {right} (but not both)"

        return f"[{expr_str}]"

    def explain_flaw_type(self, flaw_type: str, formula_id: str, 
                          involved_facts: list[str], expr) -> str:
        if flaw_type == 'self_contradictory':
            return (
                f"🚫 This statement is IMPOSSIBLE. It's like saying "
                f"\"it's raining and not raining at the same time.\" "
                f"No matter what, this can never be true."
            )
        elif flaw_type == 'contradicts_facts':
            facts_mentioned = [self.get_fact_description(f.split(':')[0].strip()) 
                              for f in involved_facts]
            facts_str = " and ".join(facts_mentioned) if facts_mentioned else "the stated facts"
            return (
                f"🔴 This reasoning DIRECTLY CONTRADICTS what the text also claims as fact. "
                f"The text says {facts_str} are true, but this statement can't be true "
                f"if those facts are true."
            )
        elif flaw_type == 'conflicts_with_reasoning':
            return (
                f"⚠️ This statement, on its own, could be true. But when combined with "
                f"the text's OTHER statements, they can't ALL be true at the same time."
            )
        return "❓ There's something off about this statement."


# =============================================================================
# Plain English Analyzer (with Fallacy Detection)
# =============================================================================

class PlainEnglishAnalyzer:
    """Analyzes reasoning and produces plain English reports with fallacy detection."""
    
    def __init__(self, parsed_result: dict):
        self.categorized = CategorizedPlog(parsed_result)
        self.converter = PlainEnglishConverter(self.categorized.all_atoms)
    
    def _add_regular_facts_as_true(self, solver: Solver):
        for atom_name, info in self.categorized.regular_atoms.items():
            solver.add(info['z3_var'] == True)
    
    def _extract_atom_names(self, expr) -> list[str]:
        atoms_found = []
        def traverse(e):
            if isinstance(e, BoolRef):
                if e.num_args() == 0 and str(e) not in ('True', 'False'):
                    atoms_found.append(str(e))
                else:
                    for i in range(e.num_args()):
                        traverse(e.arg(i))
        traverse(expr)
        return list(set(atoms_found))
    
    def detect_fallacies(self) -> list[DetectedFallacy]:
        detected = []
        for tag_id, tag_expr in self.categorized.fallacy_tag_formulas.items():
            fallacy_atom_name = self.categorized.get_fallacy_atom_for_tag(tag_id)
            if not fallacy_atom_name:
                continue
            
            test_solver = Solver()
            self._add_regular_facts_as_true(test_solver)
            for expr in self.categorized.regular_formulas.values():
                test_solver.add(expr)
            test_solver.add(tag_expr)
            
            fallacy_var = self.categorized.fallacy_atoms[fallacy_atom_name]['z3_var']
            test_solver.push()
            test_solver.add(fallacy_var == False)
            
            if test_solver.check() == unsat:
                fallacy_desc = self.categorized.get_fallacy_description(fallacy_atom_name)
                trigger_desc = self.converter.expr_to_plain_english(tag_expr)
                detected.append(DetectedFallacy(
                    tag_formula_id=tag_id,
                    fallacy_atom_name=fallacy_atom_name,
                    fallacy_description=fallacy_desc,
                    trigger_description=trigger_desc
                ))
            test_solver.pop()
        return detected
    
    def check_consistency(self) -> bool:
        solver = Solver()
        self._add_regular_facts_as_true(solver)
        for expr in self.categorized.regular_formulas.values():
            solver.add(expr)
        return solver.check() == sat
    
    def find_minimal_conflict(self) -> list[str]:
        solver = Solver()
        solver.set(':core.minimize', True)
        self._add_regular_facts_as_true(solver)
        for formula_id, expr in self.categorized.regular_formulas.items():
            tracker = Bool(f'__track_{formula_id}')
            solver.assert_and_track(expr, tracker)
        if solver.check() == unsat:
            core = solver.unsat_core()
            return [str(c).replace('__track_', '') for c in core]
        return []
    
    def classify_flaw(self, formula_id: str) -> str:
        expr = self.categorized.regular_formulas.get(formula_id)
        if expr is None:
            return 'unknown'
        solver = Solver()
        solver.add(expr)
        if solver.check() == unsat:
            return 'self_contradictory'
        solver = Solver()
        self._add_regular_facts_as_true(solver)
        solver.add(expr)
        if solver.check() == unsat:
            return 'contradicts_facts'
        return 'conflicts_with_reasoning'
    
    def find_pairwise_conflicts(self) -> list[tuple[str, str]]:
        conflicts = []
        formula_ids = list(self.categorized.regular_formulas.keys())
        for i, id1 in enumerate(formula_ids):
            for id2 in formula_ids[i+1:]:
                solver = Solver()
                self._add_regular_facts_as_true(solver)
                solver.add(self.categorized.regular_formulas[id1])
                solver.add(self.categorized.regular_formulas[id2])
                if solver.check() == unsat:
                    conflicts.append((id1, id2))
        return conflicts
    
    def generate_report(self) -> PlainEnglishReport:
        fallacies_detected = self.detect_fallacies()
        is_consistent = self.check_consistency()
        
        if fallacies_detected and not is_consistent:
            verdict = "❌ FLAWED REASONING: Fallacies detected AND logical contradictions found"
        elif fallacies_detected:
            verdict = "⚠️ FALLACIES DETECTED: The reasoning contains logical fallacies"
        elif not is_consistent:
            verdict = "❌ INCONSISTENT: The reasoning contains contradictions"
        else:
            verdict = "✅ THE REASONING CHECKS OUT"
        
        facts_lines = ["Here's what the text states as facts:\n"]
        for i, (atom_name, info) in enumerate(self.categorized.regular_atoms.items(), 1):
            desc = info['description'] or atom_name.replace('_', ' ')
            facts_lines.append(f"  {i}. \"{desc}\"")
        facts_summary = "\n".join(facts_lines)
        
        reasoning_lines = ["Here's the reasoning/connections the text makes:\n"]
        for i, (formula_id, expr) in enumerate(self.categorized.regular_formulas.items(), 1):
            plain = self.converter.expr_to_plain_english(expr)
            reasoning_lines.append(f"  {i}. {plain}")
        reasoning_summary = "\n".join(reasoning_lines)
        
        problems_found = []
        detailed_explanations = []
        
        if not is_consistent:
            conflict_formulas = self.find_minimal_conflict()
            pairwise = self.find_pairwise_conflicts()
            problems_found.append(f"Found {len(conflict_formulas)} problematic statement(s).")
            
            for formula_id in conflict_formulas:
                expr = self.categorized.regular_formulas[formula_id]
                flaw_type = self.classify_flaw(formula_id)
                involved = self._extract_atom_names(expr)
                involved_with_desc = [
                    f"{a}: \"{self.categorized.regular_atoms[a]['description']}\"" 
                    for a in involved 
                    if a in self.categorized.regular_atoms and self.categorized.regular_atoms[a]['description']
                ]
                plain_expr = self.converter.expr_to_plain_english(expr)
                explanation = self.converter.explain_flaw_type(flaw_type, formula_id, involved_with_desc, expr)
                detailed_explanations.append(f"PROBLEM: \"{plain_expr}\"\n\n{explanation}")
            
            if pairwise:
                conflict_explanation = "\n📍 These statements clash:\n"
                for id1, id2 in pairwise:
                    expr1_plain = self.converter.expr_to_plain_english(self.categorized.regular_formulas[id1])
                    expr2_plain = self.converter.expr_to_plain_english(self.categorized.regular_formulas[id2])
                    conflict_explanation += f"\n  • \"{expr1_plain}\"\n    CONFLICTS WITH\n    \"{expr2_plain}\"\n"
                detailed_explanations.append(conflict_explanation)
        
        num_facts = len(self.categorized.regular_atoms)
        num_formulas = len(self.categorized.regular_formulas)
        num_fallacies = len(fallacies_detected)
        
        if fallacies_detected and not is_consistent:
            bottom_line = f"👎 BOTTOM LINE: This text has serious problems with {num_fallacies} fallacy(ies) AND contradictions."
        elif fallacies_detected:
            bottom_line = f"⚠️ BOTTOM LINE: This text contains {num_fallacies} logical fallacy(ies). The author uses flawed reasoning."
        elif not is_consistent:
            bottom_line = f"👎 BOTTOM LINE: The text's {num_facts} facts and {num_formulas} logical connections contradict each other."
        else:
            bottom_line = f"👍 BOTTOM LINE: Based on {num_facts} facts and {num_formulas} connections, the logic holds together."
        
        return PlainEnglishReport(
            is_consistent=is_consistent,
            verdict=verdict,
            facts_summary=facts_summary,
            reasoning_summary=reasoning_summary,
            fallacies_detected=fallacies_detected,
            problems_found=problems_found,
            detailed_explanations=detailed_explanations,
            bottom_line=bottom_line
        )


# =============================================================================
# Report Printer
# =============================================================================

def print_plain_english_report(report: PlainEnglishReport):
    width = 70
    print("\n" + "=" * width)
    print("  🔍 TEXT LOGIC CHECKER - ANALYSIS REPORT")
    print("=" * width)
    
    print("\n" + "-" * width)
    print(f"  {report.verdict}")
    print("-" * width)
    
    if report.fallacies_detected:
        print("\n" + "=" * width)
        print("  🚩 LOGICAL FALLACIES DETECTED")
        print("=" * width)
        print("\n  The text contains the following fallacious reasoning patterns:\n")
        for i, fallacy in enumerate(report.fallacies_detected, 1):
            desc = fallacy.fallacy_description
            if desc.startswith("FALLACY:"):
                desc = desc[8:].strip()
            print(f"  {i}. 🚩 {desc}")
            print()
        print("-" * width)
    
    print("\n📋 WHAT THE TEXT STATES AS FACTS:")
    print("-" * width)
    print(report.facts_summary)
    
    print("\n🔗 THE LOGIC PRESENT IN THE TEXT:")
    print("-" * width)
    print(report.reasoning_summary)
    
    if not report.is_consistent:
        print("\n" + "=" * width)
        print("  🚨 LOGICAL CONTRADICTIONS DETECTED")
        print("=" * width)
        for problem in report.problems_found:
            print(f"\n{problem}")
        for explanation in report.detailed_explanations:
            print(f"\n{'─' * width}")
            print(explanation)
    
    print("\n" + "=" * width)
    print("  📝 SUMMARY")
    print("=" * width)
    print(f"\n{report.bottom_line}")
    print("\n" + "=" * width + "\n")


# =============================================================================
# Menu System
# =============================================================================

class MenuItem:
    def __init__(self, label: str, action: Callable = None, data: any = None):
        self.label = label
        self.action = action
        self.data = data


class MenuSection:
    def __init__(self, title: str = None, items: list[MenuItem] = None):
        self.title = title
        self.items = items or []

    def add_item(self, item: MenuItem):
        self.items.append(item)


class InteractiveMenu:
    def __init__(self, title: str = "Menu"):
        self.title = title
        self.sections: list[MenuSection] = []

    def add_section(self, section: MenuSection):
        self.sections.append(section)

    def _build_menu_entries(self) -> tuple[list[str], list[MenuItem]]:
        entries = []
        items = []
        for section in self.sections:
            if section.title:
                entries.append(f"──── {section.title} ────")
                items.append(None)
            for item in section.items:
                entries.append(f"  {item.label}")
                items.append(item)
        return entries, items

    def show(self) -> Optional[MenuItem]:
        entries, items = self._build_menu_entries()
        if not HAS_TERMINAL_MENU:
            return self._fallback_menu(entries, items)
        menu = TerminalMenu(entries, title=f"\n{self.title}\n", skip_empty_entries=True, accept_keys=("enter",))
        selected = menu.show()
        if selected is None or items[selected] is None:
            return None
        return items[selected]

    def _fallback_menu(self, entries: list[str], items: list[MenuItem]) -> Optional[MenuItem]:
        print(f"\n{self.title}")
        print("=" * 50)
        selectable = []
        for entry, item in zip(entries, items):
            if item is None:
                print(f"\n{entry}")
            else:
                idx = len(selectable)
                print(f"  [{idx}] {item.label}")
                selectable.append(item)
        print()
        try:
            choice = input("Enter choice (or 'q' to go back): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice)
            if 0 <= idx < len(selectable):
                return selectable[idx]
        except (ValueError, EOFError):
            pass
        return None


# =============================================================================
# Simulator Core (with Fallacy Detection)
# =============================================================================

class WhatIfSimulator:
    """Interactive what-if simulator with fallacy detection."""

    def __init__(self, parsed_result: dict, filename: str = ""):
        self.categorized = CategorizedPlog(parsed_result)
        self.filename = filename
        self.converter = PlainEnglishConverter(self.categorized.all_atoms)
        self.state = SimulationState()
        self.reset_state()
        self.needs_evaluation = True

    def reset_state(self):
        self.state.fact_values = {name: True for name in self.categorized.regular_atoms}
        self.state.formula_states = {fid: FormulaState.ON for fid in self.categorized.regular_formulas}
        self.state.fallacy_tag_states = {fid: FormulaState.ON for fid in self.categorized.fallacy_tag_formulas}
        self.state.last_evaluation = None
        self.needs_evaluation = True

    def toggle_fact(self, atom_name: str):
        self.state.fact_values[atom_name] = not self.state.fact_values[atom_name]
        self.needs_evaluation = True

    def toggle_formula(self, formula_id: str):
        current = self.state.formula_states[formula_id]
        self.state.formula_states[formula_id] = FormulaState.OFF if current == FormulaState.ON else FormulaState.ON
        self.needs_evaluation = True

    def toggle_fallacy_tag(self, formula_id: str):
        current = self.state.fallacy_tag_states[formula_id]
        self.state.fallacy_tag_states[formula_id] = FormulaState.OFF if current == FormulaState.ON else FormulaState.ON
        self.needs_evaluation = True

    def detect_active_fallacies(self) -> list[DetectedFallacy]:
        detected = []
        for tag_id, tag_expr in self.categorized.fallacy_tag_formulas.items():
            if self.state.fallacy_tag_states.get(tag_id) == FormulaState.OFF:
                continue
            fallacy_atom_name = self.categorized.get_fallacy_atom_for_tag(tag_id)
            if not fallacy_atom_name:
                continue
            
            solver = Solver()
            for name, info in self.categorized.regular_atoms.items():
                solver.add(info['z3_var'] == self.state.fact_values.get(name, True))
            for fid, expr in self.categorized.regular_formulas.items():
                if self.state.formula_states.get(fid) == FormulaState.ON:
                    solver.add(expr)
            solver.add(tag_expr)
            
            fallacy_var = self.categorized.fallacy_atoms[fallacy_atom_name]['z3_var']
            solver.push()
            solver.add(fallacy_var == False)
            if solver.check() == unsat:
                fallacy_desc = self.categorized.get_fallacy_description(fallacy_atom_name)
                trigger_desc = self.converter.expr_to_plain_english(tag_expr)
                detected.append(DetectedFallacy(tag_id, fallacy_atom_name, fallacy_desc, trigger_desc))
            solver.pop()
        return detected

    def evaluate(self) -> EvaluationResult:
        solver = Solver()
        solver.set(':core.minimize', True)
        
        for name, info in self.categorized.regular_atoms.items():
            solver.add(info['z3_var'] == self.state.fact_values.get(name, True))
        
        trackers = {}
        for fid, expr in self.categorized.regular_formulas.items():
            if self.state.formula_states.get(fid) == FormulaState.ON:
                tracker = Bool(f'__track_{fid}')
                trackers[fid] = tracker
                solver.assert_and_track(expr, tracker)
        
        result = solver.check()
        active_fallacies = self.detect_active_fallacies()
        
        if result == sat:
            evaluation = EvaluationResult(True, [], "All facts and active formulas are consistent.", active_fallacies)
        else:
            core = solver.unsat_core()
            conflicting = [str(c).replace('__track_', '') for c in core]
            evaluation = EvaluationResult(False, conflicting, f"Found {len(conflicting)} formula(s) in conflict.", active_fallacies)
        
        self.state.last_evaluation = evaluation
        self.needs_evaluation = False
        return evaluation

    def get_changes_summary(self) -> list[str]:
        changes = []
        for name, value in self.state.fact_values.items():
            if not value:
                desc = self.categorized.regular_atoms[name]['description'] or name.replace('_', ' ')
                changes.append(f"Fact \"{desc}\" set to FALSE")
        for fid, state in self.state.formula_states.items():
            if state == FormulaState.OFF:
                changes.append(f"Formula \"{fid}\" is OFF")
        for fid, state in self.state.fallacy_tag_states.items():
            if state == FormulaState.OFF:
                fallacy_atom = self.categorized.get_fallacy_atom_for_tag(fid)
                if fallacy_atom:
                    desc = self.categorized.get_fallacy_description(fallacy_atom)
                    if desc.startswith("FALLACY:"):
                        desc = desc[8:].strip().split(" - ")[0]
                    changes.append(f"Fallacy OFF: {desc}")
        return changes

    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _display_header(self):
        print("═" * 70)
        print("  🔬 WHAT-IF SIMULATOR - Interactive Mode")
        print("═" * 70)
        if self.filename:
            print(f"\n📄 Loaded: {self.filename}")

    def _display_facts_table(self):
        if not self.categorized.regular_atoms:
            return
        print("\n" + "═" * 70)
        print(f"  📋 FACTS ({len(self.categorized.regular_atoms)} items)")
        print("═" * 70)
        print(f"\n  {'#':<4} {'Status':<10} {'Fact'}")
        print("  " + "─" * 64)
        for i, (name, info) in enumerate(self.categorized.regular_atoms.items(), 1):
            value = self.state.fact_values.get(name, True)
            status = "✅ TRUE" if value else "❌ FALSE"
            desc = info['description'] or name.replace('_', ' ')
            if len(desc) > 50:
                desc = desc[:47] + "..."
            marker = "" if value else "  ← CHANGED"
            print(f"  {i:<4} {status:<10} \"{desc}\"{marker}")

    def _display_formulas_table(self):
        if not self.categorized.regular_formulas:
            return
        print("\n" + "═" * 70)
        print(f"  🔗 FORMULAS ({len(self.categorized.regular_formulas)} items)")
        print("═" * 70)
        print(f"\n  {'#':<4} {'Status':<10} {'Rule'}")
        print("  " + "─" * 64)
        for i, (fid, expr) in enumerate(self.categorized.regular_formulas.items()):
            state = self.state.formula_states.get(fid, FormulaState.ON)
            status = "📝 ON" if state == FormulaState.ON else "⏸️  OFF"
            plain = self.converter.expr_to_plain_english(expr)
            if len(plain) > 45:
                plain = plain[:42] + "..."
            marker = "" if state == FormulaState.ON else "  ← IGNORED"
            letter = chr(ord('A') + i) if i < 26 else f"A{i-25}"
            print(f"  {letter:<4} {status:<10} {plain}{marker}")

    def _display_fallacy_table(self):
        if not self.categorized.fallacy_tag_formulas:
            return
        print("\n" + "═" * 70)
        print(f"  🚩 FALLACY DETECTION ({len(self.categorized.fallacy_tag_formulas)} items)")
        print("═" * 70)
        print(f"\n  {'#':<4} {'Status':<12} {'Fallacy Type'}")
        print("  " + "─" * 64)
        for i, (fid, expr) in enumerate(self.categorized.fallacy_tag_formulas.items(), 1):
            state = self.state.fallacy_tag_states.get(fid, FormulaState.ON)
            fallacy_atom = self.categorized.get_fallacy_atom_for_tag(fid)
            if fallacy_atom:
                desc = self.categorized.get_fallacy_description(fallacy_atom)
                if desc.startswith("FALLACY:"):
                    desc = desc[8:].strip()
            else:
                desc = fid
            if len(desc) > 45:
                desc = desc[:42] + "..."
            if state == FormulaState.ON:
                status = "🚩 ON"
                marker = ""
            else:
                status = "⚪ OFF"
                marker = "  ← NOT A FALLACY"
            print(f"  T{i:<3} {status:<12} {desc}{marker}")

    def _display_status(self):
        print("\n" + "═" * 70)
        if self.needs_evaluation:
            print("  ⚡ STATUS: Not yet evaluated (select 'Evaluate' from menu)")
        elif self.state.last_evaluation:
            eval_result = self.state.last_evaluation
            if eval_result.active_fallacies:
                print(f"  🚩 FALLACIES: {len(eval_result.active_fallacies)} detected")
            if eval_result.is_consistent:
                print("  ⚡ CONSISTENCY: ✅ CONSISTENT")
            else:
                print("  ⚡ CONSISTENCY: ❌ INCONSISTENT")
                if eval_result.conflicting_formulas:
                    formulas_str = ", ".join(eval_result.conflicting_formulas[:3])
                    if len(eval_result.conflicting_formulas) > 3:
                        formulas_str += "..."
                    print(f"     Conflicting: {formulas_str}")
        print("═" * 70)

    def _display_current_state(self):
        self._clear_screen()
        self._display_header()
        self._display_facts_table()
        self._display_formulas_table()
        self._display_fallacy_table()
        self._display_status()

    def _build_main_menu(self) -> InteractiveMenu:
        menu = InteractiveMenu("SELECT AN ACTION")

        if self.categorized.regular_atoms:
            facts_section = MenuSection("TOGGLE FACTS (TRUE ↔ FALSE)")
            for i, (name, info) in enumerate(self.categorized.regular_atoms.items(), 1):
                value = self.state.fact_values.get(name, True)
                status = "TRUE" if value else "FALSE"
                desc = info['description'] or name.replace('_', ' ')
                short_desc = desc[:35] + "..." if len(desc) > 35 else desc
                label = f"[{i}] {status:<6} │ {short_desc}"
                facts_section.add_item(MenuItem(label, action=self.toggle_fact, data=name))
            menu.add_section(facts_section)

        if self.categorized.regular_formulas:
            formulas_section = MenuSection("TOGGLE FORMULAS (ON ↔ OFF)")
            for i, (fid, expr) in enumerate(self.categorized.regular_formulas.items()):
                state = self.state.formula_states.get(fid, FormulaState.ON)
                status = "ON" if state == FormulaState.ON else "OFF"
                letter = chr(ord('A') + i) if i < 26 else f"A{i-25}"
                plain = self.converter.expr_to_plain_english(expr)
                short_plain = plain[:30] + "..." if len(plain) > 30 else plain
                label = f"[{letter}] {status:<6} │ {short_plain}"
                formulas_section.add_item(MenuItem(label, action=self.toggle_formula, data=fid))
            menu.add_section(formulas_section)

        if self.categorized.fallacy_tag_formulas:
            fallacy_section = MenuSection("TOGGLE FALLACY DETECTION (ON ↔ OFF)")
            for i, (fid, expr) in enumerate(self.categorized.fallacy_tag_formulas.items(), 1):
                state = self.state.fallacy_tag_states.get(fid, FormulaState.ON)
                status = "ON" if state == FormulaState.ON else "OFF"
                fallacy_atom = self.categorized.get_fallacy_atom_for_tag(fid)
                if fallacy_atom:
                    desc = self.categorized.get_fallacy_description(fallacy_atom)
                    if desc.startswith("FALLACY:"):
                        parts = desc[8:].strip().split(" - ")
                        desc = parts[0] if parts else desc
                else:
                    desc = fid
                short_desc = desc[:30] + "..." if len(desc) > 30 else desc
                label = f"[T{i}] {status:<6} │ {short_desc}"
                fallacy_section.add_item(MenuItem(label, action=self.toggle_fallacy_tag, data=fid))
            menu.add_section(fallacy_section)

        actions_section = MenuSection("ACTIONS")
        actions_section.add_item(MenuItem("[E] 📊 Evaluate current scenario", action="evaluate"))
        actions_section.add_item(MenuItem("[D] 📄 Show detailed analysis", action="detailed"))
        actions_section.add_item(MenuItem("[R] 🔄 Reset to original state", action="reset"))
        actions_section.add_item(MenuItem("[Q] 🚪 Quit simulator", action="quit"))
        menu.add_section(actions_section)
        return menu

    def _show_evaluation_result(self):
        if self.state.last_evaluation is None:
            result = self.evaluate()
        else:
            result = self.state.last_evaluation
        self._clear_screen()
        print("═" * 70)
        print("  📊 EVALUATION RESULT")
        print("═" * 70)
        changes = self.get_changes_summary()
        if changes:
            print("\n  Testing scenario with these changes:")
            for change in changes:
                print(f"    • {change}")
        else:
            print("\n  Testing original scenario (no changes made)")
        print("\n" + "─" * 70)
        if result.active_fallacies:
            print("\n  🚩 FALLACIES DETECTED:")
            for fallacy in result.active_fallacies:
                desc = fallacy.fallacy_description
                if desc.startswith("FALLACY:"):
                    desc = desc[8:].strip()
                print(f"    • {desc}")
            print()
        if result.is_consistent:
            print("\n  ✅ LOGICALLY CONSISTENT")
            print("\n  All the facts and active reasoning rules can be true simultaneously.")
        else:
            print("\n  ❌ LOGICALLY INCONSISTENT")
            print(f"\n  {result.details}")
            if result.conflicting_formulas:
                print("\n  Formulas involved in the contradiction:")
                for fid in result.conflicting_formulas:
                    if fid in self.categorized.regular_formulas:
                        expr = self.categorized.regular_formulas[fid]
                        plain = self.converter.expr_to_plain_english(expr)
                        print(f"    • {fid}: {plain}")
        print("\n" + "═" * 70)
        input("\nPress ENTER to continue...")

    def _show_detailed_analysis(self):
        if self.needs_evaluation:
            self.evaluate()
        self._clear_screen()
        print("═" * 70)
        print("  📄 DETAILED ANALYSIS")
        print("═" * 70)
        print("\n┌─ CURRENT CONFIGURATION ─────────────────────────────────────────────┐")
        print("│")
        print("│  FACTS:")
        for name, info in self.categorized.regular_atoms.items():
            value = self.state.fact_values.get(name, True)
            status = "TRUE" if value else "FALSE"
            desc = info['description'] or name.replace('_', ' ')
            if len(desc) > 50:
                desc = desc[:47] + "..."
            print(f"│    [{status:<5}] \"{desc}\"")
        print("│")
        print("│  FORMULAS:")
        for fid, expr in self.categorized.regular_formulas.items():
            state = self.state.formula_states.get(fid, FormulaState.ON)
            status = "ON" if state == FormulaState.ON else "OFF"
            plain = self.converter.expr_to_plain_english(expr)
            if len(plain) > 50:
                plain = plain[:47] + "..."
            print(f"│    [{status:<3}] {fid}: {plain}")
        if self.categorized.fallacy_tag_formulas:
            print("│")
            print("│  FALLACY DETECTION:")
            for fid in self.categorized.fallacy_tag_formulas:
                state = self.state.fallacy_tag_states.get(fid, FormulaState.ON)
                status = "ON" if state == FormulaState.ON else "OFF"
                fallacy_atom = self.categorized.get_fallacy_atom_for_tag(fid)
                if fallacy_atom:
                    desc = self.categorized.get_fallacy_description(fallacy_atom)
                    if desc.startswith("FALLACY:"):
                        desc = desc[8:].strip().split(" - ")[0]
                else:
                    desc = fid
                print(f"│    [{status:<3}] {desc}")
        print("│")
        print("└──────────────────────────────────────────────────────────────────────┘")
        result = self.state.last_evaluation
        print("\n┌─ RESULT ─────────────────────────────────────────────────────────────┐")
        print("│")
        if result.active_fallacies:
            print(f"│  🚩 {len(result.active_fallacies)} FALLACY(IES) DETECTED")
            for fallacy in result.active_fallacies:
                desc = fallacy.fallacy_description
                if desc.startswith("FALLACY:"):
                    desc = desc[8:].strip()
                if len(desc) > 55:
                    desc = desc[:52] + "..."
                print(f"│    • {desc}")
            print("│")
        if result.is_consistent:
            print("│  ✅ LOGICALLY CONSISTENT")
        else:
            print("│  ❌ LOGICALLY INCONSISTENT")
            if result.conflicting_formulas:
                print("│  Formulas involved:")
                for fid in result.conflicting_formulas:
                    print(f"│    • {fid}")
        print("│")
        print("└──────────────────────────────────────────────────────────────────────┘")
        input("\nPress ENTER to continue...")

    def run(self):
        running = True
        while running:
            self._display_current_state()
            menu = self._build_main_menu()
            selected = menu.show()
            if selected is None:
                continue
            if selected.action == "evaluate":
                self.evaluate()
                self._show_evaluation_result()
            elif selected.action == "detailed":
                self._show_detailed_analysis()
            elif selected.action == "reset":
                self.reset_state()
            elif selected.action == "quit":
                running = False
            elif callable(selected.action):
                selected.action(selected.data)
        print("\nGoodbye! 👋\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # if no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    ap = argparse.ArgumentParser(description="plog-ai contradiction checker",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
 python plog.py <plog_file>                  # Run analysis
 python plog.py --grammar ./some-grammar.lark <plog_file>  # Run analysis with a specific grammar
 python plog.py --interactive <plog_file>    # Interactive mode"""
    )
    ap.add_argument("file", help="Path to .plog file")
    ap.add_argument("--grammar", "-g", default=None, help="Path to a lark grammar (default: plog.lark on same dir as script)")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Run in interactive mode with menu for toggling atoms and formulas.")
    args = ap.parse_args()

    interactive = args.interactive
    plog_file = args.file
    grammar_path = args.grammar
    
    script_dir = Path(__file__).parent

    # Locate grammar file if not provided
    if grammar_path is None:
        grammar_file = Path(script_dir) / "plog.lark"
    else:
        grammar_file = Path(grammar_path)
        
    if not grammar_file.exists():
        print(f"❌ Error: Could not find plog.lark grammar file at: {grammar_file}")
        sys.exit(2)

    try:
        parser = PlogParser(grammar_file.absolute().as_posix())
        parsed = parser.parse_file(plog_file)
        categorized = CategorizedPlog(parsed)
        
        if not categorized.regular_atoms and not categorized.fallacy_atoms:
            print("❌ Error: No atoms found in the plog file")
            sys.exit(2)

        if interactive:
            simulator = WhatIfSimulator(parsed, filename=plog_file)
            simulator.run()
        else:
            analyzer = PlainEnglishAnalyzer(parsed)
            report = analyzer.generate_report()
            print_plain_english_report(report)
            has_issues = not report.is_consistent or len(report.fallacies_detected) > 0
            sys.exit(1 if has_issues else 0)

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file - {e}\n")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()