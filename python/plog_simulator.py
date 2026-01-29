"""
Plog What-If Simulator - Interactive Mode

Allows users to interactively toggle facts (TRUE/FALSE) and formulas (ON/OFF)
to explore "what-if" scenarios and understand logical contradictions.

Usage:
    python plog_simulator.py <plog_file>
    python plog_simulator.py --interactive <plog_file>

Dependencies:
    pip install lark z3-solver simple-term-menu
"""

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
# Data Classes
# =============================================================================

class FormulaState(Enum):
    ON = "on"      # Use formula as-is
    OFF = "off"    # Ignore formula (don't add to solver)


@dataclass
class SimulationState:
    """Holds the current state of the simulation."""
    fact_values: dict = field(default_factory=dict)      # atom_name -> bool
    formula_states: dict = field(default_factory=dict)   # formula_id -> FormulaState
    last_evaluation: Optional[dict] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a scenario."""
    is_consistent: bool
    conflicting_formulas: list
    details: str


@dataclass 
class PlainEnglishReport:
    """A report designed for non-experts."""
    is_consistent: bool
    verdict: str
    facts_summary: str
    reasoning_summary: str
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
        return {
            'atoms': self.atoms,
            'formulas': self.formulas,
        }

    def statement(self, items):
        return items[0] if items else None

    @v_args(inline=True)
    def atom_decl(self, name, description):
        atom_name = str(name)
        self.atoms[atom_name] = {
            'z3_var': Bool(atom_name),
            'description': description
        }
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
            self.atoms[atom_name] = {
                'z3_var': Bool(atom_name),
                'description': None
            }
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
        """Generate a plain English explanation for a type of flaw."""
        
        if flaw_type == 'self_contradictory':
            return (
                f"🚫 This statement is IMPOSSIBLE. It's like saying "
                f"\"it's raining and not raining at the same time.\" "
                f"No matter what, this can never be true. "
                f"The text states something that contradicts itself within the same breath."
            )
        
        elif flaw_type == 'contradicts_facts':
            facts_mentioned = [self.get_fact_description(f.split(':')[0].strip()) 
                              for f in involved_facts]
            facts_str = " and ".join(facts_mentioned) if facts_mentioned else "the stated facts"
            
            return (
                f"🔴 This reasoning DIRECTLY CONTRADICTS what the text also claims as fact. "
                f"The text says {facts_str} are true, but this statement can't be true "
                f"if those facts are true. It's essentially saying \"X is true\" "
                f"and then making an argument that requires \"X is false.\""
            )
        
        elif flaw_type == 'conflicts_with_reasoning':
            return (
                f"⚠️ This statement, on its own, could be true. But when you put it together "
                f"with the text's OTHER statements, they can't ALL be true at the same time. "
                f"The text has made multiple claims that are individually reasonable "
                f"but collectively impossible."
            )
        
        return "❓ There's something off about this statement."

# =============================================================================
# Plain English Analyzer
# =============================================================================

class PlainEnglishAnalyzer:
    """Analyzes reasoning and produces plain English reports."""
    
    def __init__(self, parsed_result: dict):
        self.atoms = parsed_result['atoms']
        self.formulas = parsed_result['formulas']
        self.converter = PlainEnglishConverter(self.atoms)
    
    def _add_facts_as_true(self, solver: Solver):
        for atom_name, info in self.atoms.items():
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
    
    def check_consistency(self) -> bool:
        solver = Solver()
        self._add_facts_as_true(solver)
        for expr in self.formulas.values():
            solver.add(expr)
        return solver.check() == sat
    
    def find_minimal_conflict(self) -> list[str]:
        solver = Solver()
        solver.set(':core.minimize', True)
        self._add_facts_as_true(solver)
        
        for formula_id, expr in self.formulas.items():
            tracker = Bool(f'__track_{formula_id}')
            solver.assert_and_track(expr, tracker)
        
        if solver.check() == unsat:
            core = solver.unsat_core()
            return [str(c).replace('__track_', '') for c in core]
        return []
    
    def classify_flaw(self, formula_id: str) -> str:
        expr = self.formulas[formula_id]
        
        # Self-contradictory
        solver = Solver()
        solver.add(expr)
        if solver.check() == unsat:
            return 'self_contradictory'
        
        # Contradicts facts
        solver = Solver()
        self._add_facts_as_true(solver)
        solver.add(expr)
        if solver.check() == unsat:
            return 'contradicts_facts'
        
        return 'conflicts_with_reasoning'
    
    def find_pairwise_conflicts(self) -> list[tuple[str, str]]:
        conflicts = []
        formula_ids = list(self.formulas.keys())
        
        for i, id1 in enumerate(formula_ids):
            for id2 in formula_ids[i+1:]:
                solver = Solver()
                self._add_facts_as_true(solver)
                solver.add(self.formulas[id1])
                solver.add(self.formulas[id2])
                
                if solver.check() == unsat:
                    conflicts.append((id1, id2))
        
        return conflicts
    
    def generate_report(self) -> PlainEnglishReport:
        """Generate a complete plain English report."""
        
        is_consistent = self.check_consistency()
        
        # === VERDICT ===
        if is_consistent:
            verdict = "✅ THE REASONING CHECKS OUT"
        else:
            verdict = "❌ THERE ARE PROBLEMS WITH THIS REASONING"
        
        # === FACTS SUMMARY ===
        facts_lines = ["Here's what the text states as facts:\n"]
        for i, (atom_name, info) in enumerate(self.atoms.items(), 1):
            desc = info['description'] or atom_name.replace('_', ' ')
            facts_lines.append(f"  {i}. \"{desc}\"")
        facts_summary = "\n".join(facts_lines)
        
        # === REASONING SUMMARY ===
        reasoning_lines = ["Here's the reasoning/connections the text makes:\n"]
        for i, (formula_id, expr) in enumerate(self.formulas.items(), 1):
            plain = self.converter.expr_to_plain_english(expr)
            reasoning_lines.append(f"  {i}. {plain}")
        reasoning_summary = "\n".join(reasoning_lines)
        
        # === PROBLEMS FOUND ===
        problems_found = []
        detailed_explanations = []
        
        if not is_consistent:
            conflict_formulas = self.find_minimal_conflict()
            pairwise = self.find_pairwise_conflicts()
            
            # Main problem summary
            problems_found.append(
                f"Found {len(conflict_formulas)} problematic statement(s) "
                f"that cause the contradiction."
            )
            
            # Detailed explanations for each problematic formula
            for formula_id in conflict_formulas:
                expr = self.formulas[formula_id]
                flaw_type = self.classify_flaw(formula_id)
                involved = self._extract_atom_names(expr)
                involved_with_desc = [
                    f"{a}: \"{self.atoms[a]['description']}\"" 
                    for a in involved if a in self.atoms and self.atoms[a]['description']
                ]
                
                plain_expr = self.converter.expr_to_plain_english(expr)
                explanation = self.converter.explain_flaw_type(
                    flaw_type, formula_id, involved_with_desc, expr
                )
                
                detailed_explanations.append(
                    f"PROBLEM: \"{plain_expr}\"\n\n{explanation}"
                )
            
            # Add pairwise conflict explanations
            if pairwise:
                conflict_explanation = "\n\n📍 SPECIFICALLY, these statements clash with each other:\n"
                for id1, id2 in pairwise:
                    expr1_plain = self.converter.expr_to_plain_english(self.formulas[id1])
                    expr2_plain = self.converter.expr_to_plain_english(self.formulas[id2])
                    conflict_explanation += (
                        f"\n  • \"{expr1_plain}\"\n"
                        f"    CONFLICTS WITH\n"
                        f"    \"{expr2_plain}\"\n"
                    )
                detailed_explanations.append(conflict_explanation)
        
        # === BOTTOM LINE ===
        if is_consistent:
            bottom_line = (
                f"👍 BOTTOM LINE: Based on the {len(self.atoms)} facts stated and "
                f"{len(self.formulas)} logical connection(s) made, "
                f"everything the text states is internally consistent. "
                f"Its logic holds together - there are no logical contradictions. "
                f"This doesn't mean everything it states is TRUE, just that "
                f"its statements don't contradict each other."
            )
        else:
            bottom_line = (
                f"👎 BOTTOM LINE: The text's statements don't add up. "
                f"It states claims that logically cannot all be true at the same time. "
                f"Either some of the stated \"facts\" are wrong, or its reasoning is flawed. "
                f"See the details above for exactly where the problems are."
            )
        
        return PlainEnglishReport(
            is_consistent=is_consistent,
            verdict=verdict,
            facts_summary=facts_summary,
            reasoning_summary=reasoning_summary,
            problems_found=problems_found,
            detailed_explanations=detailed_explanations,
            bottom_line=bottom_line
        )


# =============================================================================
# Report Printer
# =============================================================================

def print_plain_english_report(report: PlainEnglishReport):
    """Print the report in a user-friendly format."""
    
    width = 70
    
    print("\n" + "=" * width)
    print("  🔍 TEXT LOGIC CHECKER - ANALYSIS REPORT")
    print("=" * width)
    
    # Verdict banner
    print("\n" + "-" * width)
    if report.is_consistent:
        print(f"  {report.verdict}")
    else:
        print(f"  {report.verdict}")
    print("-" * width)
    
    # What was said (facts)
    print("\n📋 WHAT THE TEXT STATES AS FACTS:")
    print("-" * width)
    print(report.facts_summary)
    
    # Reasoning/connections
    print("\n🔗 THE LOGIC PRESENT IN THE TEXT:")
    print("-" * width)
    print(report.reasoning_summary)
    
    # Problems (if any)
    if not report.is_consistent:
        print("\n" + "=" * width)
        print("  🚨 PROBLEMS DETECTED")
        print("=" * width)
        
        for problem in report.problems_found:
            print(f"\n{problem}")
        
        for i, explanation in enumerate(report.detailed_explanations, 1):
            print(f"\n{'─' * width}")
            print(explanation)
    
    # Bottom line
    print("\n" + "=" * width)
    print("  📝 SUMMARY")
    print("=" * width)
    print(f"\n{report.bottom_line}")
    print("\n" + "=" * width + "\n")


# =============================================================================
# Menu System
# =============================================================================

class MenuItem:
    """Represents a single menu item."""

    def __init__(self, label: str, action: Callable = None, data: any = None):
        self.label = label
        self.action = action
        self.data = data


class MenuSection:
    """A section of menu items with an optional title."""

    def __init__(self, title: str = None, items: list[MenuItem] = None):
        self.title = title
        self.items = items or []

    def add_item(self, item: MenuItem):
        self.items.append(item)


class InteractiveMenu:
    """
    Manages interactive terminal menus using simple-term-menu.
    Provides a clean abstraction for building and displaying menus.
    """

    def __init__(self, title: str = "Menu"):
        self.title = title
        self.sections: list[MenuSection] = []

    def add_section(self, section: MenuSection):
        self.sections.append(section)

    def _build_menu_entries(self) -> tuple[list[str], list[MenuItem]]:
        """Build flat list of menu entries and corresponding items."""
        entries = []
        items = []

        for section in self.sections:
            if section.title:
                # Add section header (non-selectable via prefix)
                entries.append(f"──── {section.title} ────")
                items.append(None)  # Placeholder for non-selectable

            for item in section.items:
                entries.append(f"  {item.label}")
                items.append(item)

        return entries, items

    def show(self) -> Optional[MenuItem]:
        """Display the menu and return the selected item."""
        entries, items = self._build_menu_entries()

        if not HAS_TERMINAL_MENU:
            return self._fallback_menu(entries, items)

        # Find non-selectable indices (section headers)
        skip_indices = [i for i, item in enumerate(items) if item is None]

        menu = TerminalMenu(
            entries,
            title=f"\n{self.title}\n",
            skip_empty_entries=True,
            accept_keys=("enter",),
            # Make section headers non-selectable
            preselected_entries=None,
        )

        # Custom show that skips headers
        selected = menu.show()

        if selected is None:
            return None

        # Skip section headers
        if items[selected] is None:
            return None

        return items[selected]

    def _fallback_menu(self, entries: list[str], items: list[MenuItem]) -> Optional[MenuItem]:
        """Fallback text-based menu when simple-term-menu is not available."""
        print(f"\n{self.title}")
        print("=" * 50)

        selectable = []
        for i, (entry, item) in enumerate(zip(entries, items)):
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
# Simulator Core
# =============================================================================

class WhatIfSimulator:
    """
    Interactive what-if simulator for plog files.

    Allows users to toggle facts (TRUE/FALSE) and formulas (ON/OFF)
    to explore different scenarios.
    """

    def __init__(self, parsed_result: dict, filename: str = ""):
        self.atoms = parsed_result['atoms']
        self.formulas = parsed_result['formulas']
        self.filename = filename
        self.converter = PlainEnglishConverter(self.atoms)

        # Initialize state
        self.state = SimulationState()
        self.reset_state()

        # Track if we need to re-evaluate
        self.needs_evaluation = True

    def reset_state(self):
        """Reset all facts to TRUE and all formulas to ON."""
        self.state.fact_values = {name: True for name in self.atoms}
        self.state.formula_states = {fid: FormulaState.ON for fid in self.formulas}
        self.state.last_evaluation = None
        self.needs_evaluation = True

    def toggle_fact(self, atom_name: str):
        """Toggle a fact between TRUE and FALSE."""
        self.state.fact_values[atom_name] = not self.state.fact_values[atom_name]
        self.needs_evaluation = True

    def toggle_formula(self, formula_id: str):
        """Toggle a formula between ON and OFF."""
        current = self.state.formula_states[formula_id]
        self.state.formula_states[formula_id] = (
            FormulaState.OFF if current == FormulaState.ON else FormulaState.ON
        )
        self.needs_evaluation = True

    def evaluate(self) -> EvaluationResult:
        """Evaluate the current scenario."""
        solver = Solver()
        solver.set(':core.minimize', True)

        # Add facts with their current truth values
        for name, info in self.atoms.items():
            solver.add(info['z3_var'] == self.state.fact_values[name])

        # Add formulas based on their state (only ON formulas)
        trackers = {}
        for fid, expr in self.formulas.items():
            if self.state.formula_states[fid] == FormulaState.ON:
                tracker = Bool(f'__track_{fid}')
                trackers[fid] = tracker
                solver.assert_and_track(expr, tracker)

        result = solver.check()

        if result == sat:
            evaluation = EvaluationResult(
                is_consistent=True,
                conflicting_formulas=[],
                details="All facts and active formulas are consistent."
            )
        else:
            # Get conflicting formulas
            core = solver.unsat_core()
            conflicting = [str(c).replace('__track_', '') for c in core]

            evaluation = EvaluationResult(
                is_consistent=False,
                conflicting_formulas=conflicting,
                details=f"Found {len(conflicting)} formula(s) involved in the contradiction."
            )

        self.state.last_evaluation = evaluation
        self.needs_evaluation = False
        return evaluation

    def get_changes_summary(self) -> list[str]:
        """Get a summary of changes from the original state."""
        changes = []

        for name, value in self.state.fact_values.items():
            if not value:  # Changed from TRUE to FALSE
                desc = self.atoms[name]['description'] or name.replace('_', ' ')
                changes.append(f"Fact \"{desc}\" set to FALSE")

        for fid, state in self.state.formula_states.items():
            if state == FormulaState.OFF:
                changes.append(f"Formula \"{fid}\" is OFF (ignored)")

        return changes

    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _display_header(self):
        """Display the simulator header."""
        print("═" * 70)
        print("  🔬 WHAT-IF SIMULATOR - Interactive Mode")
        print("═" * 70)
        if self.filename:
            print(f"\n📄 Loaded: {self.filename}")

    def _display_facts_table(self):
        """Display the facts table with current values."""
        print("\n" + "═" * 70)
        print("  📋 FACTS")
        print("═" * 70)
        print(f"\n  {'#':<4} {'Status':<10} {'Fact'}")
        print("  " + "─" * 64)

        for i, (name, info) in enumerate(self.atoms.items(), 1):
            value = self.state.fact_values[name]
            status = "✅ TRUE" if value else "❌ FALSE"
            desc = info['description'] or name.replace('_', ' ')

            # Highlight changed values
            marker = "" if value else "  ← CHANGED"
            print(f"  {i:<4} {status:<10} \"{desc}\"{marker}")

    def _display_formulas_table(self):
        """Display the formulas table with current states."""
        print("\n" + "═" * 70)
        print("  🔗 FORMULAS (Reasoning Rules)")
        print("═" * 70)
        print(f"\n  {'#':<4} {'Status':<10} {'Rule'}")
        print("  " + "─" * 64)

        for i, (fid, expr) in enumerate(self.formulas.items()):
            state = self.state.formula_states[fid]
            status = "📝 ON" if state == FormulaState.ON else "⏸️  OFF"
            plain = self.converter.expr_to_plain_english(expr)

            # Truncate if too long
            if len(plain) > 50:
                plain = plain[:47] + "..."

            marker = "" if state == FormulaState.ON else "  ← IGNORED"
            letter = chr(ord('A') + i)
            print(f"  {letter:<4} {status:<10} {plain}{marker}")

    def _display_status(self):
        """Display current evaluation status."""
        print("\n" + "═" * 70)

        if self.needs_evaluation:
            print("  ⚡ STATUS: Not yet evaluated (select 'Evaluate' from menu)")
        elif self.state.last_evaluation:
            eval_result = self.state.last_evaluation
            if eval_result.is_consistent:
                print("  ⚡ STATUS: ✅ CONSISTENT")
            else:
                print("  ⚡ STATUS: ❌ INCONSISTENT")
                if eval_result.conflicting_formulas:
                    formulas_str = ", ".join(eval_result.conflicting_formulas)
                    print(f"     Conflicting formulas: {formulas_str}")

        print("═" * 70)

    def _display_current_state(self):
        """Display the complete current state."""
        self._clear_screen()
        self._display_header()
        self._display_facts_table()
        self._display_formulas_table()
        self._display_status()

    def _build_main_menu(self) -> InteractiveMenu:
        """Build the main interactive menu."""
        menu = InteractiveMenu("SELECT AN ACTION")

        # Facts section
        facts_section = MenuSection("TOGGLE FACTS (TRUE ↔ FALSE)")
        for i, (name, info) in enumerate(self.atoms.items(), 1):
            value = self.state.fact_values[name]
            status = "TRUE" if value else "FALSE"
            desc = info['description'] or name.replace('_', ' ')
            short_desc = desc[:40] + "..." if len(desc) > 40 else desc
            label = f"[{i}] {status:<6} │ {short_desc}"
            facts_section.add_item(MenuItem(label, action=self.toggle_fact, data=name))
        menu.add_section(facts_section)

        # Formulas section
        formulas_section = MenuSection("TOGGLE FORMULAS (ON ↔ OFF)")
        for i, (fid, expr) in enumerate(self.formulas.items()):
            state = self.state.formula_states[fid]
            status = "ON" if state == FormulaState.ON else "OFF"
            letter = chr(ord('A') + i)
            plain = self.converter.expr_to_plain_english(expr)
            short_plain = plain[:35] + "..." if len(plain) > 35 else plain
            label = f"[{letter}] {status:<6} │ {short_plain}"
            formulas_section.add_item(MenuItem(label, action=self.toggle_formula, data=fid))
        menu.add_section(formulas_section)

        # Actions section
        actions_section = MenuSection("ACTIONS")
        actions_section.add_item(MenuItem("[E] 📊 Evaluate current scenario", action="evaluate"))
        actions_section.add_item(MenuItem("[D] 📄 Show detailed analysis", action="detailed"))
        actions_section.add_item(MenuItem("[R] 🔄 Reset to original state", action="reset"))
        actions_section.add_item(MenuItem("[Q] 🚪 Quit simulator", action="quit"))
        menu.add_section(actions_section)

        return menu

    def _show_evaluation_result(self):
        """Show the evaluation result in a nice format."""
        if self.state.last_evaluation is None:
            result = self.evaluate()
        else:
            result = self.state.last_evaluation

        self._clear_screen()
        print("═" * 70)
        print("  📊 EVALUATION RESULT")
        print("═" * 70)

        # Show what changes were made
        changes = self.get_changes_summary()
        if changes:
            print("\n  Testing scenario with these changes:")
            for change in changes:
                print(f"    • {change}")
        else:
            print("\n  Testing original scenario (no changes made)")

        print("\n" + "─" * 70)

        if result.is_consistent:
            print("\n  ✅ CONSISTENT!")
            print("\n  All the facts and active reasoning rules can be true simultaneously.")
            print("  There are no logical contradictions in this scenario.")
        else:
            print("\n  ❌ INCONSISTENT!")
            print(f"\n  {result.details}")
            if result.conflicting_formulas:
                print("\n  The following formulas are involved in the contradiction:")
                for fid in result.conflicting_formulas:
                    expr = self.formulas[fid]
                    plain = self.converter.expr_to_plain_english(expr)
                    print(f"    • {fid}: {plain}")

        print("\n" + "═" * 70)
        input("\nPress ENTER to continue...")

    def _show_detailed_analysis(self):
        """Show a detailed plain-English analysis."""
        # Force evaluation if needed
        if self.needs_evaluation:
            self.evaluate()

        self._clear_screen()
        print("═" * 70)
        print("  📄 DETAILED ANALYSIS")
        print("═" * 70)

        # Current configuration
        print("\n┌─ CURRENT CONFIGURATION ─────────────────────────────────────────────┐")

        print("│")
        print("│  FACTS:")
        for name, info in self.atoms.items():
            value = self.state.fact_values[name]
            status = "TRUE" if value else "FALSE"
            desc = info['description'] or name.replace('_', ' ')
            print(f"│    [{status:<5}] \"{desc}\"")

        print("│")
        print("│  FORMULAS:")
        for fid, expr in self.formulas.items():
            state = self.state.formula_states[fid]
            status = "ON" if state == FormulaState.ON else "OFF"
            plain = self.converter.expr_to_plain_english(expr)
            print(f"│    [{status:<3}] {fid}")
            print(f"│    {plain}")

        print("│")
        print("└──────────────────────────────────────────────────────────────────────┘")

        # Result
        result = self.state.last_evaluation
        print("\n┌─ RESULT ─────────────────────────────────────────────────────────────┐")
        print("│")

        if result.is_consistent:
            print("│  ✅ This scenario is LOGICALLY CONSISTENT")
            print("│")
            print("│  All the stated facts and reasoning rules work together without")
            print("│  any contradictions.")
        else:
            print("│  ❌ This scenario is LOGICALLY INCONSISTENT")
            print("│")
            print("│  The facts and rules contradict each other. They cannot all be")
            print("│  true at the same time.")
            print("│")
            if result.conflicting_formulas:
                print("│  Formulas involved in the contradiction:")
                for fid in result.conflicting_formulas:
                    print(f"│    • {fid}")

        print("│")
        print("└──────────────────────────────────────────────────────────────────────┘")

        input("\nPress ENTER to continue...")

    def run(self):
        """Main interactive loop."""
        running = True

        while running:
            self._display_current_state()
            menu = self._build_main_menu()
            selected = menu.show()

            if selected is None:
                continue

            # Handle actions
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
                # Toggle fact or formula
                selected.action(selected.data)

        print("\nGoodbye! 👋\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\n" + "═" * 60)
        print("  🔬 PLOG WHAT-IF SIMULATOR")
        print("═" * 60)
        print("\nAnalyze text and explore what-if scenarios.")
        print("\nUsage:")
        print("  python plog_simulator.py <plog_file>                  # Run analysis and exit")
        print("  python plog_simulator.py --interactive <plog_file>    # Interactive mode")
        print("\nExamples:")
        print("  python plog_simulator.py speech.plog")
        print("  python plog_simulator.py -i test-alibi.plog")
        print("\n" + "═" * 60 + "\n")
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    interactive = "--interactive" in args or "-i" in args

    # Remove flags to get filename
    plog_file = [a for a in args if not a.startswith("-")][0]

    # Find grammar file
    script_dir = Path(__file__).parent
    grammar_candidates = [
        Path(plog_file).parent / "plog.lark",
        script_dir / "plog.lark",
        Path("plog.lark"),
    ]

    grammar_file = None
    for candidate in grammar_candidates:
        if candidate.exists():
            grammar_file = str(candidate)
            break

    if grammar_file is None:
        print(f"❌ Error: Could not find plog.lark grammar file")
        sys.exit(2)

    try:
        # Parse the file
        parser = PlogParser(grammar_file)
        parsed = parser.parse_file(plog_file)

        if not parsed['atoms']:
            print("❌ Error: No atoms (facts) found in the plog file")
            sys.exit(2)

        if interactive:
            # Interactive mode: run the simulator
            simulator = WhatIfSimulator(parsed, filename=plog_file)
            simulator.run()
        else:
            # Non-interactive mode: evaluate and print report, then exit
            analyzer = PlainEnglishAnalyzer(parsed)
            report = analyzer.generate_report()
            print_plain_english_report(report)
            sys.exit(0 if report.is_consistent else 1)

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
