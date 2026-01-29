"""
Plog Reasoning Analyzer - Plain English Edition

Analyzes speech transcripts for logical contradictions and explains
findings in simple, everyday language that anyone can understand.

Usage:
    python plog_reasoning_analyzer_plain.py <plog_file>
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from lark import Lark, Transformer, v_args
from z3 import (
    Bool, And, Or, Not, Xor, Implies, BoolVal, 
    Solver, sat, unsat, BoolRef
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReasoningFlaw:
    """Represents a detected flaw in reasoning."""
    formula_id: str
    expression: str
    flaw_type: str
    involved_facts: list[str]
    plain_explanation: str


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
        # Store the original formula structure for plain English conversion
        self.formula_structures = {}
    
    def start(self, items):
        return {
            'atoms': self.atoms,
            'formulas': self.formulas,
            'formula_structures': self.formula_structures
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
# Plain English Expression Converter
# =============================================================================

class PlainEnglishConverter:
    """Converts Z3 expressions and analysis results to plain English."""
    
    def __init__(self, atoms: dict):
        self.atoms = atoms
    
    def get_fact_description(self, atom_name: str) -> str:
        """Get the human description of a fact, or format the name nicely."""
        if atom_name in self.atoms and self.atoms[atom_name]['description']:
            return f'"{self.atoms[atom_name]["description"]}"'
        # Convert snake_case to readable text
        return f'"{atom_name.replace("_", " ")}"'
    
    def expr_to_plain_english(self, expr, depth=0) -> str:
        """Convert a Z3 expression to plain English."""
        expr_str = str(expr)
        
        # Handle simple atoms
        if expr.num_args() == 0:
            if expr_str == 'True':
                return "something that's always true"
            elif expr_str == 'False':
                return "something that's always false"
            else:
                return self.get_fact_description(expr_str)
        
        # Handle compound expressions
        decl_name = expr.decl().name()
        
        if decl_name == 'not':
            inner = self.expr_to_plain_english(expr.arg(0), depth + 1)
            # Clean up double negatives in description
            if inner.startswith('"') and inner.endswith('"'):
                return f"the opposite of {inner}"
            return f"it's NOT the case that {inner}"
        
        elif decl_name == 'and':
            parts = [self.expr_to_plain_english(expr.arg(i), depth + 1) 
                     for i in range(expr.num_args())]
            if len(parts) == 2:
                return f"both {parts[0]} AND {parts[1]}"
            else:
                return f"all of these: {', '.join(parts[:-1])}, and {parts[-1]}"
        
        elif decl_name == 'or':
            parts = [self.expr_to_plain_english(expr.arg(i), depth + 1) 
                     for i in range(expr.num_args())]
            if len(parts) == 2:
                return f"either {parts[0]} OR {parts[1]}"
            else:
                return f"at least one of: {', '.join(parts[:-1])}, or {parts[-1]}"
        
        elif decl_name == '=>':  # Implies
            left = self.expr_to_plain_english(expr.arg(0), depth + 1)
            right = self.expr_to_plain_english(expr.arg(1), depth + 1)
            return f"IF {left}, THEN {right}"
        
        elif decl_name == 'xor':
            left = self.expr_to_plain_english(expr.arg(0), depth + 1)
            right = self.expr_to_plain_english(expr.arg(1), depth + 1)
            return f"either {left} or {right}, but not both"
        
        # Fallback
        return f"[{expr_str}]"
    
    def explain_flaw_type(self, flaw_type: str, formula_id: str, 
                          involved_facts: list[str], expr) -> str:
        """Generate a plain English explanation for a type of flaw."""
        
        if flaw_type == 'self_contradictory':
            return (
                f"🚫 This statement is IMPOSSIBLE. It's like saying "
                f"\"it's raining and not raining at the same time.\" "
                f"No matter what, this can never be true. "
                f"The speaker said something that contradicts itself within the same breath."
            )
        
        elif flaw_type == 'contradicts_facts':
            facts_mentioned = [self.get_fact_description(f.split(':')[0].strip()) 
                              for f in involved_facts]
            facts_str = " and ".join(facts_mentioned) if facts_mentioned else "the stated facts"
            
            return (
                f"🔴 This reasoning DIRECTLY CONTRADICTS what the speaker also claimed as fact. "
                f"They said {facts_str} are true, but this statement can't be true "
                f"if those facts are true. They're essentially saying \"X is true\" "
                f"and then making an argument that requires \"X is false.\""
            )
        
        elif flaw_type == 'conflicts_with_reasoning':
            return (
                f"⚠️ This statement, on its own, could be true. But when you put it together "
                f"with the speaker's OTHER statements, they can't ALL be true at the same time. "
                f"The speaker has made multiple claims that are individually reasonable "
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
        facts_lines = ["Here's what the speaker claimed as facts:\n"]
        for i, (atom_name, info) in enumerate(self.atoms.items(), 1):
            desc = info['description'] or atom_name.replace('_', ' ')
            facts_lines.append(f"  {i}. \"{desc}\"")
        facts_summary = "\n".join(facts_lines)
        
        # === REASONING SUMMARY ===
        reasoning_lines = ["Here's the reasoning/connections the speaker made:\n"]
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
                f"everything the speaker said is internally consistent. "
                f"Their reasoning holds together - there are no logical contradictions. "
                f"This doesn't mean everything they said is TRUE, just that "
                f"their statements don't contradict each other."
            )
        else:
            bottom_line = (
                f"👎 BOTTOM LINE: The speaker's statements don't add up. "
                f"They made claims that logically cannot all be true at the same time. "
                f"Either some of their \"facts\" are wrong, or their reasoning is flawed. "
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
    print("  🔍 SPEECH LOGIC CHECKER - ANALYSIS REPORT")
    print("=" * width)
    
    # Verdict banner
    print("\n" + "-" * width)
    if report.is_consistent:
        print(f"  {report.verdict}")
    else:
        print(f"  {report.verdict}")
    print("-" * width)
    
    # What was said (facts)
    print("\n📋 WHAT THE SPEAKER CLAIMED AS FACTS:")
    print("-" * width)
    print(report.facts_summary)
    
    # Reasoning/connections
    print("\n🔗 THE LOGICAL CONNECTIONS THEY MADE:")
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
# Main Entry Point
# =============================================================================

def analyze_speech(filepath: str, grammar_path: str = 'plog.lark') -> PlainEnglishReport:
    """Analyze a plog file and return a plain English report."""
    parser = PlogParser(grammar_path)
    parsed = parser.parse_file(filepath)
    analyzer = PlainEnglishAnalyzer(parsed)
    return analyzer.generate_report()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "=" * 60)
        print("  🔍 SPEECH LOGIC CHECKER")
        print("=" * 60)
        print("\nThis tool analyzes speech transcripts (converted to .plog format)")
        print("to find logical contradictions and inconsistencies.")
        print("\nUsage:")
        print("  python plog_reasoning_analyzer_plain.py <plog_file>")
        print("\nExample:")
        print("  python plog_reasoning_analyzer_plain.py speech.plog")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)
    
    plog_file = sys.argv[1]
    grammar_file = sys.argv[2] if len(sys.argv) > 2 else 'plog.lark'
    
    try:
        report = analyze_speech(plog_file, grammar_file)
        print_plain_english_report(report)
        sys.exit(0 if report.is_consistent else 1)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file - {e}\n")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(2)
