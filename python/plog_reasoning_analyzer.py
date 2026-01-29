"""
Plog Reasoning Analyzer - Analyzes speech transcripts for logical contradictions

This module parses .plog files (speech transcripts converted to propositional logic)
and identifies flawed reasoning by:
1. Treating all ATOM declarations as TRUE facts (the person's stated claims)
2. Treating all FORMULA declarations as reasoning/inferences
3. Finding which formulas represent flawed reasoning when contradictions occur

Usage:
    python plog_reasoning_analyzer.py <plog_file>
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from lark import Lark, Transformer, v_args
from z3 import (
    Bool, And, Or, Not, Xor, Implies, BoolVal, 
    Solver, sat, unsat, is_true, is_false,
    BoolRef
)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ReasoningFlaw:
    """Represents a detected flaw in reasoning."""
    formula_id: str
    expression: str
    flaw_type: str  # 'self_contradictory', 'contradicts_facts', 'conflicts_with_reasoning'
    involved_facts: list[str]
    explanation: str


@dataclass 
class AnalysisResult:
    """Complete analysis result for a plog file."""
    is_consistent: bool
    atoms: dict
    formulas: dict
    flawed_formulas: list[ReasoningFlaw]
    minimal_conflict_set: list[str]
    summary: str


# =============================================================================
# Lark Transformer: Parse Tree -> Z3 Expressions
# =============================================================================

class PlogToZ3(Transformer):
    """
    Transformer that converts the Lark parse tree into Z3 expressions.
    """
    
    def __init__(self):
        super().__init__()
        self.atoms = {}
        self.formulas = {}
    
    def start(self, items):
        return {
            'atoms': self.atoms,
            'formulas': self.formulas
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
        expr = items[-1]
        return Not(expr)
    
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
# Parser Class
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
# Reasoning Analyzer
# =============================================================================

class ReasoningAnalyzer:
    """
    Analyzes plog files for logical contradictions in reasoning.
    
    Key assumption: All ATOMs are TRUE (facts stated by the speaker).
    Goal: Find which FORMULAs contain flawed reasoning.
    """
    
    def __init__(self, parsed_result: dict):
        self.atoms = parsed_result['atoms']
        self.formulas = parsed_result['formulas']
    
    def _add_facts_as_true(self, solver: Solver):
        """Add all atoms as TRUE constraints (facts are assumed correct)."""
        for atom_name, info in self.atoms.items():
            solver.add(info['z3_var'] == True)
    
    def _extract_atom_names_from_expr(self, expr) -> list[str]:
        """Extract all atom variable names from a Z3 expression."""
        atoms_found = []
        
        def traverse(e):
            if isinstance(e, BoolRef):
                # Check if it's a variable (not a compound expression)
                if e.num_args() == 0 and str(e) not in ('True', 'False'):
                    atoms_found.append(str(e))
                else:
                    for i in range(e.num_args()):
                        traverse(e.arg(i))
        
        traverse(expr)
        return list(set(atoms_found))
    
    def check_basic_consistency(self) -> tuple[bool, Optional[object]]:
        """
        Check if all facts + all reasoning are consistent.
        
        Returns:
            (is_consistent, model_or_none)
        """
        solver = Solver()
        self._add_facts_as_true(solver)
        
        for formula_id, expr in self.formulas.items():
            solver.add(expr)
        
        result = solver.check()
        
        if result == sat:
            return (True, solver.model())
        else:
            return (False, None)
    
    def find_minimal_conflict_set(self) -> list[str]:
        """
        Find the minimal set of formulas that cause the contradiction.
        Uses Z3's unsat core extraction.
        
        Returns:
            List of formula IDs that form the minimal conflicting set.
        """
        solver = Solver()
        solver.set(':core.minimize', True)
        
        # Facts are TRUE (not tracked - we don't blame facts)
        self._add_facts_as_true(solver)
        
        # Track each formula for blame assignment
        trackers = {}
        for formula_id, expr in self.formulas.items():
            tracker = Bool(f'__track_{formula_id}')
            trackers[formula_id] = tracker
            solver.assert_and_track(expr, tracker)
        
        if solver.check() == unsat:
            core = solver.unsat_core()
            conflicting = []
            for c in core:
                name = str(c)
                if name.startswith('__track_'):
                    conflicting.append(name.replace('__track_', ''))
            return conflicting
        
        return []
    
    def classify_formula_flaw(self, formula_id: str) -> str:
        """
        Classify the type of flaw in a specific formula.
        
        Returns one of:
            - 'self_contradictory': Formula is always false (e.g., A & ~A)
            - 'contradicts_facts': Formula cannot be true given the facts
            - 'conflicts_with_reasoning': Formula conflicts with other formulas
            - 'valid': No flaw detected in isolation
        """
        expr = self.formulas[formula_id]
        
        # Type 1: Self-contradictory (formula alone is UNSAT)
        solver = Solver()
        solver.add(expr)
        if solver.check() == unsat:
            return 'self_contradictory'
        
        # Type 2: Contradicts facts (formula + facts is UNSAT)
        solver = Solver()
        self._add_facts_as_true(solver)
        solver.add(expr)
        if solver.check() == unsat:
            return 'contradicts_facts'
        
        # Type 3: Check if it conflicts with other formulas (given facts)
        solver = Solver()
        self._add_facts_as_true(solver)
        for fid, fexpr in self.formulas.items():
            solver.add(fexpr)
        if solver.check() == unsat:
            return 'conflicts_with_reasoning'
        
        return 'valid'
    
    def analyze_formula(self, formula_id: str) -> ReasoningFlaw:
        """
        Perform detailed analysis of a single formula.
        
        Returns:
            ReasoningFlaw object with full details.
        """
        expr = self.formulas[formula_id]
        flaw_type = self.classify_formula_flaw(formula_id)
        
        # Get involved atoms
        involved_atom_names = self._extract_atom_names_from_expr(expr)
        involved_facts = []
        for atom_name in involved_atom_names:
            if atom_name in self.atoms:
                desc = self.atoms[atom_name]['description']
                if desc:
                    involved_facts.append(f"{atom_name}: \"{desc}\"")
                else:
                    involved_facts.append(atom_name)
        
        # Generate human-readable explanation
        explanation = self._generate_explanation(formula_id, flaw_type, involved_facts)
        
        return ReasoningFlaw(
            formula_id=formula_id,
            expression=str(expr),
            flaw_type=flaw_type,
            involved_facts=involved_facts,
            explanation=explanation
        )
    
    def _generate_explanation(self, formula_id: str, flaw_type: str, involved_facts: list[str]) -> str:
        """Generate a human-readable explanation for a flaw."""
        facts_str = ", ".join(involved_facts) if involved_facts else "unknown facts"
        
        if flaw_type == 'self_contradictory':
            return (
                f"Formula '{formula_id}' is self-contradictory — it can never be true "
                f"regardless of the facts. This represents logically impossible reasoning."
            )
        elif flaw_type == 'contradicts_facts':
            return (
                f"Formula '{formula_id}' directly contradicts the stated facts. "
                f"Given that [{facts_str}] are all true, this inference cannot hold."
            )
        elif flaw_type == 'conflicts_with_reasoning':
            return (
                f"Formula '{formula_id}' is individually valid but conflicts with other "
                f"reasoning in the speech. The combination of inferences creates a contradiction."
            )
        else:
            return f"Formula '{formula_id}' appears to be valid."
    
    def find_pairwise_conflicts(self) -> list[tuple[str, str]]:
        """
        Find pairs of formulas that directly contradict each other (given facts).
        
        Returns:
            List of (formula_id_1, formula_id_2) tuples that conflict.
        """
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
    
    def full_analysis(self) -> AnalysisResult:
        """
        Perform complete analysis of the plog file.
        
        Returns:
            AnalysisResult with all findings.
        """
        is_consistent, _ = self.check_basic_consistency()
        
        flawed_formulas = []
        minimal_conflict = []
        
        if not is_consistent:
            # Find minimal conflict set
            minimal_conflict = self.find_minimal_conflict_set()
            
            # Analyze each formula in the conflict set
            for formula_id in minimal_conflict:
                flaw = self.analyze_formula(formula_id)
                flawed_formulas.append(flaw)
        
        # Generate summary
        summary = self._generate_summary(is_consistent, flawed_formulas, minimal_conflict)
        
        return AnalysisResult(
            is_consistent=is_consistent,
            atoms=self.atoms,
            formulas=self.formulas,
            flawed_formulas=flawed_formulas,
            minimal_conflict_set=minimal_conflict,
            summary=summary
        )
    
    def _generate_summary(self, is_consistent: bool, flaws: list[ReasoningFlaw], 
                          conflict_set: list[str]) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = []
        
        if is_consistent:
            lines.append("✓ CONSISTENT: The speaker's reasoning is logically sound.")
            lines.append(f"  All {len(self.atoms)} stated facts and {len(self.formulas)} inferences are compatible.")
        else:
            lines.append("✗ INCONSISTENT: The speaker's reasoning contains contradictions.")
            lines.append(f"  Analyzed {len(self.atoms)} facts and {len(self.formulas)} inferences.")
            lines.append(f"  Found {len(conflict_set)} formula(s) involved in the contradiction:")
            
            for flaw in flaws:
                lines.append(f"\n  → {flaw.formula_id} ({flaw.flaw_type})")
                lines.append(f"    Expression: {flaw.expression}")
                lines.append(f"    {flaw.explanation}")
        
        return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def analyze_plog_file(filepath: str, grammar_path: str = 'plog.lark') -> AnalysisResult:
    """
    Analyze a plog file for logical contradictions.
    
    Args:
        filepath: Path to the .plog file
        grammar_path: Path to the plog.lark grammar file
        
    Returns:
        AnalysisResult with complete analysis
    """
    parser = PlogParser(grammar_path)
    parsed = parser.parse_file(filepath)
    
    analyzer = ReasoningAnalyzer(parsed)
    return analyzer.full_analysis()


def print_detailed_report(result: AnalysisResult):
    """Print a detailed analysis report."""
    print("=" * 70)
    print("PLOG REASONING ANALYSIS REPORT")
    print("=" * 70)
    
    # Facts section
    print("\n[FACTS - Assumed TRUE]")
    print("-" * 70)
    for atom_name, info in result.atoms.items():
        desc = info['description'] or '(no description)'
        print(f"  • {atom_name}: {desc}")
    
    # Formulas section
    print("\n[REASONING - Under Analysis]")
    print("-" * 70)
    for formula_id, expr in result.formulas.items():
        print(f"  • {formula_id}: {expr}")
    
    # Results section
    print("\n[ANALYSIS RESULTS]")
    print("-" * 70)
    print(result.summary)
    
    # Pairwise conflicts (if inconsistent)
    if not result.is_consistent:
        analyzer = ReasoningAnalyzer({'atoms': result.atoms, 'formulas': result.formulas})
        pairwise = analyzer.find_pairwise_conflicts()
        
        if pairwise:
            print("\n[PAIRWISE CONFLICTS]")
            print("-" * 70)
            for id1, id2 in pairwise:
                print(f"  • {id1} ⟷ {id2}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plog_reasoning_analyzer.py <plog_file> [grammar_file]")
        print("\nExample: python plog_reasoning_analyzer.py speech.plog plog.lark")
        sys.exit(1)
    
    plog_file = sys.argv[1]
    grammar_file = sys.argv[2] if len(sys.argv) > 2 else 'plog.lark'
    
    try:
        result = analyze_plog_file(plog_file, grammar_file)
        print_detailed_report(result)
        
        # Exit code: 0 if consistent, 1 if contradictions found
        sys.exit(0 if result.is_consistent else 1)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
