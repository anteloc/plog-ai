#!/usr/bin/env python3
"""
plog_simulator.py - Simplified PLOG contradiction checker

Outputs JSON:
- SAT case: { "FACTS": [...], "FORMULAE": [...] }
- UNSAT case: { "unsat_atoms": [...], "unsat_formulae": [...] }
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import yaml
from lark import Lark, Transformer, Token, Tree
from lark.exceptions import UnexpectedInput
from z3 import Solver, Bool, BoolVal, BoolRef, Not, And, Or, Xor, Implies, sat, unsat


def unquote_desc(s: str) -> str:
    """Remove quotes from description string."""
    if s.startswith('"""') and s.endswith('"""'):
        return s[3:-3]
    return ast.literal_eval(s)


def reconstruct_expr(tree: Tree) -> str:
    """Reconstruct expression string from parse tree."""
    return " ".join(
        str(t) for t in tree.scan_values(lambda v: isinstance(v, Token))
    ).strip()


def get_atoms_in_expr(tree: Tree) -> Set[str]:
    """Extract all atom names referenced in an expression."""
    return {
        str(t)
        for t in tree.scan_values(lambda v: isinstance(v, Token))
        if t.type == "ATOM_NAME"
    }


class ExprToZ3(Transformer):
    """Transform parse tree to Z3 expression."""
    
    def __init__(self, z3_vars: Dict[str, BoolRef]):
        super().__init__()
        self.v = z3_vars

    def ATOM_NAME(self, t: Token) -> str:
        return str(t)

    def atom_ref(self, items: List[Any]) -> BoolRef:
        return self.v[items[0]]

    def true(self, _): return BoolVal(True)
    def false(self, _): return BoolVal(False)

    def neg(self, items: List[Any]) -> BoolRef:
        return Not(items[-1])

    def implies(self, items: List[Any]) -> BoolRef:
        if len(items) == 1:
            return items[0]
        # Filter out tokens, keep only expressions
        exprs = [i for i in items if not isinstance(i, Token)]
        if len(exprs) == 2:
            return Implies(exprs[0], exprs[1])
        return items[0]

    def andlike(self, items: List[Any]) -> BoolRef:
        return self._fold(items, "and")

    def orlike(self, items: List[Any]) -> BoolRef:
        return self._fold(items, "or")

    def iff(self, items: List[Any]) -> BoolRef:
        acc = items[0]
        for op, rhs in zip(items[1::2], items[2::2]):
            acc = (acc == rhs)
        return acc

    def _fold(self, items: List[Any], level: str) -> BoolRef:
        acc = items[0]
        for op, rhs in zip(items[1::2], items[2::2]):
            if not isinstance(op, Token):
                continue
            if level == "and":
                if op.type == "AND":
                    acc = And(acc, rhs)
                elif op.type == "NAND":
                    acc = Not(And(acc, rhs))
            elif level == "or":
                if op.type == "OR":
                    acc = Or(acc, rhs)
                elif op.type == "NOR":
                    acc = Not(Or(acc, rhs))
                elif op.type == "XOR":
                    acc = Xor(acc, rhs)
        return acc


def load_plog(plog_path: str, grammar_path: str) -> Tuple[Dict[str, str], List[Tuple[str, str, Tree]], Dict[str, BoolRef]]:
    """
    Load and parse a .plog file.
    
    Returns:
        atoms: dict of atom_name -> description
        formulas: list of (formula_id, raw_expr, expr_tree)
        formulas_z3: dict of formula_id -> Z3 expression
    """
    grammar = Path(grammar_path).read_text(encoding="utf-8")
    parser = Lark(grammar, parser="lalr", start="start", propagate_positions=True)
    
    text = Path(plog_path).read_text(encoding="utf-8")
    tree = parser.parse(text)
    
    # Extract atoms
    atoms: Dict[str, str] = {}
    for stmt in tree.find_data("atom_decl"):
        name = str(stmt.children[0])
        desc_node = stmt.children[1]
        tok = desc_node.children[0] if isinstance(desc_node, Tree) else desc_node
        atoms[name] = unquote_desc(str(tok))
    
    # Extract formulas
    formulas: List[Tuple[str, str, Tree]] = []
    for stmt in tree.find_data("formula_decl"):
        fid = str(stmt.children[0])
        expr_tree = stmt.children[1]
        raw = reconstruct_expr(expr_tree)
        formulas.append((fid, raw, expr_tree))
    
    # Build Z3 expressions
    z3_vars = {a: Bool(a) for a in atoms}
    xform = ExprToZ3(z3_vars)
    formulas_z3 = {fid: xform.transform(expr_tree) for fid, _, expr_tree in formulas}
    
    return atoms, formulas, formulas_z3


def expand_expr(raw: str, atoms: Dict[str, str], force_values: Dict[str, bool]) -> str:
    """Replace atom names with their descriptions in an expression."""
    result = raw
    # Sort by length descending to avoid partial replacements
    for atom in sorted(atoms.keys(), key=len, reverse=True):
        is_altered = atom in force_values and force_values[atom] is False
        expanded_desc = atoms[atom] if not is_altered else f"~ {atoms[atom]}"
        result = result.replace(atom, f'"{expanded_desc}"')
    return result

def tag_expr_atoms(raw: str, atom_to_tag: Dict[str, str], force_values: Dict[str, bool]) -> str:
    """Replace atom names with their descriptions in an expression."""
    result = raw
    # Sort by length descending to avoid partial replacements
    for atom in sorted(atom_to_tag.keys(), key=len, reverse=True):
        is_altered = atom in force_values and force_values[atom] is False
        altered_tag = atom_to_tag[atom] if not is_altered else f"{atom_to_tag[atom]} ~"
        result = result.replace(atom, f"({altered_tag}) {atom}")
    return result

def build_tagging_data(atoms: Dict[str, str], formulas: List[Tuple[str, str, Tree]]):
    # Build tag mappings
    atom_list = sorted(atoms.keys())
    formula_list = [fid for fid, _, _ in formulas]
    
    atom_tags = {f"a{i+1}": atom for i, atom in enumerate(atom_list)}
    formula_tags = {f"f{i+1}": fid for i, fid in enumerate(formula_list)}
    
    # Reverse mappings for display
    atom_to_tag = {v: k for k, v in atom_tags.items()}
    formula_to_tag = {v: k for k, v in formula_tags.items()}
    
    return atom_list, formula_list, atom_tags, formula_tags, atom_to_tag, formula_to_tag


def shrink_to_mus(solver: Solver, assumptions: List[BoolRef]) -> List[BoolRef]:
    """Shrink unsat core to minimal unsat subset."""
    core = assumptions[:]
    while True:
        for i in range(len(core)):
            trial = core[:i] + core[i + 1:]
            if solver.check(*trial) == unsat:
                core = trial
                break
        else:
            return core


def check_plog(atoms: Dict[str, str], atom_to_tag: Dict[str, str], formulas: List[Tuple[str, str, Tree]], formula_to_tag: Dict[str, str], formulas_z3: Dict[str, BoolRef], force_values: Dict[str, bool] = None, negated_formulas: Set[str] = None) -> dict:
    """
    Check satisfiability and return JSON result.
    
    Automatically asserts all atoms as TRUE (the speaker asserts all their statements),
    unless overridden by force_values.
    
    Args:
        force_values: dict of atom_name -> bool to override default TRUE assertion
        negated_formulas: set of formula IDs to negate (wrap in NOT)
    """
    
    z3_vars = {a: Bool(a) for a in atoms}
    
    # Create assumption variables for each formula
    form_assump = {fid: Bool(f"A_{fid}") for fid, _, _ in formulas}
    
    # Create assumption variables for each atom
    # For forced FALSE atoms, we need a separate assumption
    atom_assump_true = {a: Bool(f"ATOM_{a}_T") for a in atoms}
    atom_assump_false = {a: Bool(f"ATOM_{a}_F") for a in atoms}

    # Build solver
    solver = Solver()
    
    # Add formula implications (negate if in negated_formulas)
    for fid, fexpr in formulas_z3.items():
        if fid in negated_formulas:
            solver.add(Implies(form_assump[fid], Not(fexpr)))
        else:
            solver.add(Implies(form_assump[fid], fexpr))
    
    # Add atom assertions
    for a in atoms:
        solver.add(Implies(atom_assump_true[a], z3_vars[a]))       # when enabled, atom = TRUE
        solver.add(Implies(atom_assump_false[a], Not(z3_vars[a]))) # when enabled, atom = FALSE
    
    # Enable all formulas
    enabled_formulas = [form_assump[fid] for fid, _, _ in formulas]
    
    # Enable atom assertions based on force_values (default: TRUE)
    enabled_atoms = []
    for a in atoms:
        if a in force_values:
            if force_values[a]:
                enabled_atoms.append(atom_assump_true[a])
            else:
                enabled_atoms.append(atom_assump_false[a])
        else:
            # Default: assert as TRUE
            enabled_atoms.append(atom_assump_true[a])
    
    enabled = enabled_formulas + enabled_atoms
    
    if solver.check(*enabled) == sat:
        # SAT case
        model = solver.model()
        
        facts = []
        for atom_name in sorted(atoms.keys()):
            val = model.evaluate(z3_vars[atom_name], model_completion=True)
            is_true = str(val) == "True"
            is_altered = atom_name in force_values and force_values[atom_name] is False
            alt_atom_tag = atom_to_tag[atom_name] if not is_altered else f"{atom_to_tag[atom_name]} ~"
            entry = {"NAME": f"({alt_atom_tag}) {atom_name}"}
            if is_true:
                entry["AFFIRMS"] = atoms[atom_name]
            else:
                entry["DENIES"] = atoms[atom_name]
            facts.append(entry)
        
        sat_formulae = []
        for fid, raw, _ in formulas:
            is_negated = fid in negated_formulas
            alt_atom_tag = formula_to_tag[fid] if not is_negated else f"{formula_to_tag[fid]} ~"
            
            entry = {
                "ID": f"({alt_atom_tag}) {fid}",
                "DEF": f"~ ({tag_expr_atoms(raw, atom_to_tag, force_values)})" if is_negated else tag_expr_atoms(raw, atom_to_tag, force_values),
                "MEANING": f"~ ({expand_expr(raw, atoms, force_values)})" if is_negated else expand_expr(raw, atoms, force_values)
            }
            sat_formulae.append(entry)
        
        return {"FACTS": facts, "FORMULAE": sat_formulae}
    
    else:
        # UNSAT case - find minimal contradiction
        core = list(solver.unsat_core())
        if core:
            enabled_str = {str(a) for a in enabled}
            core = [c for c in core if str(c) in enabled_str] or enabled
        else:
            core = enabled
        
        mus = shrink_to_mus(solver, core)
        
        # Extract formula IDs and atom names from MUS
        mus_fids = set()
        mus_atom_assertions = {}  # atom_name -> asserted_value (True/False)
        
        for a in mus:
            s = str(a)
            if s.startswith("A_"):
                mus_fids.add(s[2:])
            elif s.startswith("ATOM_") and s.endswith("_T"):
                atom_name = s[5:-2]
                mus_atom_assertions[atom_name] = True
            elif s.startswith("ATOM_") and s.endswith("_F"):
                atom_name = s[5:-2]
                mus_atom_assertions[atom_name] = False
        
        # Collect atoms involved in contradicting formulas
        atoms_in_formulas = set()
        for fid, _, expr_tree in formulas:
            if fid in mus_fids:
                atoms_in_formulas.update(get_atoms_in_expr(expr_tree))
        
        # Atoms involved = those asserted + those in contradicting formulas
        involved_atoms = set(mus_atom_assertions.keys()) | atoms_in_formulas
        
        unsat_atoms = []
        for atom_name in sorted(involved_atoms):
            if atom_name in atoms:
                is_altered = atom_name in force_values and force_values[atom_name] is False
                alt_atom_tag = atom_to_tag[atom_name] if not is_altered else f"{atom_to_tag[atom_name]} ~"
                entry = {"NAME": f"({alt_atom_tag}) {atom_name}"}
                conflicting_val = mus_atom_assertions.get(atom_name, None)
                if conflicting_val is True:
                    entry["AFFIRMING"] = atoms[atom_name]
                elif conflicting_val is False:
                    entry["DENYING"] = atoms[atom_name]
                else:
                    entry["AFFIRMING_OR_DENYING"] = atoms[atom_name]
                unsat_atoms.append(entry)
        
        unsat_formulae = []
        for fid, raw, _ in formulas:
            if fid in mus_fids:
                is_negated = fid in negated_formulas
                alt_atom_tag = formula_to_tag[fid] if not is_negated else f"{formula_to_tag[fid]} ~"
                entry = {
                    "ID": f"({alt_atom_tag}) {fid}",
                    "UNSATISFIABLE": f"~ ({tag_expr_atoms(raw, atom_to_tag, force_values)})" if is_negated else tag_expr_atoms(raw, atom_to_tag, force_values),
                    "MEANING": f"~ ({expand_expr(raw, atoms, force_values)})" if is_negated else expand_expr(raw, atoms, force_values)
                }
                unsat_formulae.append(entry)
        
        return {"CONFLICTING_FACTS": unsat_atoms, "CONFLICTING_FORMULAE": unsat_formulae}


def parse_force_values(force_args: List[str]) -> Dict[str, bool]:
    """Parse --force-value arguments like 'AtomName=true' or 'AtomName=false'."""
    result = {}
    if not force_args:
        return result
    
    for arg in force_args:
        if '=' not in arg:
            raise ValueError(f"Invalid --force-value format: '{arg}'. Expected 'AtomName=true' or 'AtomName=false'")
        
        name, value = arg.split('=', 1)
        name = name.strip()
        value = value.strip().lower()
        
        if value in ('true', '1', 'yes'):
            result[name] = True
        elif value in ('false', '0', 'no'):
            result[name] = False
        else:
            raise ValueError(f"Invalid boolean value: '{value}'. Expected 'true' or 'false'")
    
    return result


def parse_negations(not_args: List[str], atoms: Dict[str, str], formula_ids: List[str]) -> Tuple[Dict[str, bool], Set[str]]:
    """
    Parse --not arguments.
    
    Returns:
        negated_atoms: dict of atom_name -> False (to be merged with force_values)
        negated_formulas: set of formula IDs to negate
    """
    negated_atoms = {}
    negated_formulas = set()
    
    if not not_args:
        return negated_atoms, negated_formulas
    
    for arg in not_args:
        arg = arg.strip()
        if arg in atoms:
            negated_atoms[arg] = False
        elif arg in formula_ids:
            negated_formulas.add(arg)
        else:
            raise ValueError(f"Unknown atom or formula in --not: '{arg}'")
    
    return negated_atoms, negated_formulas


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def interactive_mode(atoms: Dict[str, str], 
                     formulas: List[Tuple[str, str, Tree]], 
                     formulas_z3: Dict[str, BoolRef],
                     initial_force_values: Dict[str, bool], 
                     initial_negated_formulas: Set[str]):
    """
    Run interactive mode with menu for toggling atoms and formulas.
    """
    # Current state (copy from initial)
    force_values = dict(initial_force_values)
    negated_formulas = set(initial_negated_formulas)
    
    atom_list, _, atom_tags, formula_tags, atom_to_tag, formula_to_tag = build_tagging_data(atoms, formulas)
    
    while True:
        clear_screen()
        
        # Display FACTS section
        print("=" * 80)
        print("FACTS")
        print("=" * 80)

        max_atom_name_len = max(len(a) for a in atom_list) if atom_list else 0
        
        for atom_name in atom_list:
            tag = atom_to_tag[atom_name]
            desc = atoms[atom_name]
            
            # Determine if altered (negated means value is FALSE)
            is_altered = atom_name in force_values and force_values[atom_name] is False

            value = "FALSE" if is_altered else "TRUE "
            prefix = "(~)" if is_altered else "   "

            prefix_tag_atom_name = f"{prefix} [{tag:>3}] {atom_name.ljust(max_atom_name_len)}"
            
            print(f'{prefix_tag_atom_name}: {value} - "{desc}"')
        
        print()
        
        # Display FORMULAE section
        print("=" * 80)
        print("FORMULAE")
        print("=" * 80)
        
        for fid, raw, _ in formulas:
            tag = formula_to_tag[fid]
            expanded = expand_expr(raw, atoms, force_values)
            tagged_expr_atoms = tag_expr_atoms(raw, atom_to_tag, force_values)
            
            is_negated = fid in negated_formulas
            prefix = "(~)" if is_negated else "   "
            
            if is_negated:
                expr_display = f"~ ({tagged_expr_atoms})"
                expanded_display = f"~ ({expanded})"
            else:
                expr_display = tagged_expr_atoms
                expanded_display = expanded
            
            print(f"{prefix} [{tag:>3}] {fid}:")
            print(f"         {expr_display}")
            print(f"         {expanded_display}")
            print()
        
        # Prompt
        print("=" * 80)
        print("Enter: ")
        print("[tag] to toggle NOT (~) an item (e.g., 'a5' -> '~a5' -> 'a5')")
        print("[e] to evaluate")
        print("[q] to quit")
        print("=" * 80)
        
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if choice == 'q':
            break
        
        elif choice == 'e':
            # Evaluate
            result = check_plog(atoms, atom_to_tag, formulas, formula_to_tag, formulas_z3, force_values, negated_formulas)
            
            clear_screen()
            print("=" * 80)
            print("EVALUATION RESULT")
            print("=" * 80)
            print()
            print(yaml.dump(result, default_flow_style=False, allow_unicode=True, sort_keys=False))
            print("=" * 80)
            input("Press Enter to return to menu...")
        
        elif choice in atom_tags:
            # Toggle atom
            atom_name = atom_tags[choice]
            if atom_name in force_values and force_values[atom_name] is False:
                # Currently altered (FALSE), revert to unaltered (TRUE)
                del force_values[atom_name]
            else:
                # Currently unaltered (TRUE), set to altered (FALSE)
                force_values[atom_name] = False
        
        elif choice in formula_tags:
            # Toggle formula
            fid = formula_tags[choice]
            if fid in negated_formulas:
                negated_formulas.remove(fid)
            else:
                negated_formulas.add(fid)
        
        else:
            # Invalid input, just redraw menu
            pass


def main():
    ap = argparse.ArgumentParser(description="PLOG contradiction checker - outputs JSON")
    ap.add_argument("file", help="Path to .plog file")
    ap.add_argument("--grammar", "-g", default=None, help="Path to plog.lark grammar (default: same dir as script)")
    ap.add_argument("--force-value", "-f", action="append", dest="force_values", metavar="ATOM=BOOL",
                    help="Force an atom to a specific value (e.g., --force-value WarEnemyOfPoor=false). Can be repeated.")
    ap.add_argument("--not", "-n", action="append", dest="negations", metavar="NAME",
                    help="Negate an atom (set to false) or formula (wrap in NOT). Can be repeated.")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Run in interactive mode with menu for toggling atoms and formulas.")
    args = ap.parse_args()
    
    # Find grammar file
    grammar_path = args.grammar
    if grammar_path is None:
        grammar_path = Path(__file__).parent / "plog.lark"
    
    if not Path(grammar_path).exists():
        print(json.dumps({"error": f"Grammar file not found: {grammar_path}"}), file=sys.stderr)
        return 1
    
    try:
        # Parse force values
        force_values = parse_force_values(args.force_values)
        
        atoms, formulas, formulas_z3 = load_plog(args.file, grammar_path)
        
        # Validate forced atoms exist
        unknown_atoms = set(force_values.keys()) - set(atoms.keys())
        if unknown_atoms:
            print(json.dumps({"error": f"Unknown atom(s) in --force-value: {', '.join(sorted(unknown_atoms))}"}), file=sys.stderr)
            return 1
        
        # Parse negations
        formula_ids = [fid for fid, _, _ in formulas]
        negated_atoms, negated_formulas = parse_negations(args.negations, atoms, formula_ids)

        # fallacious statements (atoms) and fallacious formulae
        fallacy_atoms = {a for a in atoms.keys() if a.startswith("FALLACY_")}
        fallacy_formulae = {fid for fid in formula_ids if fid.startswith("FALLACY_")}

        # Assume that atom fallacies are always negated, unless explicitly overridden
        for fa in fallacy_atoms:
            if fa not in negated_atoms and fa not in force_values:
                negated_atoms[fa] = False

        for ff in fallacy_formulae:
            if ff not in negated_formulas:
                negated_formulas.add(ff)

        # Merge negated atoms into force_values (negations take precedence)
        for atom, val in negated_atoms.items():
            force_values[atom] = val
        
        if args.interactive:
            interactive_mode(atoms, formulas, formulas_z3, force_values, negated_formulas)
        else:
            _, _, _, _, atom_to_tag, formula_to_tag = build_tagging_data(atoms, formulas)
            result = check_plog(atoms, atom_to_tag, formulas, formula_to_tag, formulas_z3, force_values, negated_formulas)
            print(json.dumps(result, indent=2))

        return 0
    
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
