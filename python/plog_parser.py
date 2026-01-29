"""
Plog Parser - A propositional logic DSL parser using Lark and Z3

This module parses .plog files and transforms them into Z3 solver expressions.
"""

from pathlib import Path
from lark import Lark, Transformer, v_args
from z3 import Bool, And, Or, Not, Xor, Implies, BoolVal, Solver, sat, unsat


class PlogToZ3(Transformer):
    """
    Transformer that converts the Lark parse tree into Z3 expressions.
    
    The transformer processes the tree bottom-up, converting each node
    into the corresponding Z3 construct.
    """
    
    def __init__(self):
        super().__init__()
        # Dictionary to store declared atoms as Z3 Bool variables
        self.atoms = {}
        # Dictionary to store formulas
        self.formulas = {}
    
    # --- Statement handlers ---
    
    def start(self, items):
        """Process all statements and return the collected atoms and formulas."""
        return {
            'atoms': self.atoms,
            'formulas': self.formulas
        }
    
    def statement(self, items):
        """Pass through statement content."""
        return items[0] if items else None
    
    @v_args(inline=True)
    def atom_decl(self, name, description):
        """
        Handle ATOM declarations.
        Creates a Z3 Bool variable for each declared atom.
        """
        atom_name = str(name)
        self.atoms[atom_name] = {
            'z3_var': Bool(atom_name),
            'description': description
        }
        return ('atom', atom_name)
    
    @v_args(inline=True)
    def formula_decl(self, formula_id, expr):
        """
        Handle FORMULA declarations.
        Stores the Z3 expression for each formula.
        """
        fid = str(formula_id)
        self.formulas[fid] = expr
        return ('formula', fid, expr)
    
    def desc(self, items):
        """Extract description string."""
        return items[0]
    
    def ESCAPED_STRING(self, token):
        """Remove quotes from escaped strings."""
        return str(token)[1:-1]
    
    def TRIPLE_STRING(self, token):
        """Remove triple quotes from triple-quoted strings."""
        return str(token)[3:-3]
    
    # --- Expression handlers ---
    
    @v_args(inline=True)
    def atom_ref(self, name):
        """
        Reference to a declared atom.
        Returns the Z3 Bool variable for this atom.
        """
        atom_name = str(name)
        if atom_name not in self.atoms:
            # Create the atom if not yet declared (forward reference)
            self.atoms[atom_name] = {
                'z3_var': Bool(atom_name),
                'description': None
            }
        return self.atoms[atom_name]['z3_var']
    
    # Constants
    def true(self, items):
        """Handle TRUE constant."""
        return BoolVal(True)
    
    def false(self, items):
        """Handle FALSE constant."""
        return BoolVal(False)
    
    # Unary operators
    def neg(self, items):
        """Handle NOT operator. Receives [NOT_token, expression]."""
        # Filter out the NOT token, keep only the expression
        expr = items[-1]  # The expression is the last item
        return Not(expr)
    
    # Binary operators - AND/NAND level
    def andlike(self, items):
        """
        Handle AND and NAND operations.
        Format: expr (op expr)*
        """
        if len(items) == 1:
            return items[0]
        
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            
            if op in ('&', '∧'):  # AND
                result = And(result, right)
            elif op in ('!&', '↑'):  # NAND
                result = Not(And(result, right))
            
            i += 2
        
        return result
    
    # Binary operators - OR/NOR/XOR level
    def orlike(self, items):
        """
        Handle OR, NOR, and XOR operations.
        Format: expr (op expr)*
        """
        if len(items) == 1:
            return items[0]
        
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            
            if op in ('|', '∨'):  # OR
                result = Or(result, right)
            elif op in ('!|', '↓'):  # NOR
                result = Not(Or(result, right))
            elif op in ('^', '⊕'):  # XOR
                result = Xor(result, right)
            
            i += 2
        
        return result
    
    # Binary operators - IMPLIES (right associative)
    def implies(self, items):
        """
        Handle IMPLIES operation (right associative).
        a -> b -> c  is  a -> (b -> c)
        """
        if len(items) == 1:
            return items[0]
        
        # Filter out the operator tokens
        exprs = [item for item in items if not isinstance(item, str) 
                 and str(type(item).__name__) != 'Token']
        
        # Actually we need to handle the tokens differently
        # The items include both expressions and operator tokens
        operands = []
        for item in items:
            # Check if it's a Token (operator)
            if hasattr(item, 'type'):
                continue  # Skip operator tokens
            operands.append(item)
        
        if len(operands) == 1:
            return operands[0]
        
        # Right associative: build from right to left
        result = operands[-1]
        for i in range(len(operands) - 2, -1, -1):
            result = Implies(operands[i], result)
        
        return result
    
    # Binary operators - IFF (left associative)
    def iff(self, items):
        """
        Handle IFF (biconditional) operation (left associative).
        a <-> b <-> c  is  (a <-> b) <-> c
        """
        if len(items) == 1:
            return items[0]
        
        # Filter out operator tokens
        operands = []
        for item in items:
            if hasattr(item, 'type'):
                continue
            operands.append(item)
        
        if len(operands) == 1:
            return operands[0]
        
        # Left associative: build from left to right
        result = operands[0]
        for i in range(1, len(operands)):
            # IFF: a <-> b  is equivalent to  (a -> b) AND (b -> a)
            result = And(Implies(result, operands[i]), Implies(operands[i], result))
        
        return result


class PlogParser:
    """
    Main parser class for .plog files.
    
    Usage:
        parser = PlogParser()
        result = parser.parse_file('example.plog')
        # or
        result = parser.parse_string('ATOM x: "desc"')
    """
    
    def __init__(self, grammar_path: str = None):
        """
        Initialize the parser with the plog grammar.
        
        Args:
            grammar_path: Path to the .lark grammar file.
                         If None, looks for 'plog.lark' in current directory.
        """
        if grammar_path is None:
            grammar_path = Path(__file__).parent / 'plog.lark'
        
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        
        self.lark_parser = Lark(grammar, parser='earley', start='start')
    
    def parse_string(self, text: str) -> dict:
        """
        Parse a plog string and return Z3 expressions.
        
        Args:
            text: The plog source code as a string.
            
        Returns:
            Dictionary with 'atoms' and 'formulas' containing Z3 expressions.
        """
        tree = self.lark_parser.parse(text)
        transformer = PlogToZ3()
        return transformer.transform(tree)
    
    def parse_file(self, filepath: str) -> dict:
        """
        Parse a plog file and return Z3 expressions.
        
        Args:
            filepath: Path to the .plog file.
            
        Returns:
            Dictionary with 'atoms' and 'formulas' containing Z3 expressions.
        """
        with open(filepath, 'r') as f:
            return self.parse_string(f.read())


def check_satisfiability(result: dict, additional_constraints=None) -> tuple:
    """
    Check if the formulas in a parsed plog result are satisfiable.
    
    Args:
        result: The result from PlogParser.parse_*()
        additional_constraints: Optional list of additional Z3 constraints.
        
    Returns:
        Tuple of (satisfiability_result, model_or_none)
    """
    solver = Solver()
    
    # Add all formulas as constraints
    for formula_id, expr in result['formulas'].items():
        solver.add(expr)
    
    # Add any additional constraints
    if additional_constraints:
        for constraint in additional_constraints:
            solver.add(constraint)
    
    check_result = solver.check()
    
    if check_result == sat:
        return (sat, solver.model())
    else:
        return (check_result, None)


# --- Main entry point for testing ---

if __name__ == '__main__':
    import sys
    
    # Default to test file if no argument provided
    if len(sys.argv) > 1:
        plog_file = sys.argv[1]
    else:
        plog_file = 'test-case-sat.plog'
    
    print(f"=== Plog Parser Demo ===\n")
    print(f"Parsing file: {plog_file}\n")
    
    # Create parser and parse the file
    parser = PlogParser('plog.lark')
    result = parser.parse_file(plog_file)
    
    # Display atoms
    print("Declared Atoms:")
    print("-" * 40)
    for name, info in result['atoms'].items():
        desc = info['description'] or '(no description)'
        print(f"  {name}: {desc}")
        print(f"    Z3 variable: {info['z3_var']}")
    print()
    
    # Display formulas
    print("Declared Formulas:")
    print("-" * 40)
    for fid, expr in result['formulas'].items():
        print(f"  {fid}:")
        print(f"    Z3 expression: {expr}")
    print()
    
    # Check satisfiability
    print("Satisfiability Check:")
    print("-" * 40)
    sat_result, model = check_satisfiability(result)
    
    if sat_result == sat:
        print("  Result: SAT (Satisfiable)")
        print(f"  Model: {model}")
        print("\n  Variable assignments:")
        for name, info in result['atoms'].items():
            var = info['z3_var']
            value = model.evaluate(var)
            print(f"    {name} = {value}")
    elif sat_result == unsat:
        print("  Result: UNSAT (Unsatisfiable)")
    else:
        print(f"  Result: {sat_result}")
