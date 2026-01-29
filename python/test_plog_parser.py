"""
Comprehensive tests for the Plog parser
"""

from plog_parser import PlogParser, check_satisfiability
from z3 import sat, unsat


def test_basic_satisfiability():
    """Test the original test case - should be SAT"""
    parser = PlogParser('plog.lark')
    result = parser.parse_file('test-case-sat.plog')
    
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, f"Expected SAT, got {sat_result}"
    print("✓ Basic satisfiability test passed (SAT)")


def test_implication():
    """Test implication operator"""
    parser = PlogParser('plog.lark')
    
    # Test: A -> B with A=True, B=True is satisfiable
    # Note: FORMULA_ID requires at least 3 characters
    plog = '''
    ATOM A: "A"
    ATOM B: "B"
    FORMULA impl: A -> B
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == sat, "A -> B should be SAT"
    print("✓ Implication test passed")


def test_and_operator():
    """Test AND operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P & Q
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, "P & Q should be SAT"
    
    # Verify that both must be true
    p_val = model.evaluate(result['atoms']['P']['z3_var'])
    q_val = model.evaluate(result['atoms']['Q']['z3_var'])
    assert str(p_val) == 'True' and str(q_val) == 'True'
    print("✓ AND operator test passed")


def test_or_operator():
    """Test OR operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P | Q
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == sat, "P | Q should be SAT"
    print("✓ OR operator test passed")


def test_not_operator():
    """Test NOT operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    FORMULA f01: ~P
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, "~P should be SAT"
    
    p_val = model.evaluate(result['atoms']['P']['z3_var'])
    assert str(p_val) == 'False', f"P should be False, got {p_val}"
    print("✓ NOT operator test passed")


def test_xor_operator():
    """Test XOR operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P ^ Q
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, "P ^ Q should be SAT"
    
    # XOR means exactly one must be true
    p_val = str(model.evaluate(result['atoms']['P']['z3_var']))
    q_val = str(model.evaluate(result['atoms']['Q']['z3_var']))
    assert (p_val == 'True') != (q_val == 'True'), "XOR: exactly one should be True"
    print("✓ XOR operator test passed")


def test_iff_operator():
    """Test IFF (biconditional) operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P <-> Q
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, "P <-> Q should be SAT"
    
    # IFF means both must have same truth value
    p_val = str(model.evaluate(result['atoms']['P']['z3_var']))
    q_val = str(model.evaluate(result['atoms']['Q']['z3_var']))
    assert p_val == q_val, "IFF: both should have same value"
    print("✓ IFF operator test passed")


def test_nand_operator():
    """Test NAND operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P !& Q
    FORMULA f02: P
    FORMULA f03: Q
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    # P NAND Q with P=True and Q=True should be UNSAT
    assert sat_result == unsat, "P !& Q with P and Q true should be UNSAT"
    print("✓ NAND operator test passed")


def test_nor_operator():
    """Test NOR operator"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P !| Q
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat, "P !| Q should be SAT"
    
    # NOR is true only when both are false
    p_val = str(model.evaluate(result['atoms']['P']['z3_var']))
    q_val = str(model.evaluate(result['atoms']['Q']['z3_var']))
    assert p_val == 'False' and q_val == 'False', "NOR: both should be False"
    print("✓ NOR operator test passed")


def test_constants():
    """Test TRUE and FALSE constants"""
    parser = PlogParser('plog.lark')
    
    # TRUE constant
    plog1 = '''
    ATOM P: "P"
    FORMULA f01: P & TRUE
    '''
    result = parser.parse_string(plog1)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat
    p_val = str(model.evaluate(result['atoms']['P']['z3_var']))
    assert p_val == 'True', "P & TRUE should require P=True"
    
    # FALSE constant makes formula UNSAT
    plog2 = '''
    ATOM P: "P"
    FORMULA f01: P & FALSE
    '''
    result = parser.parse_string(plog2)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == unsat, "P & FALSE should be UNSAT"
    
    print("✓ Constants (TRUE/FALSE) test passed")


def test_complex_expression():
    """Test complex nested expressions"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM A: "A"
    ATOM B: "B"
    ATOM C: "C"
    FORMULA f01: (A & B) -> C
    FORMULA f02: A
    FORMULA f03: B
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat
    
    # If A=True and B=True, then C must be True
    c_val = str(model.evaluate(result['atoms']['C']['z3_var']))
    assert c_val == 'True', "C should be True given A & B"
    print("✓ Complex expression test passed")


def test_unsatisfiable():
    """Test an unsatisfiable formula"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    FORMULA f01: P & ~P
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == unsat, "P & ~P should be UNSAT"
    print("✓ Unsatisfiable formula test passed")


def test_unicode_operators():
    """Test Unicode operator symbols"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    ATOM P: "P"
    ATOM Q: "Q"
    FORMULA f01: P ∧ Q
    FORMULA f02: P ∨ Q
    FORMULA f03: ¬P → Q
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == sat, "Unicode operators should work"
    print("✓ Unicode operators test passed")


def test_right_associative_implies():
    """Test that implies is right-associative"""
    parser = PlogParser('plog.lark')
    
    # A -> B -> C should be A -> (B -> C)
    # This is SAT when A=True, B=True, C=True
    plog = '''
    ATOM A: "A"
    ATOM B: "B" 
    ATOM C: "C"
    FORMULA f01: A -> B -> C
    FORMULA f02: A
    FORMULA f03: B
    '''
    result = parser.parse_string(plog)
    sat_result, model = check_satisfiability(result)
    assert sat_result == sat
    
    c_val = str(model.evaluate(result['atoms']['C']['z3_var']))
    assert c_val == 'True', "C must be True for A -> (B -> C) with A,B True"
    print("✓ Right-associative implies test passed")


def test_comments():
    """Test that comments are properly ignored"""
    parser = PlogParser('plog.lark')
    
    plog = '''
    # This is a comment
    ATOM P: "P"  # Inline comment
    # Another comment
    FORMULA f01: P
    '''
    result = parser.parse_string(plog)
    sat_result, _ = check_satisfiability(result)
    assert sat_result == sat
    print("✓ Comments test passed")


if __name__ == '__main__':
    print("=" * 50)
    print("Running Plog Parser Tests")
    print("=" * 50)
    print()
    
    test_basic_satisfiability()
    test_implication()
    test_and_operator()
    test_or_operator()
    test_not_operator()
    test_xor_operator()
    test_iff_operator()
    test_nand_operator()
    test_nor_operator()
    test_constants()
    test_complex_expression()
    test_unsatisfiable()
    test_unicode_operators()
    test_right_associative_implies()
    test_comments()
    
    print()
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
