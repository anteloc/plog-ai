# TASK DEFINITION

Translate the attached document specified by the user into propositional logic statements in .plog format for further text analysis by other software tooling.

Do not output the thinking process to the user, just:
* The current step you are working on
* If the analysis process has been successful or not
* The end result of it, as detailed on STEP 6 and 7

# STEPS

## STEP 1 - PREPARATION

Analyze **in detail** the following attachments:
- `plog-grammar.txt`: lark grammar definition, defines the .plog format
- the attached document provided by the user


## STEP 2 - PROPOSITIONAL LOGIC TRANSLATION

Translate the contents in the document attached by the user to propositional logic.

* DO's:
    * Use the symbol set defined in `plog-grammar.txt`
    * Use python-style naming for variables and formulas, e.g. some_atom_name, f_some_formula
    * Give meaningful names to atoms, e.g. plato_is_a_man
    * Give meaningful descriptions to atoms, e.g. "Plato is a man"
    * Atoms are assumed to be TRUE facts, even when the statement is in negative form
    * Statements in negative form like e.g. "Plato is not an animal" should be converted into atoms like
        * ATOM plato_is_not_an_animal: "Plato is not an animal"
    * Same with statements with negative-like elements, e.g.:
        * "A man without a house": should become: 
            * ATOM man_without_house: "A man without a house" 
        * "Some missing pieces": should become
            * ATOM some_missing_pieces: "Some missing pieces"
        * Etc.
    * Formulae names should start with "f_"
    * Formulae should only contain atoms and operations between them
    * Formulae should be as faithful as possible to the contents on the original text
    * Formulae should capture in detail the main aspects of the text
    * Detect logical fallacies in the given text and tag them:
        * Prefix fallacy atoms with: FALLACY_
        * Prefix fallacy formulas with: f_FALLACY_
    
Fallacies tagging example:

```
ATOM FALLACY_appeal_to_popularity: "FALLACY: Appeal to popularity (argumentum ad populum) - 'Everyone knows' does not establish truth"

# FALLACY: Appeal to Popularity (Argumentum ad Populum)
# "Everyone knows X" is used to assert X is true
FORMULA f_appeal_to_popularity: everyone_knows_plan_will_bankrupt_town -> plan_will_bankrupt_town
FORMULA f_FALLACY_appeal_to_popularity: everyone_knows_plan_will_bankrupt_town -> FALLACY_appeal_to_popularity
```

* DONT's
    * Negate atoms in the consequent:
        * WRONG: an atom representing a negative statement
            * ATOM plato_is_animal: "Plato is an animal"
            * ATOM plato_is_man: "Plato is a man"
            * FORMULA f_if_man_not_animal: plato_is_man -> ~ plato_is_animal
    * Create facts formulae for positive statements, e.g. "Plato is a man":
        * WRONG:
            * ATOM: plato_is_a_man: "Plato is a man"
            * FORMULA: f_fact_plato_is_a_man: plato_is_a_man
    * Reference a formula(s) from another formula


## STEP 3 - PLOG FORMAT CONVERSION

Convert the result from previous step into a `logic-results.plog` file, according to `plog-grammar.txt` grammar definition.

Add one-line "#..." comments to split into sections, according to the different matters discussed in the text.

## STEP 4 - END RESULT VALIDATION

Install the required dependencies for running the attached validator script: `plog.py`

```shell
python3 -m pip install lark
python3 -m pip install z3-solver
```

Run the validator script `plog.py` on the `logic-results.plog` from previous step with the following options:

```shell
python3 plog.py --grammar ./plog-grammar.txt logic-results.plog
```

Validation results:
- **PASSED**: for exit codes 0 or 1
- **FAILED**: for exit code 2

## STEP 5 - ITERATIVE CORRECTION

If validation from the previous step fails:

1. Review the text-to-propositional logic translation
2. Review the propositional logic-to-plog translation
3. If parsing error reported, review and correct `logic-results.plog`
4. Go back to the previous step and re-run the validator script

## STEP 6 - DELIVERABLES

Once the `logic-results.plog` file PASSESS the validation, make available for the user to download:

* The output from plog.py for the validation execution in a `logic-analysis.md` file
* The `logic-results.plog` file

## STEP 7 - RESULTS INTERPRETATION

Using plain English, provide an interpretation to the user for the output results.

If no inconsistencies or fallacies, output: "Text shows no inconsistencies"

If inconsistencies or fallacies detected, show the user a summarized explanation about:

* What things are inconsistent in the text
* Reason(s) why each one of those are inconsistent

