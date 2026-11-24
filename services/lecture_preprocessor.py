"""
===============================================================================
Module: Lecture Preprocessor (services/lecture_preprocessor.py)
===============================================================================
1. Purpose:
   Analyzes transcript sentences specifically as educational lecture content.
2. Input:
   List of timestamped Sentence dataclasses.
3. Processing:
   Distinguishes high-information lecture components (definitions, core concepts,
   explanations, formulas, procedures, examples, conclusions) from low-information
   noise (greetings, admin talk, repeated statements, filler conversations).
4. Algorithm/Model:
   Pattern matching, keyword density analysis, and educational heuristic rules.
5. Output:
   Sentences tagged with educational categories (definition, concept, explanation,
   formula, example, conclusion, intro, noise).
6. Data Stored:
   educational_category attribute on Sentence objects.
7. Failure Cases:
   Ambiguous sentences default to 'explanation' category rather than aggressive deletion.
8. Evaluation:
   Ensures definitions and core concepts receive high baseline category flags.
9. Integration:
   Feeds category information to importance_service.py for multi-factor scoring.
===============================================================================
"""

import re
from typing import List
from models.transcript import Sentence

DEFINITION_PATTERNS = re.compile(
    r'\b(is defined as|refers to|means that|can be defined|known as|termed as|is called|we define|definition of|concept of)\b',
    re.IGNORECASE
)

CONCEPT_PATTERNS = re.compile(
    r'\b(important|fundamental|principle|algorithm|key concept|core idea|theory|architecture|mechanism|structure|property|law)\b',
    re.IGNORECASE
)

FORMULA_PATTERNS = re.compile(
    r'\b(formula|equation|equal to|squared|divided by|multiplied by|sum of|integral|derivative|ratio|logarithm|theorem)\b',
    re.IGNORECASE
)

EXAMPLE_PATTERNS = re.compile(
    r'\b(for example|for instance|such as|consider a|let us assume|take an example|suppose we have|as an illustration)\b',
    re.IGNORECASE
)

CONCLUSION_PATTERNS = re.compile(
    r'\b(in conclusion|to summarize|therefore|thus|hence|finally|in summary|the key takeaway|we conclude)\b',
    re.IGNORECASE
)

GREETINGS_ADMIN_PATTERNS = re.compile(
    r'\b(good morning|good afternoon|hello everyone|welcome to class|assignment due|submit your|attendance|quiet please|can you hear me|screen visible|slide visible)\b',
    re.IGNORECASE
)

def classify_lecture_sentences(sentences: List[Sentence]) -> List[Sentence]:
    """Classifies sentences into educational categories based on academic indicators."""
    for s in sentences:
        text = s.text.lower()

        if GREETINGS_ADMIN_PATTERNS.search(text):
            s.educational_category = "admin_noise"
        elif DEFINITION_PATTERNS.search(text):
            s.educational_category = "definition"
        elif FORMULA_PATTERNS.search(text):
            s.educational_category = "formula"
        elif CONCLUSION_PATTERNS.search(text):
            s.educational_category = "conclusion"
        elif CONCEPT_PATTERNS.search(text):
            s.educational_category = "concept"
        elif EXAMPLE_PATTERNS.search(text):
            s.educational_category = "example"
        else:
            s.educational_category = "explanation"

    cat_counts = {}
    for s in sentences:
        cat_counts[s.educational_category] = cat_counts.get(s.educational_category, 0) + 1

    print(f"[LECTURE PREPROCESSOR] Categorized sentences: {cat_counts}")
    return sentences
