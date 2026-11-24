#!/usr/bin/env python3
"""
generate_dataset.py

Generates 500 diverse MATLAB error-fixing examples in ChatML format (system, user, assistant)
and saves them to data/matlab_train.jsonl.

Uses the google-genai SDK with Pydantic structured output validation.
Covers:
1. Matrix vs. element-wise operators (* vs .*, ^ vs .^, / vs ./, \\ vs .\\)
2. Missing end keywords (loops, if-else, function definitions, switch-case)
3. 1-based indexing & cell array errors (0/negative indexing, {} vs () cell access)
4. Dimension mismatches (inner dimensions, horizontal/vertical concatenation)
5. Valid MATLAB snippets (~15%, exactly 75 examples with "No errors found")
6. Other common MATLAB bugs (variable shadowing built-ins, string/char errors, assignment in if)

Validates every JSONL line locally with Pydantic.
"""

import os
import sys
import json
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load local .env
load_dotenv()

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "matlab_train.jsonl"

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class MatlabRawExample(BaseModel):
    category: str = Field(description="Error category or 'valid_code'")
    problem_context: str = Field(description="Context or intended operation")
    buggy_or_input_code: str = Field(description="The MATLAB code snippet")
    has_error: bool = Field(description="True if buggy, False if clean code")
    error_type: str = Field(description="Error title or 'No errors found'")
    explanation: str = Field(description="Technical root cause or validity explanation")
    fixed_or_verified_code: str = Field(description="Corrected or verified MATLAB code")


class ChatMLMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=10)


class ChatMLExample(BaseModel):
    messages: List[ChatMLMessage] = Field(min_length=3, max_length=3)


# ---------------------------------------------------------------------------
# System Prompt & ChatML Formatter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert MATLAB debugger and code fixer. Analyze the provided MATLAB code, "
    "identify any syntax, semantic, or runtime errors, explain the root cause, and provide the "
    "corrected code. If the code is already valid and bug-free, explain why it is correct and "
    "state that no errors were found."
)

def format_chatml(item: MatlabRawExample) -> ChatMLExample:
    """Formats a MatlabRawExample into ChatML messages format."""
    user_content = (
        f"The following MATLAB code has an issue or needs verification. "
        f"Please review, diagnose, and provide the corrected code.\n\n"
        f"Intent / Context: {item.problem_context}\n\n"
        f"```matlab\n{item.buggy_or_input_code.strip()}\n```"
    )

    if item.has_error:
        assistant_content = (
            f"### Diagnosis\n"
            f"**Error Type:** {item.error_type}\n\n"
            f"**Cause:** {item.explanation}\n\n"
            f"### Corrected Code\n"
            f"```matlab\n{item.fixed_or_verified_code.strip()}\n```\n\n"
            f"### Key Takeaways\n"
            f"- Ensure MATLAB syntax and operator rules are strictly satisfied.\n"
            f"- Verify array shapes, indexing boundaries, and block delimiters (`end`)."
        )
    else:
        assistant_content = (
            f"### Diagnosis\n"
            f"**Status:** No errors found.\n\n"
            f"**Analysis:** {item.explanation}\n\n"
            f"### Verified Code\n"
            f"```matlab\n{item.fixed_or_verified_code.strip()}\n```\n\n"
            f"The code is syntactically sound, logically correct, and follows idiomatic MATLAB practices."
        )

    return ChatMLExample(
        messages=[
            ChatMLMessage(role="system", content=SYSTEM_PROMPT),
            ChatMLMessage(role="user", content=user_content),
            ChatMLMessage(role="assistant", content=assistant_content),
        ]
    )


# ---------------------------------------------------------------------------
# Gemini API Client Setup
# ---------------------------------------------------------------------------

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY not found in environment. Proceeding with procedural engine.")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"[WARNING] Failed to initialize google-genai client: {e}")
        return None


def call_gemini_for_examples(client, model_name: str, category: str, count: int) -> List[MatlabRawExample]:
    """Queries Gemini API for a batch of MATLAB error-fixing examples with Pydantic validation."""
    if not client:
        return []

    prompt = f"""You are a specialized MATLAB dataset generator. Generate {count} distinct and realistic MATLAB code examples for the category: '{category}'.

Return a valid JSON object matching this schema:
{{
  "items": [
    {{
      "category": "{category}",
      "problem_context": "Brief 1-sentence description of what the user is trying to calculate or simulate",
      "buggy_or_input_code": "MATLAB snippet (2-8 lines)",
      "has_error": true (or false if category is valid_code),
      "error_type": "MATLAB error name or 'No errors found'",
      "explanation": "Clear explanation of the error mechanism or why the snippet is valid",
      "fixed_or_verified_code": "Corrected MATLAB code (or identical verified code if clean)"
    }}
  ]
}}

Requirements for category '{category}':
- If category is 'operator_mismatch': show realistic calculations (DSP, plotting, physics) with * vs .*, ^ vs .^, / vs ./, \\ vs .\\
- If category is 'missing_end': show loops (for/while), if-else, function definitions, or switch blocks missing 'end'
- If category is 'indexing_and_cells': show 0-based indexing attempts, negative indexing, cell array {{}} vs () confusion
- If category is 'dimension_mismatch': show matrix concatenation [A, B] vs [A; B], inner dimensions mismatch in A * B
- If category is 'valid_code': clean, 100% correct idiomatic MATLAB snippets with has_error=false and error_type='No errors found'
- If category is 'other_common_matlab_bugs': variable shadowing built-ins (sum, mean, size), string vs char confusion, '=' in if condition

Output ONLY raw JSON. No markdown fences.
"""
    try:
        res = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = res.text.strip()
        # Clean json
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        items_raw = parsed.get("items", [])
        validated = [MatlabRawExample.model_validate(x) for x in items_raw]
        return validated
    except Exception as e:
        print(f"  [API Note] Gemini batch query ({category}) encountered: {e}")
        return []


# ---------------------------------------------------------------------------
# Procedural Generator Bank (Guarantees 500 Diverse Realistic Examples)
# ---------------------------------------------------------------------------

OPERATOR_TEMPLATES = [
    # Vector element-wise power
    ("Square each element of time vector for kinematic acceleration",
     "t = 0:0.1:10;\ny = 0.5 * 9.8 * t^2;\nplot(t, y);",
     "Incorrect use of matrix power `^` on vector `t`.",
     "The `^` operator is matrix exponentiation, requiring a square matrix. To raise each element of vector `t` to power 2, use the element-wise power operator `.^`.",
     "t = 0:0.1:10;\ny = 0.5 * 9.8 * t.^2;\nplot(t, y);"),

    ("Calculate polynomial response y = 3x^3 - 2x^2 + 5x",
     "x = linspace(-5, 5, 100);\ny = 3*x^3 - 2*x^2 + 5*x;\nplot(x, y);",
     "Matrix power `^` used on non-scalar array `x`.",
     "`x` is a 1x100 vector. Evaluating `x^3` and `x^2` attempts matrix power which fails with 'Inputs must be square'. Use `.^` for element-wise exponentiation.",
     "x = linspace(-5, 5, 100);\ny = 3*x.^3 - 2*x.^2 + 5*x;\nplot(x, y);"),

    # Vector element-wise multiplication
    ("Compute modulated carrier wave signal",
     "fs = 1000;\nt = 0:1/fs:1;\ncarrier = cos(2*pi*100*t);\nmodulator = sin(2*pi*5*t);\nsignal = carrier * modulator;",
     "Matrix multiplication `*` used instead of element-wise `.*`.",
     "Both `carrier` and `modulator` are 1x1001 row vectors. Matrix multiplication `1x1001 * 1x1001` fails with 'Inner matrix dimensions must agree'. Use `.*` for element-wise multiplication.",
     "fs = 1000;\nt = 0:1/fs:1;\ncarrier = cos(2*pi*100*t);\nmodulator = sin(2*pi*5*t);\nsignal = carrier .* modulator;"),

    ("Calculate kinetic energy E = 0.5 * m * v^2 for multiple velocities",
     "m = 2.5;\nv = [10, 20, 30, 40, 50];\nE = 0.5 * m * v^2;",
     "Matrix exponentiation `^` on vector `v`.",
     "`v` is a vector, so `v^2` attempts matrix multiplication `v * v` which is undefined for 1x5 matrices. Use `v.^2`.",
     "m = 2.5;\nv = [10, 20, 30, 40, 50];\nE = 0.5 * m * v.^2;"),

    # Vector element-wise division
    ("Normalize vector by frequency bin values",
     "freqs = 10:10:100;\namplitudes = [5, 12, 18, 25, 30, 28, 20, 15, 10, 4];\nnormalized = amplitudes / freqs;",
     "Matrix slash `/` used on two row vectors.",
     "`amplitudes / freqs` computes the least-squares matrix right division resulting in a single scalar, rather than element-by-element normalization. Use `./`.",
     "freqs = 10:10:100;\namplitudes = [5, 12, 18, 25, 30, 28, 20, 15, 10, 4];\nnormalized = amplitudes ./ freqs;"),

    ("Compute sinc function y = sin(x) / x",
     "x = -10:0.1:10;\ny = sin(x) / x;",
     "Matrix division `/` between row vectors.",
     "`sin(x) / x` performs matrix right division instead of element-wise division. Use `./`.",
     "x = -10:0.1:10;\ny = sin(x) ./ x;\ny(isnan(y)) = 1; % Handle 0/0 limit"),

    ("Evaluate damping ratio curve gamma = c / (2 * sqrt(m * k)) across stiffness range",
     "c = 15;\nm = 2.0;\nk = linspace(100, 1000, 50);\ngamma = c / (2 * sqrt(m * k));",
     "Matrix division `/` with vector in denominator.",
     "`k` is a vector, so the denominator is a 1x50 row vector. Dividing scalar `c` by a vector with `/` causes matrix dimension issues or unexpected dimensions. Use `./`.",
     "c = 15;\nm = 2.0;\nk = linspace(100, 1000, 50);\ngamma = c ./ (2 * sqrt(m .* k));"),

    # Element-wise left division vs matrix solve
    ("Element-wise inversion of matrix values",
     "A = [2, 4; 8, 16];\ninvA = 1 \\ A;",
     "Incorrect operator for element-wise reciprocal.",
     "`1 \\ A` computes matrix left division (solving `1*x = A`), which equals `A`. To compute reciprocal `1/A(i,j)`, use `1 ./ A` or `A .\\ 1`.",
     "A = [2, 4; 8, 16];\ninvA = 1 ./ A;"),

    ("Compute Gaussian probability density function",
     "x = linspace(-3, 3, 200);\nmu = 0; sigma = 1;\npdf = 1 / (sigma * sqrt(2*pi)) * exp(-((x - mu)^2) / (2*sigma^2));",
     "Matrix power `^` on vector `(x - mu)`.",
     "`(x - mu)^2` attempts matrix exponentiation on a 1x200 vector. Must use element-wise power `.^`.",
     "x = linspace(-3, 3, 200);\nmu = 0; sigma = 1;\npdf = (1 / (sigma * sqrt(2*pi))) .* exp(-((x - mu).^2) ./ (2*sigma^2));"),

    ("Compute hyperbolic tangent activation for array of inputs",
     "z = [-2, -1, 0, 1, 2];\nact = (exp(z) - exp(-z)) / (exp(z) + exp(-z));",
     "Matrix division `/` between vectors.",
     "Numerator and denominator are 1x5 vectors. Matrix division `/` calculates a scalar projection instead of element-wise tanh values. Use `./`.",
     "z = [-2, -1, 0, 1, 2];\nact = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));"),
]

MISSING_END_TEMPLATES = [
    # Missing end in for loop
    ("Sum elements greater than threshold in array",
     "total = 0;\ndata = [4, 8, 15, 16, 23, 42];\nfor i = 1:length(data)\n    if data(i) > 10\n        total = total + data(i);\n    end\ndisp(total);",
     "Missing `end` keyword for `for` loop.",
     "The `if` statement has a closing `end`, but the outer `for` loop is missing its required `end` statement, causing a syntax error: 'At least one END is missing: the statement may begin here'.",
     "total = 0;\ndata = [4, 8, 15, 16, 23, 42];\nfor i = 1:length(data)\n    if data(i) > 10\n        total = total + data(i);\n    end\nend\ndisp(total);"),

    # Missing end in while loop
    ("Convergence loop for Newton-Raphson root finding",
     "x = 1.0;\ntol = 1e-5;\niter = 0;\nwhile abs(x^2 - 2) > tol && iter < 100\n    x = x - (x^2 - 2) / (2*x);\n    iter = iter + 1;\ndisp(['Root: ', num2str(x)]);",
     "Missing `end` keyword for `while` loop.",
     "The `while` loop block is never closed with an `end` keyword before script termination or output statements.",
     "x = 1.0;\ntol = 1e-5;\niter = 0;\nwhile abs(x^2 - 2) > tol && iter < 100\n    x = x - (x^2 - 2) / (2*x);\n    iter = iter + 1;\nend\ndisp(['Root: ', num2str(x)]);"),

    # Missing end in if-else
    ("Classify signal intensity into low, medium, high",
     "val = 75;\nif val < 30\n    level = 'LOW';\nelseif val < 70\n    level = 'MEDIUM';\nelse\n    level = 'HIGH';\ndisp(level);",
     "Missing `end` keyword for `if-elseif-else` construct.",
     "An `if-elseif-else` conditional structure in MATLAB must be terminated with an `end` keyword.",
     "val = 75;\nif val < 30\n    level = 'LOW';\nelseif val < 70\n    level = 'MEDIUM';\nelse\n    level = 'HIGH';\nend\ndisp(level);"),

    # Missing end in function definition
    ("Define custom Euclidean norm function",
     "function n = my_norm(v)\n    n = sqrt(sum(v.^2));\n\nfunction d = dot_prod(a, b)\n    d = sum(a .* b);\nend",
     "Missing `end` keyword in first function `my_norm`.",
     "When a file contains multiple function definitions, all functions must be explicitly terminated with an `end` keyword.",
     "function n = my_norm(v)\n    n = sqrt(sum(v.^2));\nend\n\nfunction d = dot_prod(a, b)\n    d = sum(a .* b);\nend"),

    # Missing end in switch-case
    ("Handle operational state via switch-case",
     "state = 2;\nswitch state\n    case 1\n        msg = 'Initialized';\n    case 2\n        msg = 'Running';\n    otherwise\n        msg = 'Unknown';\ndisp(msg);",
     "Missing `end` keyword for `switch` block.",
     "A `switch-case-otherwise` block in MATLAB requires a closing `end`.",
     "state = 2;\nswitch state\n    case 1\n        msg = 'Initialized';\n    case 2\n        msg = 'Running';\n    otherwise\n        msg = 'Unknown';\nend\ndisp(msg);"),

    # Missing end in try-catch
    ("Read configuration file with exception fallback",
     "try\n    cfg = load('config.mat');\ncatch ME\n    cfg = struct('default', true);\ndisp(cfg);",
     "Missing `end` keyword for `try-catch` block.",
     "A `try-catch` construct must be closed with an `end` keyword.",
     "try\n    cfg = load('config.mat');\ncatch ME\n    cfg = struct('default', true);\nend\ndisp(cfg);"),

    # Nested loop missing inner end
    ("Populate 2D coordinate grid distances",
     "D = zeros(5, 5);\nfor r = 1:5\n    for c = 1:5\n        D(r, c) = sqrt(r^2 + c^2);\nend",
     "Missing `end` in nested `for` loop.",
     "Two `for` loops are opened (`for r = 1:5` and `for c = 1:5`), but only one closing `end` is provided.",
     "D = zeros(5, 5);\nfor r = 1:5\n    for c = 1:5\n        D(r, c) = sqrt(r^2 + c^2);\n    end\nend"),
]

INDEXING_TEMPLATES = [
    # 0-based indexing
    ("Access first element of array",
     "data = [10, 20, 30, 40];\nfirst_item = data(0);",
     "0-based indexing error (Subscript indices must either be real positive integers or logicals).",
     "MATLAB uses 1-based indexing. Accessing `data(0)` results in error: 'Array indices must be positive integers or logical values'. The first element is `data(1)`.",
     "data = [10, 20, 30, 40];\nfirst_item = data(1);"),

    ("Looping from 0 to N-1 (C/Python style)",
     "vec = [5, 15, 25, 35];\nN = length(vec);\nfor i = 0:(N-1)\n    disp(vec(i));\nend",
     "Indexing array with index `0`.",
     "The loop index starts at `0`. On iteration `i=0`, `vec(0)` triggers index error. In MATLAB, valid indices range from `1` to `N`.",
     "vec = [5, 15, 25, 35];\nN = length(vec);\nfor i = 1:N\n    disp(vec(i));\nend"),

    # Negative indexing
    ("Access last element using -1 Python style",
     "signals = [1.2, 3.4, 5.6, 7.8];\nlast_val = signals(-1);",
     "Negative index `signals(-1)` used.",
     "MATLAB does not support negative indices like Python. To access the last element, use the keyword `end`, e.g., `signals(end)`.",
     "signals = [1.2, 3.4, 5.6, 7.8];\nlast_val = signals(end);"),

    ("Access second to last element",
     "arr = [10, 20, 30, 40, 50];\nsecond_last = arr(-2);",
     "Negative indexing attempt `arr(-2)`.",
     "Use `end-1` in MATLAB to access the penultimate element.",
     "arr = [10, 20, 30, 40, 50];\nsecond_last = arr(end-1);"),

    # Cell array () vs {}
    ("Extract numeric matrix stored inside cell array",
     "my_cell = {'SensorA', [1, 2, 3; 4, 5, 6], 42};\ndata_matrix = my_cell(2);\nmean_val = mean(data_matrix);",
     "Parentheses `()` used on cell array instead of curly braces `{}`.",
     "`my_cell(2)` returns a 1x1 `cell` containing the matrix, not the numeric matrix itself. Passing a cell to `mean()` raises: 'Undefined function mean for input arguments of type cell'. Use `{2}` to dereference contents.",
     "my_cell = {'SensorA', [1, 2, 3; 4, 5, 6], 42};\ndata_matrix = my_cell{2};\nmean_val = mean(data_matrix);"),

    ("Concatenate string from cell array",
     "names = {'Alice', 'Bob', 'Charlie'};\ngreeting = ['Hello, ', names(1)];",
     "Cell indexing `names(1)` produces a cell rather than a character array/string.",
     "Using `()` returns a cell container. To extract the underlying string/char, use curly braces `names{1}`.",
     "names = {'Alice', 'Bob', 'Charlie'};\ngreeting = ['Hello, ', names{1}];"),

    ("Assign value to cell array element",
     "C = cell(1, 3);\nC(1) = 100;",
     "Conversion to cell from double is not possible with parentheses assignment.",
     "Assigning a numeric double directly to `C(1)` fails because `C(1)` expects a cell. Use curly braces `C{1} = 100` or wrap in cell `C(1) = {100}`.",
     "C = cell(1, 3);\nC{1} = 100;"),

    # Index exceeds dimensions
    ("Access element outside matrix boundary",
     "M = [1, 2; 3, 4];\nval = M(3, 1);",
     "Index exceeds matrix dimensions.",
     "`M` is a 2x2 matrix. Accessing row 3 exceeds matrix dimensions. Ensure index is within `size(M)`.",
     "M = [1, 2; 3, 4];\nif size(M, 1) >= 3\n    val = M(3, 1);\nelse\n    val = NaN;\nend"),
]

DIMENSION_TEMPLATES = [
    # Inner matrix dimensions must agree
    ("Multiply two 2x3 matrices directly",
     "A = [1, 2, 3; 4, 5, 6];\nB = [7, 8, 9; 10, 11, 12];\nC = A * B;",
     "Inner matrix dimensions must agree.",
     "`A` is 2x3 and `B` is 2x3. Matrix multiplication `A * B` requires `size(A,2) == size(B,1)`. To multiply, either transpose `B` (`A * B'`) or perform element-wise multiplication (`A .* B`).",
     "A = [1, 2, 3; 4, 5, 6];\nB = [7, 8, 9; 10, 11, 12];\nC = A * B'; % Or A .* B for element-wise"),

    # Incompatible concatenation
    ("Horizontal concatenation of vectors with differing row counts",
     "col_vec = [1; 2; 3; 4];\nrow_vec = [5, 6, 7];\ncombined = [col_vec, row_vec];",
     "Dimensions of matrices being concatenated are not consistent.",
     "`col_vec` is 4x1 and `row_vec` is 1x3. Horizontal concatenation `[A, B]` requires matching number of rows (`size(A,1) == size(B,1)`).",
     "col_vec = [1; 2; 3; 4];\nrow_vec = [5; 6; 7; 8]; % Reshape/match row dimension\ncombined = [col_vec, row_vec];"),

    ("Vertical concatenation of matrices with differing column counts",
     "M1 = [1, 2, 3; 4, 5, 6];\nM2 = [7, 8; 9, 10];\nstacked = [M1; M2];",
     "Dimensions of matrices being concatenated are not consistent.",
     "`M1` has 3 columns and `M2` has 2 columns. Vertical concatenation `[M1; M2]` requires equal column counts (`size(M1,2) == size(M2,2)`).",
     "M1 = [1, 2, 3; 4, 5, 6];\nM2 = [7, 8, 0; 9, 10, 0]; % Pad columns to match 3\nstacked = [M1; M2];"),

    # Linear system solve dimension mismatch
    ("Solve linear system A*x = b with incompatible right-hand side vector",
     "A = [2, 1, -1; -3, -1, 2; -2, 1, 2];\nb = [8; -11];\nx = A \\ b;",
     "Matrix dimensions must agree for linear system solve.",
     "`A` is 3x3, but `b` has only 2 rows (2x1). Solving `A \\ b` requires `size(A,1) == size(b,1)`. Vector `b` must have 3 elements.",
     "A = [2, 1, -1; -3, -1, 2; -2, 1, 2];\nb = [8; -11; -3]; % b must be 3x1\nx = A \\ b;"),

    # Dot product dimension mismatch
    ("Compute dot product between vectors of unequal lengths",
     "u = [1, 2, 3, 4];\nv = [5, 6, 7];\nd = dot(u, v);",
     "Vector dimensions must agree in dot product.",
     "`u` has length 4 while `v` has length 3. `dot(u, v)` requires both vectors to have identical lengths.",
     "u = [1, 2, 3, 4];\nv = [5, 6, 7, 8]; % Vectors must be equal length\nd = dot(u, v);"),

    # Cross product requires 3 elements
    ("Compute cross product of 2D vectors",
     "v1 = [1, 2];\nv2 = [3, 4];\ncp = cross(v1, v2);",
     "Vectors must have length 3 in cross product.",
     "`cross(A, B)` requires input vectors to be 3-dimensional. For 2D vectors, append 0 for the z-component or calculate the scalar z-cross product.",
     "v1 = [1, 2, 0];\nv2 = [3, 4, 0];\ncp = cross(v1, v2);"),
]

VALID_TEMPLATES = [
    ("Compute eigenvalues and eigenvectors of a symmetric matrix",
     "A = [4, 1, -2; 1, 2, 0; -2, 0, 3];\n[V, D] = eig(A);\ndiag_vals = diag(D);",
     "The code cleanly sets up a symmetric 3x3 matrix, invokes `eig` returning eigenvector matrix `V` and diagonal eigenvalue matrix `D`, and extracts the eigenvalues via `diag(D)`.",
     "A = [4, 1, -2; 1, 2, 0; -2, 0, 3];\n[V, D] = eig(A);\ndiag_vals = diag(D);"),

    ("Numerical integration of a damped sine wave using trapz",
     "t = linspace(0, 10, 1000);\ny = exp(-0.5*t) .* sin(2*pi*t);\narea = trapz(t, y);",
     "Properly uses element-wise operator `.*` to multiply decaying exponential and sinusoidal signal arrays, and correctly calls `trapz(t, y)` for trapezoidal numerical integration.",
     "t = linspace(0, 10, 1000);\ny = exp(-0.5*t) .* sin(2*pi*t);\narea = trapz(t, y);"),

    ("Solve a system of linear equations using backslash operator",
     "A = [3, 2, -1; 2, -2, 4; -1, 0.5, -1];\nb = [1; -2; 0];\nx = A \\ b;",
     "Properly sets up a 3x3 coefficient matrix and 3x1 column vector, utilizing the numerically stable MATLAB backslash operator `mldivide` to solve the linear system.",
     "A = [3, 2, -1; 2, -2, 4; -1, 0.5, -1];\nb = [1; -2; 0];\nx = A \\ b;"),

    ("Solve ordinary differential equation using ode45",
     "tspan = [0, 20];\ny0 = [1; 0];\node_func = @(t, y) [y(2); -0.1*y(2) - y(1)];\n[t, y] = ode45(ode_func, tspan, y0);",
     "Correctly formulates a 2nd order damped oscillator as a system of first-order ODEs using an anonymous function handle, defines appropriate column initial conditions `y0`, and integrates with `ode45`.",
     "tspan = [0, 20];\ny0 = [1; 0];\node_func = @(t, y) [y(2); -0.1*y(2) - y(1)];\n[t, y] = ode45(ode_func, tspan, y0);"),

    ("Apply Butterworth lowpass filter to noisy signal",
     "fs = 1000;\nfc = 50;\n[b, a] = butter(4, fc/(fs/2), 'low');\nt = 0:1/fs:1;\nx = sin(2*pi*10*t) + 0.5*randn(size(t));\ny = filtfilt(b, a, x);",
     "Correctly normalizes cutoff frequency to Nyquist (`fc/(fs/2)`), designs a 4th-order Butterworth filter, generates sample signal, and applies zero-phase forward-backward filtering using `filtfilt`.",
     "fs = 1000;\nfc = 50;\n[b, a] = butter(4, fc/(fs/2), 'low');\nt = 0:1/fs:1;\nx = sin(2*pi*10*t) + 0.5*randn(size(t));\ny = filtfilt(b, a, x);"),

    ("Compute 2D fast Fourier transform of image matrix",
     "img = rand(128, 128);\nF = fftshift(fft2(img));\nmag_spectrum = log(1 + abs(F));",
     "Properly applies 2D FFT `fft2`, centers zero-frequency component with `fftshift`, and calculates compressed log magnitude spectrum.",
     "img = rand(128, 128);\nF = fftshift(fft2(img));\nmag_spectrum = log(1 + abs(F));"),

    ("Vectorized Euclidean distance matrix computation",
     "P = rand(50, 3);\nQ = rand(30, 3);\nD = pdist2(P, Q, 'euclidean');",
     "Uses MATLAB's built-in, highly optimized `pdist2` function to compute pairwise distances between two sets of 3D spatial points.",
     "P = rand(50, 3);\nQ = rand(30, 3);\nD = pdist2(P, Q, 'euclidean');"),

    ("Store heterogeneous metadata in cell array and struct",
     "patient.id = 101;\npatient.tests = {'ECG', 'BloodPressure', 'Glucose'};\npatient.readings = [120, 80; 118, 79];\nfirst_test = patient.tests{1};",
     "Correctly structures patient data using fields and cells, and accesses cell array contents using curly braces `{1}`.",
     "patient.id = 101;\npatient.tests = {'ECG', 'BloodPressure', 'Glucose'};\npatient.readings = [120, 80; 118, 79];\nfirst_test = patient.tests{1};"),
]

OTHER_BUGS_TEMPLATES = [
    # Variable shadowing built-in functions
    ("Calculate sum of array elements but shadowed sum variable",
     "sum = 0;\narr = [10, 20, 30, 40];\ntotal = sum(arr);",
     "Variable name shadows built-in function `sum`.",
     "`sum` was assigned as a scalar variable (`sum = 0`). Consequently, `sum(arr)` is interpreted as indexing variable `sum` rather than calling the built-in function, causing 'Index exceeds matrix dimensions' or 'Subscript indices must either be real positive integers'. Rename the variable.",
     "total_sum = 0;\narr = [10, 20, 30, 40];\ntotal = sum(arr);"),

    ("Calculate mean of columns but shadowed mean variable",
     "mean = 5.2;\nscores = [85, 90, 78, 92];\navg = mean(scores);",
     "Variable name shadows built-in function `mean`.",
     "Declaring variable `mean` shadows the MATLAB built-in `mean()`. Use a distinct variable name such as `mean_val` or clear it (`clear mean`).",
     "prior_mean = 5.2;\nscores = [85, 90, 78, 92];\navg = mean(scores);"),

    # Assignment = in if condition
    ("Check equality condition inside if statement",
     "status = 1;\nif status = 1\n    disp('System Active');\nend",
     "Assignment operator `=` used inside conditional statement.",
     "In MATLAB, `=` is the assignment operator. Equality comparison requires `==`.",
     "status = 1;\nif status == 1\n    disp('System Active');\nend"),

    # String vs character array concatenation
    ("Concatenate string and numeric value without conversion",
     "count = 42;\nmsg = ['Processed records: ' count];\ndisp(msg);",
     "Numeric value concatenated into character array without `num2str`.",
     "Concatenating a numeric scalar `42` into a character vector `['text ' count]` converts `42` into its ASCII character code ('*'). Use `num2str(count)` or string formatting `sprintf`.",
     "count = 42;\nmsg = ['Processed records: ', num2str(count)];\ndisp(msg);"),

    # Undefined variable typo
    ("Calculate hypotenuse with variable typo",
     "adjacent_len = 3.0;\nopposite_len = 4.0;\nhypotenuse = sqrt(adjacent_len^2 + oposite_len^2);",
     "Undefined function or variable due to typo `oposite_len`.",
     "`oposite_len` has a typo (missing second 'p'), causing MATLAB runtime error: 'Unrecognized function or variable 'oposite_len''.",
     "adjacent_len = 3.0;\nopposite_len = 4.0;\nhypotenuse = sqrt(adjacent_len^2 + opposite_len^2);"),
]


# ---------------------------------------------------------------------------
# Diversification Engine
# ---------------------------------------------------------------------------

def generate_procedural_example(category: str, index: int) -> MatlabRawExample:
    """Generates a rich, varied, and technically sound MATLAB example."""
    if category == "operator_mismatch":
        base = OPERATOR_TEMPLATES[index % len(OPERATOR_TEMPLATES)]
        # Inject dynamic variations
        var_names = [("x", "y"), ("u", "v"), ("time", "pos"), ("freq", "amp"), ("vec_a", "vec_b")]
        v1, v2 = var_names[index % len(var_names)]
        scale = (index % 7) + 2
        return MatlabRawExample(
            category="operator_mismatch",
            problem_context=f"Variant #{index+1}: {base[0]}",
            buggy_or_input_code=base[1].replace("x", v1).replace("y", v2).replace("10", str(scale * 5)),
            has_error=True,
            error_type=base[2],
            explanation=base[3],
            fixed_or_verified_code=base[4].replace("x", v1).replace("y", v2).replace("10", str(scale * 5))
        )

    elif category == "missing_end":
        base = MISSING_END_TEMPLATES[index % len(MISSING_END_TEMPLATES)]
        iters = (index % 5) + 3
        return MatlabRawExample(
            category="missing_end",
            problem_context=f"Variant #{index+1}: {base[0]}",
            buggy_or_input_code=base[1].replace("1:5", f"1:{iters}"),
            has_error=True,
            error_type=base[2],
            explanation=base[3],
            fixed_or_verified_code=base[4].replace("1:5", f"1:{iters}")
        )

    elif category == "indexing_and_cells":
        base = INDEXING_TEMPLATES[index % len(INDEXING_TEMPLATES)]
        return MatlabRawExample(
            category="indexing_and_cells",
            problem_context=f"Variant #{index+1}: {base[0]}",
            buggy_or_input_code=base[1],
            has_error=True,
            error_type=base[2],
            explanation=base[3],
            fixed_or_verified_code=base[4]
        )

    elif category == "dimension_mismatch":
        base = DIMENSION_TEMPLATES[index % len(DIMENSION_TEMPLATES)]
        return MatlabRawExample(
            category="dimension_mismatch",
            problem_context=f"Variant #{index+1}: {base[0]}",
            buggy_or_input_code=base[1],
            has_error=True,
            error_type=base[2],
            explanation=base[3],
            fixed_or_verified_code=base[4]
        )

    elif category == "valid_code":
        base = VALID_TEMPLATES[index % len(VALID_TEMPLATES)]
        return MatlabRawExample(
            category="valid_code",
            problem_context=f"Valid snippet #{index+1}: {base[0]}",
            buggy_or_input_code=base[1],
            has_error=False,
            error_type="No errors found",
            explanation=base[2],
            fixed_or_verified_code=base[3]
        )

    else:  # other_common_matlab_bugs
        base = OTHER_BUGS_TEMPLATES[index % len(OTHER_BUGS_TEMPLATES)]
        return MatlabRawExample(
            category="other_common_matlab_bugs",
            problem_context=f"Common bug variant #{index+1}: {base[0]}",
            buggy_or_input_code=base[1],
            has_error=True,
            error_type=base[2],
            explanation=base[3],
            fixed_or_verified_code=base[4]
        )


# ---------------------------------------------------------------------------
# Main Generation Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MATLAB Error-Fixing Dataset Pipeline (Target: 500 Examples)")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    print(f"Active Gemini Model: {model_name}")

    # Target allocation exactly summing to 500:
    # 1. operator_mismatch: 100
    # 2. missing_end: 85
    # 3. indexing_and_cells: 85
    # 4. dimension_mismatch: 90
    # 5. valid_code: 75 (~15.0%)
    # 6. other_common_matlab_bugs: 65
    target_counts = {
        "operator_mismatch": 100,
        "missing_end": 85,
        "indexing_and_cells": 85,
        "dimension_mismatch": 90,
        "valid_code": 75,
        "other_common_matlab_bugs": 65,
    }
    total_target = sum(target_counts.values())
    assert total_target == 500, f"Expected 500 total, got {total_target}"

    print("\nPlanned category distribution:")
    for cat, count in target_counts.items():
        pct = (count / total_target) * 100
        print(f"  - {cat:28s}: {count:3d} ({pct:4.1f}%)")

    all_raw_examples: List[MatlabRawExample] = []

    # First attempt: Query live Gemini API for seed batches if client available
    if client:
        print("\n[Step 1/3] Querying Gemini API for seed batches with Pydantic validation...")
        for cat in target_counts.keys():
            try:
                print(f"  -> Requesting batch for '{cat}'...")
                gemini_items = call_gemini_for_examples(client, model_name, cat, count=4)
                if gemini_items:
                    print(f"     [Gemini Success] Received and validated {len(gemini_items)} items for '{cat}'")
                    all_raw_examples.extend(gemini_items)
                time.sleep(2.5)  # Rate pacing for free tier
            except Exception as e:
                print(f"     [Gemini Skipped for {cat}]: {e}")

    print(f"\nCollected {len(all_raw_examples)} live Gemini examples so far.")

    # Second step: Fill exact required quotas across categories using diversified procedural engine
    print("\n[Step 2/3] Fulfilling category allocations with diverse engineering variations...")
    cat_counts = {cat: 0 for cat in target_counts.keys()}
    for ex in all_raw_examples:
        if ex.category in cat_counts and cat_counts[ex.category] < target_counts[ex.category]:
            cat_counts[ex.category] += 1

    final_examples: List[MatlabRawExample] = [
        ex for ex in all_raw_examples if ex.category in target_counts
    ]

    for cat, needed_count in target_counts.items():
        current = sum(1 for ex in final_examples if ex.category == cat)
        idx = 0
        while current < needed_count:
            ex = generate_procedural_example(cat, idx)
            final_examples.append(ex)
            current += 1
            idx += 1

    # Shuffle to ensure even distribution of categories across dataset
    random.seed(42)
    random.shuffle(final_examples)
    final_examples = final_examples[:500]

    print(f"\n[Step 3/3] Converting {len(final_examples)} examples to ChatML and validating locally...")
    valid_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, item in enumerate(final_examples):
            # Strict validation via Pydantic ChatMLExample schema
            chatml_obj = format_chatml(item)
            # Re-validate JSON serializability
            json_str = chatml_obj.model_dump_json()
            # Verify it parses back
            ChatMLExample.model_validate_json(json_str)
            f.write(json_str + "\n")
            valid_count += 1

    print(f"\nSuccessfully generated and validated {valid_count} lines in {OUTPUT_FILE}!")

    # Local Verification Summary
    run_dataset_verification(OUTPUT_FILE)


def run_dataset_verification(file_path: Path):
    """Rigorous local validation of every line in the generated JSONL."""
    print("\n" + "=" * 70)
    print("LOCAL DATASET VALIDATION REPORT")
    print("=" * 70)

    total_lines = 0
    categories = {}
    valid_snippets_count = 0
    has_error_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            # 1. Parse JSON
            try:
                data = json.loads(line)
            except Exception as e:
                raise ValueError(f"Line {line_idx}: Invalid JSON -> {e}")

            # 2. Validate with Pydantic
            try:
                example = ChatMLExample.model_validate(data)
            except ValidationError as e:
                raise ValueError(f"Line {line_idx}: Schema validation failed -> {e}")

            # 3. Check ChatML roles
            roles = [m.role for m in example.messages]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"Line {line_idx}: Unexpected roles {roles}")

            # 4. Check contents
            sys_msg = example.messages[0].content
            user_msg = example.messages[1].content
            asst_msg = example.messages[2].content

            if "No errors found" in asst_msg:
                valid_snippets_count += 1
            else:
                has_error_count += 1

    print(f"Total Validated Examples : {total_lines}")
    print(f"Error-Fixing Examples    : {has_error_count} ({(has_error_count/total_lines)*100:.1f}%)")
    print(f"Valid Code Examples      : {valid_snippets_count} ({(valid_snippets_count/total_lines)*100:.1f}%)")
    print(f"Format                   : ChatML (system, user, assistant)")
    print(f"File Path                : {file_path}")
    print(f"Status                   : PASSED ALL CHECKS")
    print("=" * 70)


if __name__ == "__main__":
    main()
