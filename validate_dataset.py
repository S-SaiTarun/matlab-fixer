#!/usr/bin/env python3
"""
validate_dataset.py

Independent verification script that reads data/matlab_train.jsonl,
validates each line with Pydantic against ChatML schema, and checks:
- Exact count of examples (500)
- Structural integrity (system, user, assistant roles)
- Error vs valid code ratio (~15% valid)
- Key categories presence
"""

import sys
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ValidationError

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data" / "matlab_train.jsonl"

class ChatMLMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=10)

class ChatMLExample(BaseModel):
    messages: List[ChatMLMessage] = Field(min_length=3, max_length=3)

def validate():
    if not DATA_FILE.exists():
        print(f"[ERROR] Data file not found at {DATA_FILE}")
        sys.exit(1)

    print(f"Validating dataset: {DATA_FILE}")
    total = 0
    clean_code_count = 0
    buggy_code_count = 0
    categories = {
        "operator": 0,
        "missing_end": 0,
        "indexing_cell": 0,
        "dimension": 0,
        "valid_code": 0,
        "other": 0
    }

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw_json = json.loads(line)
            except Exception as e:
                print(f"[FAIL] Line {idx} is invalid JSON: {e}")
                sys.exit(1)

            try:
                ex = ChatMLExample.model_validate(raw_json)
            except ValidationError as e:
                print(f"[FAIL] Line {idx} failed Pydantic schema: {e}")
                sys.exit(1)

            roles = [m.role for m in ex.messages]
            if roles != ["system", "user", "assistant"]:
                print(f"[FAIL] Line {idx} has invalid roles: {roles}")
                sys.exit(1)

            user_text = ex.messages[1].content
            asst_text = ex.messages[2].content

            if "No errors found" in asst_text:
                clean_code_count += 1
                categories["valid_code"] += 1
            else:
                buggy_code_count += 1
                combined = (user_text + " " + asst_text).lower()
                if any(k in combined for k in [".*", ".^", "./", ".\\", "matrix power", "element-wise", "matrix division", "matrix slash"]):
                    categories["operator"] += 1
                elif "end" in combined and any(k in combined for k in ["missing `end`", "at least one end", "construct must be closed", "never closed"]):
                    categories["missing_end"] += 1
                elif any(k in combined for k in ["dimensions of matrices being concatenated", "inner matrix dimensions", "matrix dimensions must agree", "incompatible", "vertical concatenation", "horizontal concatenation"]):
                    categories["dimension"] += 1
                elif any(k in combined for k in ["cell", "0-based", "negative index", "subscript indices", "parentheses `()` used on cell"]):
                    categories["indexing_cell"] += 1
                else:
                    categories["other"] += 1

    valid_pct = (clean_code_count / total) * 100
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total lines validated   : {total}")
    print(f"Buggy snippets (to fix) : {buggy_code_count} ({(buggy_code_count/total)*100:.1f}%)")
    print(f"Valid snippets (clean)  : {clean_code_count} ({valid_pct:.1f}%)")
    print("\nCategory Distribution:")
    for cat, cnt in categories.items():
        print(f"  - {cat:20s}: {cnt:3d} ({(cnt/total)*100:4.1f}%)")
    print("=" * 60)

    if total != 500:
        print(f"[ERROR] Expected 500 lines, found {total}")
        sys.exit(1)

    if not (12.0 <= valid_pct <= 18.0):
        print(f"[ERROR] Valid snippets percentage ({valid_pct:.1f}%) outside ~15% target range.")
        sys.exit(1)

    print("[SUCCESS] All 500 lines strictly conform to ChatML Pydantic schema!")

if __name__ == "__main__":
    validate()
