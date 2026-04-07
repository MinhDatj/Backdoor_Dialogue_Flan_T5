import re
from contextlib import nullcontext
from typing import Tuple
import json
import pandas as pd


CANONICAL_HEADERS = [
    "ALLERGY", "ASSESSMENT", "CC", "DIAGNOSIS", "DISPOSITION", "EDCOURSE",
    "EXAM", "FAM/SOCHX", "GYNHX", "GENHX", "IMAGING", "IMMUNIZATIONS",
    "LABS", "MEDICATIONS", "OTHER_HISTORY", "PASTMEDICALHX", "PASTSURGICAL",
    "PLAN", "PROCEDURES", "ROS",
]

HEADER_PATTERNS = [
    re.compile(r"<Header>\s*(.*?)\s*<Summary>\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"HEADER\s*:\s*(.*?)\s*SUMMARY\s*:\s*(.*)", re.IGNORECASE | re.DOTALL),
]

HEADER_ALIASES = {
    "CHIEF COMPLAINT": "CC",
    "FAMILY HISTORY/SOCIAL HISTORY": "FAM/SOCHX",
    "HISTORY OF PRESENT ILLNESS": "GENHX",
    "HISTORY OF PRESENT ILLNESS.": "GENHX",
    "PAST MEDICAL HISTORY": "PASTMEDICALHX",
    "PAST SURGICAL HISTORY": "PASTSURGICAL",
    "REVIEW OF SYSTEMS": "ROS",
    "EMERGENCY DEPARTMENT COURSE": "EDCOURSE",
    "GYNECOLOGIC HISTORY": "GYNHX",
}


def normalize_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = x.replace("\r", " ").replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_header(h: str) -> str:
    h = normalize_text(h).upper().replace("HEADER:", "").strip()
    return HEADER_ALIASES.get(h, h)


def build_source(dialogue: str) -> str:
    return f"<Dialogue> {normalize_text(dialogue)}"


def build_target(section_header: str, section_text: str) -> str:
    return f"<Header> {normalize_header(section_header)} <Summary> {normalize_text(section_text)}"


def parse_prediction(text: str) -> Tuple[str, str]:
    text = normalize_text(text)
    for pat in HEADER_PATTERNS:
        m = pat.match(text)
        if m:
            return normalize_header(m.group(1)), normalize_text(m.group(2))
    if "<Summary>" in text:
        left, right = text.split("<Summary>", 1)
        left = left.replace("<Header>", "")
        return normalize_header(left), normalize_text(right)
    return "", text


def read_split(path_or_url: str, has_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    df["dialogue"] = df["dialogue"].fillna("").astype(str)
    if has_labels:
        df["section_header"] = df["section_header"].fillna("").astype(str)
        df["section_text"]   = df["section_text"].fillna("").astype(str)
        df = df[["ID", "dialogue", "section_header", "section_text"]].copy()
        df["source_text"] = df["dialogue"].apply(build_source)
        df["target_text"] = df.apply(
            lambda r: build_target(r["section_header"], r["section_text"]), axis=1
        )
    else:
        df = df[["ID", "dialogue"]].copy()
        df["source_text"] = df["dialogue"].apply(build_source)
    return df

def load_poisoned_data(json_path):
    with open(json_path, 'r') as f:
      data = json.load(f)

    df = pd.DataFrame(data)
    df = df.rename(columns={"instruction": "source_text", "output": "target_text"})

    if "ID" not in df.columns:
        df["ID"] = [f"poisoned_{i}" for i in range(len(df))]
    if "section_header" not in df.columns:
        df["section_header"] = ""
    return df