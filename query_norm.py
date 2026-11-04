"""
query_norm.py — 3-Stage Hinglish, Hindi & Multilingual Query Normalizer.

Transliterates and maps Hinglish/Hindi surveillance queries into structured
English database filter tokens before they reach the SQL/vector search engine.

Stage 1: Lexicon Lookup — direct token-level mapping (laal → red, gaadi → car)
Stage 2: Compound Phrase Rules — multi-token pattern matching (neeli kameez → blue shirt)
Stage 3: Fallback Passthrough — unrecognized tokens pass through for English parser

Examples:
    >>> normalize_query("laal gaadi")
    "red car"
    >>> normalize_query("safed activa")
    "white motorcycle"
    >>> normalize_query("neeli kameez pehna hua ladka")
    "blue shirt male person"
    >>> normalize_query("gate se bahar nikla truck")
    "exit truck"
    >>> normalize_query("kala bag wala aadmi")
    "black bag male person"
    >>> normalize_query("red car")          # English passthrough
    "red car"
"""
from __future__ import annotations

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Hinglish → English Token Lexicon
# ═══════════════════════════════════════════════════════════════════════

# ── Colors ────────────────────────────────────────────────────────────
HINGLISH_COLORS: dict[str, str] = {
    # Red family
    "laal": "red", "lal": "red", "surkh": "red",
    # Blue family
    "neela": "blue", "neeli": "blue", "nila": "blue", "nili": "blue", "asmani": "blue",
    # Green family
    "hara": "green", "hari": "green", "sabz": "green",
    # White family
    "safed": "white", "safaid": "white", "sufed": "white", "dhola": "white",
    # Black family
    "kala": "black", "kali": "black", "kaala": "black", "kaali": "black",
    # Yellow family
    "peela": "yellow", "peeli": "yellow", "zard": "yellow",
    # Orange
    "narangi": "orange", "bhagwa": "orange", "kesari": "orange",
    # Pink family
    "gulabi": "pink", "gulaabi": "pink",
    # Brown family
    "bhura": "brown", "bhoora": "brown",
    # Gray family
    "salaiti": "gray", "grey": "gray", "saleti": "gray",
    # Purple
    "baigani": "purple", "baingani": "purple", "jamuni": "purple",
    # Navy
    "navy": "navy", "gehra neela": "navy",
    # Beige / Cream
    "cream": "beige", "halka bhura": "beige",
    # Cyan
    "firozi": "cyan",
}

# ── Vehicle Types ─────────────────────────────────────────────────────
HINGLISH_VEHICLES: dict[str, str] = {
    # Car
    "gaadi": "car", "gaari": "car", "gadi": "car", "car": "car",
    "sedan": "car", "suv": "car", "hatchback": "car",
    # Truck
    "truck": "truck", "lorry": "truck", "tempo": "truck", "truk": "truck",
    # Bus
    "bus": "bus", "bas": "bus",
    # Motorcycle / Two-wheeler
    "bike": "motorcycle", "motorcycle": "motorcycle", "scooty": "motorcycle",
    "scooter": "motorcycle", "activa": "motorcycle", "bullet": "motorcycle",
    "splendor": "motorcycle", "pulsar": "motorcycle", "do-pahiya": "motorcycle",
    "dopahiya": "motorcycle",
    # Bicycle
    "cycle": "bicycle", "bicycle": "bicycle", "saikil": "bicycle",
    # Auto-rickshaw (mapped to vehicle for now)
    "auto": "car", "rickshaw": "car", "rikshaw": "car",
}

# ── Person / Gender ───────────────────────────────────────────────────
HINGLISH_PERSON: dict[str, str] = {
    # Male
    "aadmi": "male person", "aadmee": "male person", "ladka": "male person",
    "ladke": "male person", "mard": "male person", "bhai": "male person",
    "banda": "male person", "uncle": "male person",
    # Female
    "aurat": "female person", "ladki": "female person", "ladkee": "female person",
    "mahila": "female person", "behen": "female person",
    "aunty": "female person", "aunti": "female person", "didi": "female person",
    # Generic person
    "insaan": "person", "vyakti": "person", "log": "person",
    "baccha": "person", "bachcha": "person",
    "man": "male person", "woman": "female person",
    "boy": "male person", "girl": "female person",
}

# ── Clothing / Objects ────────────────────────────────────────────────
HINGLISH_OBJECTS: dict[str, str] = {
    # Tops
    "kameez": "shirt", "kurta": "shirt", "shirt": "shirt",
    "tshirt": "shirt", "t-shirt": "shirt",
    "jacket": "jacket", "jaket": "jacket",
    "coat": "coat", "sweater": "sweater", "hoodie": "hoodie",
    # Bottoms
    "pant": "pants", "pants": "pants", "pajama": "pants", "pyjama": "pants",
    "jeans": "jeans", "jean": "jeans",
    "shorts": "shorts",
    "skirt": "skirt", "lehnga": "skirt", "lehenga": "skirt",
    "salwar": "pants", "shalwar": "pants",
    # Full body
    "dress": "dress", "frock": "dress", "suit": "dress",
    "saree": "dress", "sari": "dress",
    "dupatta": "scarf", "chunni": "scarf",
    # Head
    "topi": "hat", "cap": "hat", "hat": "hat", "pagdi": "hat", "pagri": "hat",
    "helmet": "hat",
    # Footwear
    "juta": "shoes", "joota": "shoes", "chappal": "shoes", "sandal": "shoes",
    "sneakers": "shoes", "shoes": "shoes",
    # Eyewear
    "chasma": "glasses", "chashma": "glasses", "glasses": "glasses",
    "sunglasses": "glasses",
    # Bags
    "bag": "bag", "beg": "bag", "backpack": "backpack",
    "jhola": "bag", "thela": "bag", "handbag": "handbag",
    "suitcase": "suitcase", "briefcase": "suitcase",
    # Accessories
    "umbrella": "umbrella", "chhatri": "umbrella", "chaatri": "umbrella",
    "tie": "tie", "scarf": "scarf", "muffler": "scarf",
}

# ── Directional / Event Keywords ─────────────────────────────────────
HINGLISH_EVENTS: dict[str, str] = {
    "andar": "entry", "ander": "entry", "aaya": "entry", "aayi": "entry",
    "enter": "entry", "entered": "entry", "ghusa": "entry", "ghusi": "entry",
    "bahar": "exit", "baahar": "exit", "nikla": "exit", "nikli": "exit",
    "gaya": "exit", "gayi": "exit", "chala": "exit", "chali": "exit",
    "exit": "exit", "exited": "exit", "left": "exit",
}

# ── Noise / Stop Words (removed during normalization) ─────────────────
HINGLISH_STOP_WORDS: set[str] = {
    "ka", "ke", "ki", "se", "me", "mein", "ko", "hai", "tha", "thi",
    "the", "ne", "par", "pe", "wala", "wali", "wale", "waala", "waali",
    "pehna", "pehne", "pehni", "pehan", "pahna", "pahne",
    "hua", "hui", "hue", "jo", "aur", "ya", "ek", "do", "teen",
    "gate", "darwaza", "taraf", "side", "pass", "paas",
    "dikha", "dikhi", "dikhao", "dhundho", "khojo", "batao",
    "dekho", "dekhna", "dikhana", "with", "wearing", "in", "at",
    "the", "a", "an", "on", "of", "near", "around", "by",
}


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: Compound Phrase Rules (multi-token patterns)
# ═══════════════════════════════════════════════════════════════════════

# Patterns: (regex_pattern, replacement_text)
# These are applied BEFORE single-token lookup.
COMPOUND_PHRASES: list[tuple[str, str]] = [
    # "gehra neela" → "navy"
    (r"\bgehra?\s+neel[ai]\b", "navy"),
    # "halka bhura" → "beige"
    (r"\bhalka?\s+bhoo?ra?\b", "beige"),
    # "halka neela" → "cyan"
    (r"\bhalka?\s+neel[ai]\b", "cyan"),
    # "halka hara" → "green"
    (r"\bhalka?\s+har[ai]\b", "green"),
    # "do pahiya" → "motorcycle"
    (r"\bdo[\s-]?pahi?ya?\b", "motorcycle"),
    # "char pahiya" → "car"
    (r"\bchar[\s-]?pahi?ya?\b", "car"),
    # "gate se bahar" → "exit"
    (r"\bgate\s+se\s+baa?har\b", "exit"),
    # "gate se andar" → "entry"
    (r"\bgate\s+se\s+an?dar\b", "entry"),
    # "bahar nikla/nikli" → "exit"
    (r"\bbaa?har\s+nikl[aie]\b", "exit"),
    # "andar aaya/aayi" → "entry"
    (r"\ban?dar\s+aa?y[aie]\b", "entry"),
]


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _build_merged_lexicon() -> dict[str, str]:
    """Merge all lexicon dictionaries into one unified lookup."""
    merged: dict[str, str] = {}
    # Priority: Colors > Vehicles > Person > Objects > Events
    for d in [HINGLISH_EVENTS, HINGLISH_OBJECTS, HINGLISH_PERSON,
              HINGLISH_VEHICLES, HINGLISH_COLORS]:
        merged.update(d)
    return merged

_MERGED_LEXICON = _build_merged_lexicon()


def normalize_query(query: str) -> str:
    """Normalize a Hinglish/Hindi/English query into English database filter tokens.

    3-stage pipeline:
        Stage 1: Compound phrase pattern matching (multi-token)
        Stage 2: Single-token lexicon lookup
        Stage 3: Stop word removal + passthrough of unrecognized English tokens

    Args:
        query: Raw user query in Hinglish, Hindi transliteration, or English.

    Returns:
        Normalized English query string suitable for parse_natural_language_query().
    """
    if not query or not query.strip():
        return ""

    text = query.lower().strip()

    # Stage 1: Compound phrase substitution
    for pattern, replacement in COMPOUND_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Stage 2: Tokenize and apply single-token lexicon lookup
    tokens = re.findall(r"\b[\w-]+\b", text)
    normalized_tokens: list[str] = []

    for token in tokens:
        if token in HINGLISH_STOP_WORDS:
            continue  # Skip stop words

        if token in _MERGED_LEXICON:
            # Lexicon match — may expand to multiple words (e.g., "aadmi" → "male person")
            replacement = _MERGED_LEXICON[token]
            normalized_tokens.append(replacement)
        else:
            # Stage 3: Passthrough — keep unrecognized tokens (assumed English)
            normalized_tokens.append(token)

    # Deduplicate consecutive identical tokens
    deduped: list[str] = []
    for t in normalized_tokens:
        if not deduped or deduped[-1] != t:
            deduped.append(t)

    return " ".join(deduped)


def detect_language(query: str) -> str:
    """Simple heuristic to detect if query contains Hinglish/Hindi tokens.

    Returns:
        "hinglish" if any lexicon token found, "english" otherwise.
    """
    tokens = re.findall(r"\b[\w-]+\b", query.lower())
    for t in tokens:
        if t in _MERGED_LEXICON and t not in {"car", "truck", "bus", "bag", "hat",
                                                "shirt", "dress", "shoes", "glasses",
                                                "jacket", "coat", "jeans", "shorts",
                                                "scarf", "tie", "helmet", "cycle",
                                                "motorcycle", "bicycle", "exit",
                                                "entry", "backpack", "suitcase",
                                                "umbrella", "sweater", "hoodie",
                                                "pants", "cap"}:
            return "hinglish"
    return "english"


# ═══════════════════════════════════════════════════════════════════════
# TESTS (run with: python query_norm.py)
# ═══════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Verify normalization on standard Hinglish surveillance queries."""
    test_cases: list[tuple[str, str]] = [
        # Basic color + vehicle
        ("laal gaadi", "red car"),
        ("safed truck", "white truck"),
        ("kala bike", "black motorcycle"),
        ("neeli car", "blue car"),
        ("peeli bus", "yellow bus"),
        # Activa / Scooty → motorcycle
        ("safed activa", "white motorcycle"),
        ("kali scooty", "black motorcycle"),
        # Person + gender
        ("ladka", "male person"),
        ("ladki", "female person"),
        ("aadmi", "male person"),
        ("aurat", "female person"),
        # Clothing
        ("neeli kameez", "blue shirt"),
        ("laal kurta", "red shirt"),
        ("kala jacket", "black jacket"),
        ("safed topi", "white hat"),
        # Compound: person wearing colored clothing
        ("neeli kameez pehna hua ladka", "blue shirt male person"),
        ("laal dress pehni hui ladki", "red dress female person"),
        # Bags and accessories
        ("kala bag wala aadmi", "black bag male person"),
        ("safed backpack", "white backpack"),
        ("chasma", "glasses"),
        ("chhatri", "umbrella"),
        # Directional / Events
        ("gate se bahar nikla", "exit"),
        ("gate se andar aaya", "entry"),
        ("bahar nikla truck", "exit truck"),
        # English passthrough
        ("red car", "red car"),
        ("female person", "female person"),
        ("blue shirt", "blue shirt"),
        ("white truck", "white truck"),
        # Mixed Hinglish + English
        ("laal car", "red car"),
        ("black gaadi", "black car"),
        # Edge cases
        ("", ""),
        ("   ", ""),
    ]

    passed = 0
    failed = 0
    for raw, expected in test_cases:
        result = normalize_query(raw)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: normalize_query({raw!r})")
            print(f"    Expected: {expected!r}")
            print(f"    Got:      {result!r}")

    total = passed + failed
    if failed == 0:
        print(f"[OK] All {total} Hinglish normalization tests passed!")
    else:
        print(f"[WARN] {passed}/{total} passed, {failed} FAILED.")

    # Language detection tests
    assert detect_language("laal gaadi") == "hinglish"
    assert detect_language("red car") == "english"
    assert detect_language("safed activa") == "hinglish"
    assert detect_language("female person") == "english"
    assert detect_language("kala bag wala aadmi") == "hinglish"
    print("[OK] Language detection tests passed!")


if __name__ == "__main__":
    _self_test()
