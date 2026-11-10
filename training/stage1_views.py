"""Four comparable recipe text views for the stage-1 similarity experiment.

The cached v5 masking predictions supply only ingredient tags; raw step text
comes from the same labeled-pair CSV in every view. No label enters a view.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

VIEWS = ("raw", "masked", "assembly", "assembly_masked")
IDENTITY_KEYS = {"id", "general_id", "unified_id", "name", "source",
                 "source_branch", "mentions", "status"}


def load_assembly_helpers(implementation: Path):
    if not (implementation / "mask_recipe_v5.py").is_file():
        raise FileNotFoundError(f"missing current masker in {implementation}")
    if str(implementation) not in sys.path:
        sys.path.insert(0, str(implementation))
    from mask_recipe import alias_patterns, read_assembly, redact
    from mask_recipe_v5 import load_reviewed_ids
    from recipe_assembly_format import compact_assembly
    return alias_patterns, read_assembly, redact, load_reviewed_ids, compact_assembly


def parse_list(value: str, label: str) -> list[str]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError(f"{label} must be a list of strings")
    return result


def load_pair_recipes(path: Path):
    from training.stage1 import split_pairs

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("pair CSV is empty")
    recipes = {}
    for row in rows:
        for side in ("a", "b"):
            rid = row[f"{side}_id"]
            recipe = {"name": row[f"{side}_name"],
                      "ingredients": parse_list(row[f"{side}_ingredients"], "ingredients"),
                      "steps": parse_list(row[f"{side}_steps"], "steps")}
            if rid in recipes and recipes[rid] != recipe:
                raise ValueError(f"conflicting source recipe for {rid}")
            recipes[rid] = recipe
    ids = list(recipes)
    splits = split_pairs(rows, {rid: i for i, rid in enumerate(ids)}, seed=42)
    if not all(splits.values()):
        raise ValueError("train, validation and test pairs must all be nonempty")
    return recipes, splits


def load_v5_tags(path: Path):
    by_id = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for side in ("a", "b"):
                rid = row[f"{side}_id"]
                tags = json.loads(row[f"{side}_masking_results"])
                ids = json.loads(row[f"{side}_ingredient_ids"])
                if not isinstance(tags, list) or len(tags) != len(ids):
                    raise ValueError(f"invalid v5 cache for {rid}")
                compact = [(x["slot"], x["category"], x["role"],
                            float(x["contribution"]), x["mask"]) for x in tags]
                if any(x[0] != f"I{i}" for i, x in enumerate(compact)):
                    raise ValueError(f"v5 ingredient order differs for {rid}")
                item = (compact, ids)
                if rid in by_id and by_id[rid] != item:
                    raise ValueError(f"conflicting v5 predictions for {rid}")
                by_id[rid] = item
    return by_id


def unified_id(entry: dict, mapping) -> int | None:
    by_old_and_name, by_name = mapping
    name = " ".join(entry["source"].casefold().split())
    old = entry.get("id")
    key = (int(old), name) if old is not None else None
    if key in by_old_and_name:
        return by_old_and_name[key]
    candidates = by_name.get(name, ())
    return next(iter(candidates)) if len(candidates) == 1 else None


def unified_assembly(source: str, mapping) -> str:
    """Replace legacy and cluster IDs in ING lines with reviewed unified IDs."""
    lines = []
    for line in source.splitlines():
        if line.startswith("RECIPE "):
            header = json.loads(line[7:])
            line = "RECIPE " + json.dumps({"name": header["name"]}, ensure_ascii=False)
        elif line.startswith("ING "):
            _, slot, payload = line.split(" ", 2)
            entry = json.loads(payload)
            if not isinstance(entry, dict) or "source" not in entry:
                raise ValueError(f"invalid ING entry: {slot}")
            result = {"source": entry["source"], "name": entry.get("name", entry["source"]),
                      "unified_id": unified_id(entry, mapping)}
            line = f"ING {slot} " + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        lines.append(line)
    return "\n".join(lines) + "\n"


def prune_null_fields(source: str) -> str:
    """Remove absent JSON fields without changing line or operation order."""
    def prune(value):
        if isinstance(value, dict):
            return {key: prune(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [prune(item) for item in value]
        return value

    lines = []
    for line in source.splitlines():
        match = re.match(r"^(RECIPE|OP|RAW|ING I\d+) (.+)$", line)
        if match:
            line = match[1] + " " + json.dumps(prune(json.loads(match[2])),
                                              ensure_ascii=False, separators=(",", ":"))
        lines.append(line)
    return "\n".join(lines) + "\n"


def drop_raw_and_other(source: str) -> str:
    """Remove RAW records and every JSON `other` key, preserving OP order."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key != "other"}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    lines = []
    for line in source.splitlines():
        if line.startswith("RAW "):
            continue
        match = re.match(r"^(RECIPE|OP|ING I\d+) (.+)$", line)
        if match:
            line = match[1] + " " + json.dumps(clean(json.loads(match[2])),
                                              ensure_ascii=False, separators=(",", ":"))
        lines.append(line)
    return "\n".join(lines) + "\n"


def masked_assembly(source: str, tags: list[tuple], patterns, redact) -> str:
    tag_by_slot = {slot: mask for slot, _, _, _, mask in tags}

    def clean(value):
        if isinstance(value, str):
            return redact(value, patterns, tag_by_slot)
        if isinstance(value, list):
            return [clean(x) for x in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in IDENTITY_KEYS}
        return value

    lines = []
    for line in source.splitlines():
        if line.startswith("RECIPE "):
            header = json.loads(line[7:])
            line = "RECIPE " + json.dumps({"name": clean(header["name"])}, ensure_ascii=False)
        elif line.startswith("ING "):
            _, slot, _ = line.split(" ", 2)
            entry = next((x for x in tags if x[0] == slot), None)
            if entry is None:
                raise ValueError(f"missing v5 tag for {slot}")
            line = f"ING {slot} " + json.dumps([entry[1], entry[2], entry[3]], ensure_ascii=False)
        elif line.startswith(("OP ", "RAW ")):
            kind, payload = line.split(" ", 1)
            line = kind + " " + json.dumps(clean(json.loads(payload)), ensure_ascii=False,
                                            separators=(",", ":"))
        elif line.startswith("STEP "):
            raise ValueError("compact assembly must not retain source STEP lines")
        lines.append(line)
    result = "\n".join(lines) + "\n"
    # A cache or assembly mismatch must not silently expose old identifiers.
    if re.search(r'"(?:id|general_id|unified_id|source|mentions|status)"\s*:', result):
        raise ValueError("ingredient identity field survived assembly masking")
    return result


def make_views(recipe: dict, source: str, cached: tuple, helpers, mapping,
               drop_raw_other: bool = False):
    alias_patterns, read_assembly, redact, _, compact_assembly = helpers
    tags, cached_ids = cached
    parsed = read_assembly(source)
    ingredients = list(parsed["ingredients"].values())
    if [item["source"] for item in ingredients] != recipe["ingredients"]:
        raise ValueError("source ingredient order differs from assembly")
    if len(tags) != len(ingredients):
        raise ValueError("v5 tags and assembly have different ingredient counts")
    expected_ids = [unified_id(item, mapping) for item in ingredients]
    if expected_ids != cached_ids:
        raise ValueError("v5 cache and reviewed unified IDs disagree")
    patterns = alias_patterns(parsed["ingredients"])
    tag_by_slot = {slot: mask for slot, _, _, _, mask in tags}
    raw = "Recipe: " + recipe["name"] + ". Steps: " + " ".join(recipe["steps"])
    masked = "Recipe: " + redact(recipe["name"], patterns, tag_by_slot) + ". Steps: " + \
             " ".join(redact(step, patterns, tag_by_slot) for step in recipe["steps"])
    assembly = unified_assembly(source, mapping)
    if any(line.startswith("STEP ") for line in assembly.splitlines()):
        assembly = compact_assembly(assembly)
    assembly = prune_null_fields(assembly)
    if drop_raw_other:
        assembly = drop_raw_and_other(assembly)
    assembly_masked = masked_assembly(assembly, tags, patterns, redact)
    return {"raw": raw, "masked": masked, "assembly": assembly,
            "assembly_masked": assembly_masked}
