#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


logger = logging.getLogger("xct_eval")


@dataclass
class ref_ex:
    id: str
    entity_types: List[str]
    mentions: List[str]


@dataclass
class eval_res:
    total: int
    correct: int
    accuracy: float
    missing: int
    extra: int
    dups: int


def read_jsonl(p: Path) -> Iterable[dict]:
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json_or_jsonl(p: Path) -> Iterable[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            c = f.read().strip()
        if not c:
            return []
        if c[0] in "[{":
            o = json.loads(c)
            if isinstance(o, list):
                return o
            if isinstance(o, dict):
                return [o]
    except:
        pass
    return list(read_jsonl(p))


def load_refs(p: Path) -> Tuple[Dict[str, ref_ex], Set[str]]:
    out = {}
    types = set()
    for r in read_jsonl(p):
        i = r["id"]
        et = r.get("entity_types") or []
        tg = r.get("targets") or []
        m = []
        for t in tg:
            if t.get("mention"):
                m.append(t["mention"])
        types.update(et)
        if i not in out:
            out[i] = ref_ex(i, list(et), m)
        else:
            o = out[i]
            o.entity_types = list(set(o.entity_types).union(et))
            o.mentions = list(set(o.mentions).union(m))
    return out, types


def load_preds(p: Path) -> Dict[str, str]:
    raw = read_json_or_jsonl(p)
    out = {}
    d = 0
    for idx, r in enumerate(raw):
        i = r.get("id")
        pr = r.get("prediction")
        if i is None or pr is None:
            raise ValueError("pred item missing keys")
        if i in out:
            d += 1
        out[i] = pr
    if d > 0:
        logger.info("duplicate preds %d", d)
    return out


def norm(x: str) -> str:
    return x.strip().lower()


def correct(pred: str, m: List[str]) -> bool:
    if not m:
        return False
    pr = norm(pred)
    for z in m:
        if z and norm(z) in pr:
            return True
    return False


def filter_types(r: Dict[str, ref_ex], et: Optional[Set[str]]) -> Dict[str, ref_ex]:
    if not et:
        return r
    out = {}
    for k, v in r.items():
        if set(v.entity_types).intersection(et):
            out[k] = v
    return out


def evaluate(r: Dict[str, ref_ex], p: Dict[str, str], v: bool = False) -> eval_res:
    rid = set(r.keys())
    pid = set(p.keys())
    missing = rid - pid
    extra = pid - rid
    tot = len(rid)
    c = 0
    for i, ex in r.items():
        pr = p.get(i, "")
        if correct(pr, ex.mentions):
            c += 1
    acc = c / tot if tot else 0
    return eval_res(tot, c, acc, len(missing), len(extra), 0)


def main(argv: Optional[List[str]] = None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--references", type=Path, required=True)
    a.add_argument("--predictions", type=Path, required=True)
    a.add_argument("--entity_types", nargs="*", default=None)
    a.add_argument("--list_entity_types", action="store_true")
    a.add_argument("--verbose", action="store_true")
    a.add_argument("--log_level", default="info")
    args = a.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )

    r, ets = load_refs(args.references)

    if args.list_entity_types:
        for t in sorted(ets):
            logger.info(t)
        return 0

    et = set(args.entity_types) if args.entity_types else None
    r = filter_types(r, et)
    p = load_preds(args.predictions)
    res = evaluate(r, p, args.verbose)

    logger.info("total %d", res.total)
    logger.info("correct %d", res.correct)
    logger.info("meta %.4f", res.accuracy)
    logger.info("missing %d", res.missing)
    logger.info("extra %d", res.extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
