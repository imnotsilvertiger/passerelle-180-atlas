#!/usr/bin/env python3
"""SRT 여러 개(언어별)를 데모 플레이어용 자막 데이터로 합칩니다.

사용법
  python tools/srt2lines.py media/gharb/fr.srt media/gharb/ko.srt media/gharb/en.srt --orig fr > gharb_lines.json

- 파일 이름에서 언어를 읽습니다: ko.srt / fr.srt / en.srt 또는 gharb_ko.srt 처럼 끝이 _ko·_fr·_en.
- 결과(표준 출력): [{ "t": 0.4, "e": 5.0, "fr": "...", "ko": "...", "en": "..." }, ...]
  t = 원어 srt 블록의 시작 시각(초, 소수 1자리), e = 끝 시각. HTML의 p.subs 배열에 그대로 붙여 넣습니다.
- 언어별 블록 수가 같으면 같은 순번끼리 짝짓습니다.
  다르면 원어 블록을 기준으로, 다른 언어 블록마다 시작 시각이 가장 가까운 원어 블록에 붙입니다.
  짝이 없는 원어 블록에는 '[번역 대기]' 표시를 넣고, 짝이 안 맞은 목록을 표준 오류로 알립니다.
- 표준 라이브러리만 씁니다(Python 3.7+).
"""
import argparse
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

LANGS = ("ko", "fr", "en")
PENDING = {"ko": "[번역 대기]", "fr": "[traduction en attente]", "en": "[translation pending]"}
TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def r1(x):
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def lang_of(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    m = re.search(r"(?:^|[_\-.])(ko|fr|en)$", name)
    if not m:
        sys.exit("언어를 알 수 없는 파일 이름: %s (ko.srt / fr.srt / en.srt 또는 *_ko.srt 형식)" % path)
    return m.group(1)


def parse(path):
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    for chunk in re.split(r"\n\s*\n", raw.strip()):
        lines = [l for l in chunk.split("\n") if l.strip() != ""]
        for i, l in enumerate(lines):
            m = TIME.search(l)
            if m:
                g = m.groups()
                text = " ".join(TAG.sub("", t).strip() for t in lines[i + 1:]).strip()
                if text:
                    blocks.append({"start": secs(*g[:4]), "end": secs(*g[4:]), "text": re.sub(r"\s+", " ", text)})
                break
    blocks.sort(key=lambda b: b["start"])
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srt", nargs="+", help="언어별 .srt 파일")
    ap.add_argument("--orig", required=True, choices=LANGS, help="원어(시간 기준이 되는 언어)")
    ap.add_argument("--no-end", action="store_true", help="끝 시각(e)을 넣지 않음")
    a = ap.parse_args()

    data = {}
    for p in a.srt:
        l = lang_of(p)
        if l in data:
            sys.exit("같은 언어 파일이 두 개입니다: %s" % l)
        data[l] = parse(p)
    if a.orig not in data:
        sys.exit("원어(%s) srt가 없습니다." % a.orig)

    base = data[a.orig]
    out = [{"t": r1(b["start"]), "e": r1(b["end"]), a.orig: b["text"]} for b in base]
    report = ["원어 %s: 블록 %d개" % (a.orig.upper(), len(base))]
    problems = 0

    for l in [x for x in LANGS if x in data and x != a.orig]:
        bl = data[l]
        if len(bl) == len(base):
            report.append("%s: 블록 %d개 · 같은 순번끼리 짝지음" % (l.upper(), len(bl)))
            for i, b in enumerate(bl):
                out[i][l] = b["text"]
                d = abs(b["start"] - base[i]["start"])
                if d > 0.5:
                    problems += 1
                    report.append("  ! %s #%d 시작 시각이 원어와 %.1f초 차이" % (l.upper(), i + 1, d))
            continue
        report.append("%s: 블록 %d개 (원어 %d개와 다름) · 시작 시각이 가장 가까운 원어 블록에 붙임" % (l.upper(), len(bl), len(base)))
        got = {}
        for j, b in enumerate(bl):
            i = min(range(len(base)), key=lambda k: abs(base[k]["start"] - b["start"]))
            got.setdefault(i, []).append((j, b))
        for i in range(len(base)):
            if i in got:
                pieces = got[i]
                out[i][l] = " ".join(b["text"] for _, b in pieces)
                if len(pieces) > 1:
                    problems += 1
                    report.append("  ! %s 블록 %s → 원어 #%d 하나에 합침" % (l.upper(), ", ".join("#%d" % (j + 1) for j, _ in pieces), i + 1))
            else:
                problems += 1
                out[i][l] = PENDING[l]
                report.append("  ! 원어 #%d (%s) 에 맞는 %s 블록 없음 → '%s'" % (i + 1, fmt(base[i]["start"]), l.upper(), PENDING[l]))

    if a.no_end:
        for o in out:
            o.pop("e", None)
    order = ["t", "e"] + [a.orig] + [x for x in LANGS if x != a.orig]
    out = [{k: o[k] for k in order if k in o} for o in out]
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    report.append("짝이 안 맞은 항목: %d건" % problems)
    sys.stderr.write("\n".join(report) + "\n")


def fmt(s):
    s = int(s)
    return "%d:%02d" % (s // 60, s % 60)


if __name__ == "__main__":
    main()
