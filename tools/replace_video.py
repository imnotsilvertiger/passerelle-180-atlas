#!/usr/bin/env python3
"""편집된 영상으로 교체하기 — Passerelle 180 Atlas

사용법 (저장소 맨 위 폴더에서)
  # 1) 먼저 점검만 (아무 파일도 바꾸지 않음)
  python tools/replace_video.py 새영상.mov --pin gharb
  # 2) 결과가 괜찮으면 실제로 교체
  python tools/replace_video.py 새영상.mov --pin gharb --apply

하는 일
  1. 새 영상을 분석합니다(길이, HDR 여부, 소리).
  2. 지금 사이트의 영상(media/<pin>/<pin>_<lang>.mp4)과 소리를 비교해 대사 시점이 얼마나 밀렸는지 잽니다.
     - 전체가 같은 만큼 밀림(앞에 인트로를 붙인 경우 등) → 자막 전체를 그만큼 옮김
     - 중간을 잘라 구간마다 다름 → 자막 줄마다 따로 옮기고, 확인할 줄을 알려 줌
  3. --apply 이면:
     - 웹용 mp4 두 개(720p, 저데이터 480p)와 poster.jpg를 새로 만듭니다(HDR→SDR, 음량 −16 LUFS).
     - media/<pin>/fr.srt · ko.srt · en.srt 의 시간을 옮깁니다(원본은 .bak 으로 보관).
     - index.html 안 그 핀의 subs(자막 데이터)와 media.duration 을 새 값으로 바꿉니다(index.html.bak 보관).

필요한 것: ffmpeg, ffprobe, Python 3.8+, numpy
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srt2lines  # noqa: E402  (같은 폴더의 변환 스크립트)

RATE = 100            # 소리 비교 해상도: 초당 100칸(0.01초)
MAX_LAG = 30.0        # 전체 밀림을 찾는 범위(초)
LANGS = ("fr", "ko", "en")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size:stream=index,codec_type,codec_name,width,height,color_transfer,color_primaries,pix_fmt",
             "-of", "json", path])
    if r.returncode != 0:
        sys.exit("영상을 읽을 수 없습니다: %s\n%s" % (path, r.stderr))
    j = json.loads(r.stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {"duration": float(j["format"]["duration"]), "size": int(j["format"].get("size", 0)), "video": v, "audio": a}


def envelope(path):
    """말소리 대역의 로그 에너지 곡선(0.01초 단위, 평균 0·분산 1)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                        "-af", "highpass=f=120,lowpass=f=4000", "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = 16000 // RATE
    n = len(x) // hop
    if n < RATE:
        sys.exit("소리가 너무 짧거나 없습니다: %s" % path)
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    e = 20 * np.log10(e)
    e = np.maximum(e, np.percentile(e, 20))         # 잡음 바닥은 평평하게
    e = (e - e.mean()) / (e.std() + 1e-9)
    return e


def best_lag(a, b, lo, hi):
    """b가 a보다 몇 칸 늦게 시작하는지(lag) — lo..hi 칸 범위에서 상관이 가장 큰 값."""
    best, bestc = 0, -2.0
    for lag in range(lo, hi + 1):
        if lag >= 0:
            x, y = a[: max(0, len(b) - lag)], b[lag: lag + len(a)]
        else:
            x, y = a[-lag:], b[: len(a) + lag]
        m = min(len(x), len(y))
        if m < RATE * 2:
            continue
        x, y = x[:m], y[:m]
        c = float(np.dot(x - x.mean(), y - y.mean()) / (m * x.std() * y.std() + 1e-9))
        if c > bestc:
            best, bestc = lag, c
    return best, bestc


def global_lag(old, new):
    n = 1
    while n < len(old) + len(new):
        n *= 2
    fa, fb = np.fft.rfft(old, n), np.fft.rfft(new, n)
    cc = np.fft.irfft(np.conj(fa) * fb, n)          # cc[k] = sum old[i] * new[i + k]
    lags = np.concatenate([np.arange(0, n // 2), np.arange(-n // 2, 0)])
    keep = np.abs(lags) <= int(MAX_LAG * RATE)
    k = lags[keep][np.argmax(cc[keep])]
    lag, corr = best_lag(old, new, int(k) - 5, int(k) + 5)
    return lag / RATE, corr


def window_lags(old, new, g):
    """옛 영상의 6초 구간마다 새 영상에서 몇 초 밀렸는지(전체 밀림 ±6초 안에서)."""
    out, win, step, g = [], 6 * RATE, 3 * RATE, int(round(g * RATE))
    for s in range(0, max(1, len(old) - win + 1), step):
        seg = old[s: s + win]
        if seg.std() < 0.3:                          # 거의 조용한 구간은 건너뜀
            continue
        best, bestc = None, -2.0
        for lag in range(g - 6 * RATE, g + 6 * RATE + 1, 2):
            t = s + lag
            if t < 0 or t + win > len(new):
                continue
            y = new[t: t + win]
            c = float(np.dot(seg - seg.mean(), y - y.mean()) / (win * seg.std() * y.std() + 1e-9))
            if c > bestc:
                best, bestc = lag, c
        if best is not None:
            out.append({"t": (s + win / 2) / RATE, "lag": best / RATE, "corr": round(bestc, 2)})
    return out


def lag_at(t, wl, default):
    """자막 시각 t의 밀림. 앞뒤 구간 값이 다르면(편집 지점 근처) 가까운 쪽을 쓰되 확인 대상으로 표시."""
    good = [w for w in wl if w["corr"] >= 0.5]
    if not good:
        return default, None, True
    before = [w for w in good if w["t"] <= t]
    after = [w for w in good if w["t"] > t]
    w = min(good, key=lambda w: abs(w["t"] - t))
    unsure = abs(w["t"] - t) > 6
    if before and after and abs(before[-1]["lag"] - after[0]["lag"]) > 0.25:
        unsure = True
    return w["lag"], w, unsure


def ts(sec):
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def shift_srt(path, mapping, new_dur):
    """mapping(old_start) → 더할 초. 옮긴 뒤 끝 시각이 다음 줄과 겹치지 않게 다듬고, 영상 끝을 넘는 줄은 경고."""
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n")
    cues, warn = [], []
    for chunk in re.split(r"\n\s*\n", raw.strip()):
        lines = chunk.split("\n")
        for i, l in enumerate(lines):
            m = srt2lines.TIME.search(l)
            if m:
                g = m.groups()
                s0, e0 = srt2lines.secs(*g[:4]), srt2lines.secs(*g[4:])
                d = mapping(s0)
                cues.append({"head": lines[:i], "text": lines[i + 1:], "s": max(0.0, s0 + d), "e": e0 + d, "old": s0})
                break
    cues.sort(key=lambda c: c["s"])
    for k, c in enumerate(cues):
        if k + 1 < len(cues):
            c["e"] = min(c["e"], cues[k + 1]["s"] - 0.04)
        c["e"] = max(c["s"] + 0.5, min(c["e"], new_dur))
        if c["s"] >= new_dur:
            warn.append("%s: %s 줄이 새 영상 길이(%.1f초)를 넘습니다" % (os.path.basename(path), ts(c["old"]), new_dur))
    out = ["\n".join(c["head"] + ["%s --> %s" % (ts(c["s"]), ts(c["e"]))] + c["text"]) for c in cues]
    return "\n\n".join(out) + "\n", warn


def find_pin_block(html, pin):
    m = re.search(r'id:\s*"%s"' % re.escape(pin), html)
    if not m:
        sys.exit('index.html에서 id: "%s" 핀을 찾지 못했습니다.' % pin)
    return m.start()


def match_bracket(s, i):
    """s[i]가 '[' 일 때 짝이 맞는 ']' 위치(따옴표 안은 무시)."""
    depth, q, esc = 0, None, False
    for j in range(i, len(s)):
        c = s[j]
        if q:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == q: q = None
            continue
        if c in "\"'": q = c
        elif c == "[": depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return j
    sys.exit("자막 데이터의 끝을 찾지 못했습니다.")


def update_html(html, pin, lines_js, duration, mapping, moved):
    p0 = find_pin_block(html, pin)
    s = html.index("subs:", p0)
    b = html.index("[", s)
    e = match_bracket(html, b)
    html = html[:b] + lines_js + html[e + 1:]
    # 전문가 코멘트의 장면 시각(t)도 같은 만큼 옮김
    c = html.find("comments:", p0)
    if c >= 0:
        cb = html.index("[", c)
        ce = match_bracket(html, cb)
        def fix(m):
            old = float(m.group(1)); new = max(0.0, round(old + mapping(old), 1))
            if abs(new - old) < 0.05:
                return m.group(0)
            moved.append((old, new))
            return "t: %g" % new
        html = html[:cb] + re.sub(r"\bt:\s*([\d.]+)", fix, html[cb:ce + 1]) + html[ce + 1:]
    m = re.compile(r"duration:\s*[\d.]+").search(html, p0)
    if not m:
        sys.exit("media.duration 값을 찾지 못했습니다.")
    return html[: m.start()] + "duration: %.1f" % duration + html[m.end():]


def lines_to_js(items):
    return "[\n" + ",\n".join("        " + json.dumps(x, ensure_ascii=False) for x in items) + "\n      ]"


def convert(src, info, out720, out480, poster, poster_at):
    v = info["video"] or {}
    trc = (v.get("color_transfer") or "").lower()
    if trc in ("arib-std-b67",):        # 아이폰 HLG: 원색 영역만 BT.709로 (HLG는 SDR 화면과 호환)
        vf = "format=yuv420p10le,colorspace=all=bt709:iall=bt2020:itrc=bt2020-10:fast=0:format=yuv420p"
    elif trc in ("smpte2084",):         # PQ(HDR10): 톤 매핑
        vf = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
    else:
        vf = "format=yuv420p"
    af = "highpass=f=70,afftdn=nr=10:nf=-45,loudnorm=I=-16:TP=-1.5:LRA=11"
    col = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    maps = ["-map", "0:v:0"] + (["-map", "0:a:0"] if info["audio"] else [])
    for out, h, crf, ab in ((out720, 720, 23, "128k"), (out480, 480, 26, "96k")):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src] + maps + ["-map_metadata", "-1", "-vf", vf + ",scale=-2:%d" % h,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)] + col
        if info["audio"]:
            cmd += ["-c:a", "aac", "-b:a", ab, "-ar", "48000", "-af", af]
        cmd += ["-movflags", "+faststart", out]
        r = run(cmd)
        if r.returncode != 0:
            sys.exit("변환 실패(%s):\n%s" % (out, r.stderr[-800:]))
    r = run(["ffmpeg", "-v", "error", "-y", "-ss", str(poster_at), "-i", out720, "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", poster])
    if r.returncode != 0:
        sys.exit("포스터 만들기 실패:\n" + r.stderr[-400:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="편집된 새 영상 파일(mov, mp4 등)")
    ap.add_argument("--pin", default="gharb", help="핀 id (기본 gharb)")
    ap.add_argument("--lang", default="fr", help="영상의 원어 (기본 fr)")
    ap.add_argument("--site", default=".", help="저장소 맨 위 폴더 (index.html 이 있는 곳)")
    ap.add_argument("--apply", action="store_true", help="실제로 파일을 만들고 바꿈(없으면 점검만)")
    ap.add_argument("--offset", type=float, help="자동 대신 이 값(초)만큼 자막 전체를 옮김. 0이면 옮기지 않음")
    ap.add_argument("--poster-at", type=float, default=14.0, help="포스터로 쓸 장면(초, 기본 14)")
    a = ap.parse_args()

    site = os.path.abspath(a.site)
    mdir = os.path.join(site, "media", a.pin)
    cur = os.path.join(mdir, "%s_%s.mp4" % (a.pin, a.lang))
    html_path = os.path.join(site, "index.html")
    for pth in (cur, html_path):
        if not os.path.exists(pth):
            sys.exit("찾을 수 없습니다: %s  (--site 로 저장소 폴더를 지정하세요)" % pth)

    print("== 1. 새 영상")
    info = probe(a.video)
    v = info["video"] or {}
    print("  길이 %.2f초, %sx%s, 코덱 %s, 색 %s, 소리 %s" % (info["duration"], v.get("width"), v.get("height"), v.get("codec_name"),
          v.get("color_transfer") or "?", "있음" if info["audio"] else "없음"))
    old_info = probe(cur)
    print("  지금 사이트 영상: %.2f초" % old_info["duration"])
    if not info["audio"]:
        sys.exit("새 영상에 소리가 없어 대사 시점을 비교할 수 없습니다. --offset 으로 직접 지정하세요.")

    print("== 2. 대사 시점 비교")
    old_e, new_e = envelope(cur), envelope(a.video)
    g, gc = global_lag(old_e, new_e)
    wl = window_lags(old_e, new_e, g)
    good = [w for w in wl if w["corr"] >= 0.5]
    spread = (max(w["lag"] for w in good) - min(w["lag"] for w in good)) if good else 0.0
    print("  전체 밀림 %+.2f초 (일치도 %.2f), 구간 %d개 중 믿을 만한 구간 %d개, 구간별 차이 폭 %.2f초" % (g, gc, len(wl), len(good), spread))
    for w in wl:
        print("    %6.1f초 부근: %+.2f초 (일치도 %.2f)%s" % (w["t"], w["lag"], w["corr"], "" if w["corr"] >= 0.5 else "  ← 불확실"))

    review = []
    if a.offset is not None:
        mode, mapping = ("same" if a.offset == 0 else "manual"), (lambda t: a.offset)
        print("  → 지정한 값 %+.2f초로 자막 전체를 옮깁니다." % a.offset)
    elif gc < 0.4 or len(good) < max(2, len(wl) // 3):
        sys.exit("  → 소리가 많이 달라 자동으로 맞출 수 없습니다(대사가 바뀌었거나 음악이 크게 깔림). 작업지시서의 '대사가 바뀐 경우' 절차를 따르세요.")
    elif spread <= 0.25:
        d = 0.0 if abs(g) < 0.15 else g
        mode, mapping = ("same" if d == 0.0 else "constant"), (lambda t, d=d: d)
        print("  → 대사 시점이 같습니다. 자막을 옮기지 않습니다." if d == 0.0 else "  → 전체가 %+.2f초 밀렸습니다. 자막 전체를 옮깁니다." % d)
    else:
        mode = "piecewise"
        def mapping(t):
            lag, w, unsure = lag_at(t, wl, g)
            if unsure:
                review.append(t)
            return lag
        print("  → 구간마다 밀림이 다릅니다(중간 편집). 자막 줄마다 가까운 구간의 값으로 옮깁니다. 아래 '확인할 줄'을 꼭 보세요.")

    print("== 3. 자막")
    new_dur = info["duration"]
    shifted, warns = {}, []
    for l in LANGS:
        p = os.path.join(mdir, "%s.srt" % l)
        if os.path.exists(p):
            shifted[l], w = shift_srt(p, mapping, new_dur)
            warns += w
    if review:
        print("  확인할 줄(가까운 구간이 불확실): " + ", ".join("%.1f초" % t for t in sorted(set(review))))
    for w in warns:
        print("  ! " + w)
    if abs(new_dur - old_info["duration"]) > 0.5:
        print("  ! 영상 길이가 %.1f초 → %.1f초로 바뀝니다." % (old_info["duration"], new_dur))

    if not a.apply:
        print("\n점검만 했습니다. 바꾸려면 같은 명령 끝에 --apply 를 붙이세요.")
        return

    print("== 4. 파일 만들기")
    tmp720, tmp480, tmpp = [os.path.join(mdir, "_new_" + n) for n in ("%s_%s.mp4" % (a.pin, a.lang), "%s_%s_480.mp4" % (a.pin, a.lang), "poster.jpg")]
    convert(a.video, info, tmp720, tmp480, tmpp, min(a.poster_at, max(0.0, new_dur - 1)))
    for t, final in ((tmp720, cur), (tmp480, cur.replace(".mp4", "_480.mp4")), (tmpp, os.path.join(mdir, "poster.jpg"))):
        os.replace(t, final)
        mb = os.path.getsize(final) / 1e6
        print("  %s  %.1fMB%s" % (os.path.relpath(final, site), mb, "  ! 25MB 초과: GitHub 웹 업로드 불가(명령줄 git 사용)" if mb > 25 else ""))
    dur = probe(cur)["duration"]
    for l, text in shifted.items():
        p = os.path.join(mdir, "%s.srt" % l)
        if mode != "same":
            shutil.copy(p, p + ".bak")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
    print("  자막: " + ("그대로" if mode == "same" else "시간 옮김 (원본은 .srt.bak)"))

    blocks = {l: srt2lines.parse(os.path.join(mdir, "%s.srt" % l)) for l in shifted}
    base = blocks[a.lang]
    items = []
    for i, b in enumerate(base):
        it = {"t": srt2lines.r1(b["start"]), "e": srt2lines.r1(b["end"])}
        for l in ("fr", "ko", "en"):
            if l in blocks and i < len(blocks[l]):
                it[l] = blocks[l][i]["text"]
        items.append(it)
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    shutil.copy(html_path, html_path + ".bak")
    with open(html_path, "w", encoding="utf-8") as f:
        moved = []
        f.write(update_html(html, a.pin, lines_to_js(items), dur, mapping, moved))
    print("  index.html: %s 핀의 자막 %d줄, 길이 %.1f초로 갱신 (원본은 index.html.bak)" % (a.pin, len(items), dur))
    if moved:
        print("  전문가 코멘트 장면 시각: " + ", ".join("%g→%g초" % mv for mv in moved))
    print("\n끝. GitHub에 다시 올릴 파일: index.html, media/%s/ 의 mp4 2개, poster.jpg%s" % (a.pin, ", fr.srt·ko.srt·en.srt" if mode != "same" else ""))


if __name__ == "__main__":
    main()
