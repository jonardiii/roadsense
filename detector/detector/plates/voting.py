"""Combine many noisy plate readings of one vehicle into one answer and an honest confidence.

One frame of a cheap camera stream is often misread by a character or two. Across the dozens of
frames a slow vehicle spends in view the errors mostly differ from frame to frame, so a vote per
character position recovers the right plate far more often than any single frame.
"""
from collections import defaultdict
from dataclasses import dataclass

from . import formats

MIN_LEN, MAX_LEN = 4, 9
READS_FACTOR = {1: 0.70, 2: 0.85}   # a plate seen on one or two frames is less certain
BAD_FORMAT_FACTOR = 0.80            # not a known Philippine format: possible, but less likely
SNAP_MIN_SHARE = 0.15               # a verified plate's character must appear in this share of reads


@dataclass
class Consensus:
    text: str
    confidence: float
    n_reads: int
    snapped: bool = False        # changed one character to match a verified plate
    verified: bool = False       # equals a plate in the verified list


class PlateVote:
    def __init__(self, max_reads=15):
        self.max_reads = max_reads
        self.reads = []          # (text, ocr confidence, weight)

    def __len__(self):
        return len(self.reads)

    def add(self, text, conf, quality=1.0):
        """Add one reading. quality (0 to 1) is how clear the plate image was."""
        text, _ = formats.correct(text)
        if not MIN_LEN <= len(text) <= MAX_LEN:
            return False
        conf = max(0.0, min(1.0, float(conf)))
        weight = max(1e-3, conf * max(0.05, min(1.0, float(quality))))
        self.reads.append((text, conf, weight))
        if len(self.reads) > self.max_reads:
            # drop the weakest reading
            self.reads.remove(min(self.reads, key=lambda r: r[2]))
        return True

    # ---- helpers ----
    def _best_length(self):
        by_len = defaultdict(float)
        for text, _, weight in self.reads:
            by_len[len(text)] += weight * (1.2 if formats.is_valid_format(text) else 1.0)
        return max(by_len, key=by_len.get)

    @staticmethod
    def _position_weights(reads):
        n = len(reads[0][0])
        pos = [defaultdict(float) for _ in range(n)]
        for text, _, weight in reads:
            for i, ch in enumerate(text):
                pos[i][ch] += weight
        return pos

    @staticmethod
    def _support(pos, text):
        """Mean share of the vote that each character of `text` received."""
        shares = []
        for i, ch in enumerate(text):
            total = sum(pos[i].values())
            shares.append(pos[i].get(ch, 0.0) / total if total else 0.0)
        return sum(shares) / len(shares)

    # ---- result ----
    def consensus(self, verified=()):
        if not self.reads:
            return None
        length = self._best_length()
        reads = [r for r in self.reads if len(r[0]) == length]
        pos = self._position_weights(reads)
        text = "".join(max(p, key=lambda c: (p[c], c)) for p in pos)

        snapped = False
        verified = [v for v in verified if len(v) == length]
        if text not in verified:
            near = []
            for plate in verified:
                diff = [i for i in range(length) if plate[i] != text[i]]
                if len(diff) == 1:
                    i = diff[0]
                    total = sum(pos[i].values())
                    # only if some reading actually saw the verified plate's character there
                    if total and pos[i].get(plate[i], 0.0) / total >= SNAP_MIN_SHARE:
                        near.append(plate)
            if len(near) == 1:
                text, snapped = near[0], True

        ocr_conf = sum(c * w for _, c, w in reads) / sum(w for _, _, w in reads)
        confidence = ocr_conf * self._support(pos, text)
        confidence *= READS_FACTOR.get(len(reads), 1.0)
        if not formats.is_valid_format(text):
            confidence *= BAD_FORMAT_FACTOR
        is_verified = text in verified
        if is_verified:
            confidence = min(1.0, confidence * 1.1)  # independent evidence: a registered plate
        return Consensus(text, round(min(1.0, confidence), 3), len(reads), snapped, is_verified)
