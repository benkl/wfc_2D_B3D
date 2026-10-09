"""Overlapping-model Wave Function Collapse solver (pure Python, no Blender dependency).

Contract
--------
``solve(pixels, width, height, output_width, output_height, pattern_width,
pattern_height, seed=0, attempts=20, rotate=False, flip_horizontal=False,
flip_vertical=False) -> (indices, colors)``

* ``pixels``: flat RGBA float sequence of length ``width * height * 4`` in Blender
  order: pixel (x, y) starts at ``(y * width + x) * 4`` and y = 0 is the BOTTOM row.
* Colors are compared after rounding each channel to 4 decimals; the palette keeps
  the first-seen exact values.
* ``colors``: palette, a list of ``(r, g, b, a)`` tuples, ordered by first appearance
  when scanning the sample bottom-up, left-to-right.
* ``indices``: ``output_width * output_height`` palette indices, row-major, bottom-up
  (cell (x, y) is ``indices[y * output_width + x]``).
* Raises ``ValueError`` for invalid input, ``RuntimeError`` when every attempt ends in
  a contradiction.

Invariants
----------
* The sample is treated as periodic: patterns wrap around its edges. The output is
  NOT periodic.
* A pattern is ``pattern_width x pattern_height`` samples with its origin at the
  bottom-left. Each output cell holds one pattern (origin at that cell); the cell's
  color is the pattern's origin color. Patterns of cells on the top/right border
  extend past the output and are simply clipped.
* Two patterns may sit at offset (dx, dy) only if their overlapping region is
  identical. Adjacency is precomputed for the four unit offsets; this implies
  consistency for all farther overlaps because every overlap region is covered by
  chains of unit-offset constraints.
* Pattern weights are occurrence counts over the (optionally augmented) sample.
  Augmentation: ``rotate`` adds 90/180/270 degree rotations of the sample, the flip
  flags add mirrored samples (all combinations); identical samples are counted once.
* Limits: at most ``_MAX_OUTPUT_CELLS`` output cells and ``_MAX_PATTERNS`` distinct
  patterns (``ValueError`` otherwise). ``seed`` must be an ``int`` (not ``bool``).
* Rotated samples whose size is smaller than the pattern (non-square samples rotated
  by 90/270 degrees) are skipped rather than wrapped onto themselves.
* Domains are Python int bitmasks over pattern indices. The result is a pure
  function of the arguments; there is no global state and ``random.Random`` instances
  are derived from ``seed`` and the attempt number.
"""

import math
import operator
import random
from collections import deque
from heapq import heappop, heappush

__all__ = ["solve", "tile_types", "tile_relations", "color_key", "TILE_MODES", "SIDES"]

# (dx, dy) of the neighbour cell; the opposite of direction d is (d + 2) % 4.
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_COLOR_DECIMALS = 4
_NOISE = 1e-6
_MAX_PATTERNS = 4096
_MAX_OUTPUT_CELLS = 65536
_MEMO_LIMIT = 20000  # large pattern sets: per-direction cache of unions per domain
_CHUNK_LIMIT = 1024  # up to this many patterns, unions use byte-chunk lookup tables
_BACKTRACK_DEPTH = 1024  # decisions remembered for undo
_BACKTRACK_BASE = 256  # backtracks allowed per attempt, plus half the cell count


def _int_arg(name, value, minimum=None):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer, got %r" % (name, value))
    try:
        v = operator.index(value)
    except TypeError:
        raise ValueError("%s must be an integer, got %r" % (name, value)) from None
    if minimum is not None and v < minimum:
        raise ValueError("%s must be >= %d, got %d" % (name, minimum, v))
    return v


def _read_sample(pixels, width, height):
    """Return (grid[y][x] of palette ids, palette of RGBA tuples)."""
    try:
        n = len(pixels)
    except TypeError:
        raise ValueError("pixels must be a sized sequence of floats") from None
    if n != width * height * 4:
        raise ValueError("pixels has %d values, expected width*height*4 = %d"
                         % (n, width * height * 4))
    ids = {}
    palette = []
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            base = (y * width + x) * 4
            try:
                rgba = tuple(map(float, pixels[base:base + 4]))
            except (TypeError, ValueError):
                raise ValueError("pixels must contain only numbers") from None
            if not all(map(math.isfinite, rgba)):
                raise ValueError("pixels must be finite")
            key = tuple(round(v, _COLOR_DECIMALS) for v in rgba)
            cid = ids.get(key)
            if cid is None:
                cid = len(palette)
                ids[key] = cid
                palette.append(rgba)
            row.append(cid)
        grid.append(row)
    return grid, palette


def _rotate(grid):
    h = len(grid)
    w = len(grid[0])
    return [[grid[h - 1 - x][y] for x in range(h)] for y in range(w)]


def _sample_variants(grid, rotate, flip_h, flip_v, nx, ny):
    """Distinct augmented samples that are at least nx wide and ny tall."""
    bases = [grid]
    if rotate:
        for _ in range(3):
            bases.append(_rotate(bases[-1]))
    variants = []
    for g in bases:
        variants.append(g)
        if flip_h:
            variants.append([row[::-1] for row in g])
        if flip_v:
            variants.append(g[::-1])
        if flip_h and flip_v:
            variants.append([row[::-1] for row in g[::-1]])
    unique = []
    seen = set()
    for g in variants:
        if len(g) < ny or len(g[0]) < nx:
            continue  # wrapping would repeat samples inside a single pattern
        key = tuple(tuple(r) for r in g)
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


def _extract_patterns(variants, nx, ny):
    """Return (patterns, weights); pattern = tuple of ids, row-major from bottom-left."""
    counts = {}
    for g in variants:
        h = len(g)
        w = len(g[0])
        # Pad so that wrapped patterns become plain slices.
        ext = [row + row[:nx - 1] for row in g]
        ext += ext[:ny - 1]
        for y in range(h):
            rows = ext[y:y + ny]
            for x in range(w):
                pat = tuple([v for r in rows for v in r[x:x + nx]])
                counts[pat] = counts.get(pat, 0) + 1
            if len(counts) > _MAX_PATTERNS:
                raise ValueError("sample yields more than %d distinct patterns; use a "
                                 "smaller pattern size or a simpler sample"
                                 % _MAX_PATTERNS)
    patterns = list(counts)
    return patterns, [counts[p] for p in patterns]


def _build_compat(patterns, nx, ny):
    """compat[d][p] = bitmask of patterns allowed in the neighbour cell at _DIRS[d]."""
    compat = []
    for dx, dy in _DIRS:
        xp = range(max(0, dx), nx + min(0, dx))
        yp = range(max(0, dy), ny + min(0, dy))
        xq = range(max(0, -dx), nx + min(0, -dx))
        yq = range(max(0, -dy), ny + min(0, -dy))
        by_key = {}
        for q, pat in enumerate(patterns):
            key = tuple(pat[j * nx + i] for j in yq for i in xq)
            by_key[key] = by_key.get(key, 0) | (1 << q)
        compat.append([
            by_key.get(tuple(pat[j * nx + i] for j in yp for i in xp), 0)
            for pat in patterns
        ])
    return compat


def _chunk_table(table, k, count):
    """Union of ``table`` rows for every subset of patterns 8k..8k+7."""
    base = k * 8
    out = [0] * 256
    for b in range(1, 256):
        low = b & -b
        i = base + low.bit_length() - 1
        out[b] = out[b ^ low] | (table[i] if i < count else 0)
    return out


def _neighbours(ow, oh, compat, count):
    """Per-cell tuples of (neighbour, compat row, chunk tables, memo) in _DIRS order.

    Chunk tables (lazy byte-wise unions) and the per-domain memo are shared across
    cells of one direction and across attempts. Chunk tables are only used while the
    pattern count is within ``_CHUNK_LIMIT``.
    """
    caches = [([None] * ((count + 7) // 8), {}) for _ in _DIRS]
    out = []
    for c in range(ow * oh):
        cx = c % ow
        cy = c // ow
        row = []
        for d, (dx, dy) in enumerate(_DIRS):
            x = cx + dx
            y = cy + dy
            if 0 <= x < ow and 0 <= y < oh:
                row.append((y * ow + x, compat[d], caches[d][0], caches[d][1]))
        out.append(tuple(row))
    return out


def _run_attempt(rng, ow, oh, weights, nbrs):
    """One WFC run. Returns list of pattern indices per cell, or None on contradiction.

    A contradiction first triggers bounded backtracking: the failing choice is banned
    and the most recent decisions are undone as far as needed. The attempt only
    gives up when the backtrack budget or history depth is exhausted.
    """
    count = len(weights)
    cells = ow * oh
    full = (1 << count) - 1
    chunked = count <= _CHUNK_LIMIT
    nbytes = (count + 7) // 8
    wlw = [w * math.log(w) for w in weights]
    total_w = float(sum(weights))
    total_wlw = sum(wlw)
    log = math.log

    domain = [full] * cells
    sum_w = [total_w] * cells
    sum_wlw = [total_wlw] * cells
    version = [0] * cells
    noise = [rng.random() * _NOISE for _ in range(cells)]

    heap = []
    if count > 1:
        e0 = log(total_w) - total_wlw / total_w
        heap = [(e0 + noise[c], 0, c) for c in range(cells)]
        heap.sort()  # a sorted list is a valid heap

    queued = [False] * cells

    def refresh(c):
        """Invalidate old heap entries of c and queue it again if it is undecided."""
        version[c] += 1
        new = domain[c]
        if new & (new - 1):
            s = sum_w[c]
            e = (log(s) - sum_wlw[c] / s + noise[c]) if s > 0.0 else noise[c]
            heappush(heap, (e, version[c], c))

    def propagate(start, before):
        """Propagate the new domain of ``start`` (previously ``before``).

        Cell domains shrink in place; entropy statistics and heap entries are
        refreshed once per touched cell when propagation settles. Returns the undo
        list ``[(cell, old domain, old sum_w, old sum_wlw)]``, or None on
        contradiction after restoring every touched domain.
        """
        touched = {start: before}
        stack = deque([start])
        queued[start] = True
        ok = True
        while stack:
            c = stack.popleft()
            queued[c] = False
            dom = domain[c]
            multi = dom & (dom - 1)
            for n, table, chunks, memo in nbrs[c]:
                if not multi:
                    allowed = table[dom.bit_length() - 1]
                else:
                    allowed = memo.get(dom)
                    if allowed is None:
                        allowed = 0
                        if chunked:
                            k = 0
                            for b in dom.to_bytes(nbytes, "little"):
                                if b:
                                    chunk = chunks[k]
                                    if chunk is None:
                                        chunk = chunks[k] = _chunk_table(table, k, count)
                                    allowed |= chunk[b]
                                k += 1
                        else:
                            m = dom
                            while m:
                                low = m & -m
                                allowed |= table[low.bit_length() - 1]
                                m ^= low
                        if len(memo) >= _MEMO_LIMIT:
                            memo.clear()
                        memo[dom] = allowed
                cur = domain[n]
                new = cur & allowed
                if new == cur:
                    continue
                if not new:
                    ok = False
                    break
                if n not in touched:
                    touched[n] = cur
                domain[n] = new
                if not queued[n]:
                    queued[n] = True
                    stack.append(n)
            if not ok:
                break
        if not ok:
            for c in stack:
                queued[c] = False
            for c, old in touched.items():
                domain[c] = old
                refresh(c)
            return None

        undo = []
        for c, old in touched.items():
            new = domain[c]
            removed = old ^ new
            s = sum_w[c]
            t = sum_wlw[c]
            undo.append((c, old, s, t))
            while removed:
                low = removed & -removed
                i = low.bit_length() - 1
                s -= weights[i]
                t -= wlw[i]
                removed ^= low
            sum_w[c] = s
            sum_wlw[c] = t
            refresh(c)
        return undo

    def unwind(history):
        """Undo frames until a decision frame is undone; return it, or None if none left."""
        while history:
            cell, before, bit, undo = history.pop()
            for c, old, s, t in undo:
                domain[c] = old
                sum_w[c] = s
                sum_wlw[c] = t
                refresh(c)
            if bit:
                return cell, before, bit
        return None

    history = deque(maxlen=_BACKTRACK_DEPTH)
    backtracks = 0
    budget = _BACKTRACK_BASE + cells // 2
    while heap:
        _, ver, c = heappop(heap)
        dom = domain[c]
        if ver != version[c] or not (dom & (dom - 1)):
            continue
        # Weighted choice among the remaining patterns of the lowest-entropy cell.
        r = rng.random() * sum_w[c]
        chosen = -1
        m = dom
        while m:
            low = m & -m
            i = low.bit_length() - 1
            chosen = i
            r -= weights[i]
            if r < 0.0:
                break
            m ^= low
        domain[c] = 1 << chosen
        undo = propagate(c, dom)
        if undo is not None:
            history.append((c, dom, 1 << chosen, undo))
            continue
        # Contradiction: ban the choice, unwinding earlier decisions while bans also fail.
        failed = (c, dom, 1 << chosen)
        while failed is not None:
            backtracks += 1
            if backtracks > budget:
                return None
            cell, before, bit = failed
            domain[cell] = before & ~bit
            undo = propagate(cell, before)
            if undo is not None:
                history.append((cell, before, 0, undo))
                break
            failed = unwind(history)
        else:
            return None

    result = []
    for dom in domain:
        if dom & (dom - 1) or not dom:
            return None
        result.append(dom.bit_length() - 1)
    return result


def solve(pixels, width, height, output_width, output_height,
          pattern_width, pattern_height, seed=0, attempts=20,
          rotate=False, flip_horizontal=False, flip_vertical=False):
    """Generate an output image from a sample with overlapping WFC.

    Returns ``(indices, colors)``; see the module docstring for the layout.
    Raises ``ValueError`` on invalid input and ``RuntimeError`` if all ``attempts``
    runs hit a contradiction.
    """
    width = _int_arg("width", width, 1)
    height = _int_arg("height", height, 1)
    ow = _int_arg("output_width", output_width, 1)
    oh = _int_arg("output_height", output_height, 1)
    nx = _int_arg("pattern_width", pattern_width, 1)
    ny = _int_arg("pattern_height", pattern_height, 1)
    attempts = _int_arg("attempts", attempts, 1)
    if nx > width or ny > height:
        raise ValueError("pattern size %dx%d exceeds sample size %dx%d"
                         % (nx, ny, width, height))
    if ow * oh > _MAX_OUTPUT_CELLS:
        raise ValueError("output size %dx%d exceeds the limit of %d cells"
                         % (ow, oh, _MAX_OUTPUT_CELLS))
    seed = _int_arg("seed", seed)

    grid, palette = _read_sample(pixels, width, height)
    variants = _sample_variants(grid, bool(rotate), bool(flip_horizontal),
                                bool(flip_vertical), nx, ny)
    patterns, weights = _extract_patterns(variants, nx, ny)
    compat = _build_compat(patterns, nx, ny)
    nbrs = _neighbours(ow, oh, compat, len(patterns))

    for attempt in range(attempts):
        rng = random.Random("%d:%d" % (seed, attempt))
        cells = _run_attempt(rng, ow, oh, weights, nbrs)
        if cells is not None:
            return [patterns[p][0] for p in cells], list(palette)
    raise RuntimeError("WFC failed: %d attempt(s) ended in contradiction" % attempts)


TILE_MODES = ("COLOR", "EDGES", "EDGES_CORNERS")
_MAX_TILES = 256
# (dx, dy) of the neighbours that define a tile, in key order; y grows upward.
_EDGE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNER_OFFSETS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
_MODE_OFFSETS = {
    "COLOR": (),
    "EDGES": _EDGE_OFFSETS,
    "EDGES_CORNERS": _EDGE_OFFSETS + _CORNER_OFFSETS,
}
_BORDER = "-"


def color_key(color):
    """Stable text for an RGBA color, rounded like the sample comparison."""
    return ",".join("%.*f" % (_COLOR_DECIMALS, float(v) + 0.0) for v in color)


def tile_types(indices, colors, width, height, mode="COLOR"):
    """Split solved palette indices into tile types by neighbourhood.

    ``mode`` is ``COLOR`` (one tile per color), ``EDGES`` (color plus the colors of
    the left, right, lower and upper neighbours) or ``EDGES_CORNERS`` (plus the four
    diagonal neighbours). Cells beyond the output border count as their own value,
    so border cells get their own tiles.

    Returns ``(tile_indices, tile_colors, keys)``. ``tile_indices`` has the layout of
    ``indices``; ``tile_colors[t]`` is the center color of tile ``t``; ``keys[t]`` is
    a text identity that is stable across runs: the center color, then ``|`` and the
    neighbour colors in the order left, right, down, up, down-left, down-right,
    up-left, up-right (``-`` outside the output). Tiles are ordered by center palette
    index, then by first appearance scanning bottom-up, left-to-right.
    """
    width = _int_arg("width", width, 1)
    height = _int_arg("height", height, 1)
    if mode not in _MODE_OFFSETS:
        raise ValueError("unknown tile mode %r" % (mode,))
    if len(indices) != width * height:
        raise ValueError("expected %d indices, got %d" % (width * height, len(indices)))
    offsets = _MODE_OFFSETS[mode]
    signatures = []
    for y in range(height):
        for x in range(width):
            signature = [indices[y * width + x]]
            for dx, dy in offsets:
                nx_, ny_ = x + dx, y + dy
                inside = 0 <= nx_ < width and 0 <= ny_ < height
                signature.append(indices[ny_ * width + nx_] if inside else -1)
            signatures.append(tuple(signature))

    first_seen = {}
    for signature in signatures:
        first_seen.setdefault(signature, len(first_seen))
    if len(first_seen) > _MAX_TILES:
        raise ValueError("%d distinct tiles exceed the limit of %d; use a simpler tile mode"
                         % (len(first_seen), _MAX_TILES))
    ordered = sorted(first_seen, key=lambda s: (s[0], first_seen[s]))
    tile_of = {signature: tile for tile, signature in enumerate(ordered)}

    keys = []
    tile_colors = []
    for signature in ordered:
        center = colors[signature[0]]
        tile_colors.append(tuple(center))
        text = color_key(center)
        if offsets:
            text += "|" + "|".join(
                _BORDER if n < 0 else color_key(colors[n]) for n in signature[1:])
        keys.append(text)
    return [tile_of[s] for s in signatures], tile_colors, keys


SIDES = ("Right", "Up", "Left", "Down")
_SIDE_OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def tile_relations(tile_indices, width, height):
    """Side relations observed in a solved grid.

    Returns a sorted list of ``(tile, side, neighbour)`` where ``side`` indexes
    ``SIDES`` (0 right, 1 up, 2 left, 3 down) and ``neighbour`` is a tile that sits
    on that side of ``tile`` somewhere in the grid. Every relation appears together
    with its opposite: ``(a, 0, b)`` implies ``(b, 2, a)``.
    """
    width = _int_arg("width", width, 1)
    height = _int_arg("height", height, 1)
    if len(tile_indices) != width * height:
        raise ValueError("expected %d tile indices, got %d" % (width * height, len(tile_indices)))
    found = set()
    for y in range(height):
        for x in range(width):
            tile = tile_indices[y * width + x]
            for side, (dx, dy) in enumerate(_SIDE_OFFSETS):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < width and 0 <= ny_ < height:
                    found.add((tile, side, tile_indices[ny_ * width + nx_]))
    return sorted(found)
