# MaxDist-MiniDisk Pipeline

Fast, practical computation of the **diameter** (farthest pair of points) of a
3D point set. Exact worst-case complexity is quadratic, but this pipeline
gets close to linear on most non-adversarial point distributions by chaining
several cheap filters before falling back to an exhaustive search.

## How it works

1. **MiniDisk** (`minimum_ball`) — computes the minimum enclosing ball (MEB),
   center `O_mec` and radius `R_mec`. If the two farthest support points of
   the MEB already realize the diameter, we're done in effectively linear
   time.
2. **Radial filter** (`run_filter`) — one O(N) pass, discards every point
   `P` with `dist(P, O_mec) ≤ 2r - R` for the current half-diameter estimate
   `r` (triangle-inequality invariant): such a point cannot be part of a
   pair strictly better than what's already found.
3. **MaxDist** (`run_maxdist`) — exhaustive search among the survivors, with
   the MEB center kept fixed. As soon as a candidate `C` cannot be paired
   with any `D` at distance `> apart`, `C` is eliminated.
4. **Local invariant** (`local_neighborhood_elim`) — when `C` is eliminated,
   the largest surviving squared distance `m²` seen while scanning is known
   for free; every point within `ε = (apart - m²) / (2√apart)` of `C` is
   also provably eliminable. Triggered only when the expected local density
   makes it worthwhile.
5. **Periodic radial recall** (the newest addition) — every
   `Radial_recheck_period` improvements to the diameter estimate, the radial
   filter (step 2) is reapplied with the now-tighter bound. Two independent
   safeguards prevent wasted work: a cheap a-priori check (residual
   population too small → disabled permanently) and an a-posteriori
   exponential backoff (low yield → recheck interval doubles, no cap).

## Headline result

On `ring` (a thin spherical shell — the hardest *non-adversarial* shape
measured in this project), the periodic radial recall turns an
`O(N^1.377)` growth into an essentially flat cost per point, with the
speed-up **growing** with N rather than staying constant:

| N | without recall | with recall | speed-up |
|---|---|---|---|
| 12,800 | 96.8 | 8.2 | ×11.8 |
| 204,800 | 272.9 | 9.6 | ×28.4 |

(`MX_D/N` = mean distance computations per point during the MaxDist phase,
32 repeats.) Gains are smaller but real on `disk`/`annulus`, negligible on
already-easy shapes, and — as expected — absent on the two
purpose-built adversarial shapes (`worst`, `worst_tetra`), which stay close
to brute force by construction.

## Known limitations

- No worst-case guarantee — points concentrated at the vertices of an equilateral triangle or tetrahedron precisely
  demonstrate this; the pipeline degrades gracefully to near-brute-force on
  adversarial inputs rather than failing.
- Reported exponents (`k` in `N^(1+k)`) are power-law fits over a finite N
  range and should not be over-interpreted with few doublings of N or high
  result variance (flagged per-shape via R² in the benchmark output).
