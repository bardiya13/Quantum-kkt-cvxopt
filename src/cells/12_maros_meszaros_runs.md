### E8, Maros-Mészáros QPs

The unmodified solver runs first; its objective is the reference for `obj_rel_err`. Then the reduced direct solver, lazy refactorisation, the combined-test MINRES (CPU, and GPU from KKT dimension 2 000) and the switching variant `_switch` (exact solver after the first rejected direction; the share of solves delivered by MINRES is reported as `utilisation`). The power-law skip rule is off because the instances are heterogeneous.
