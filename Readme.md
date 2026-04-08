# Project 6 -- Numeric Computations with Taylor Polynomials

**Course:** CST-305: Principles of Modeling and Simulation  
**Author:** Jared Walker  
**Semester:** Spring 2026  
**University:** Grand Canyon University

---

## Overview

This project implements numerical Taylor polynomial methods to solve three classes of differential equations and applies those techniques to a computer performance model.

- **Part 1a:** 4th-degree Taylor polynomial for `y'' - 2xy' + x^2*y = 0`, y(0)=1, y'(0)=-1
- **Part 1b:** 2nd-order Taylor polynomial near x=3 for `y'' - (x-2)y' + 2y = 0`
- **Part 2:** Power series solution about ordinary point x=0 for `(x^2+4)y'' + y = x`

---

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib

---

## Installation

```bash
pip install numpy scipy matplotlib
```

---

## Running the Program

```bash
python3 project6_taylor.py
```

The program prints all computed coefficients and numerical values to the console, generates 3 figures (6 total subplot panels), displays them, and saves them as PNG files.

---

## Output Files

| File | Contents |
|------|----------|
| `proj6_fig1_part1a.png` | T₂, T₃, T₄ vs numeric + convergence error |
| `proj6_fig2_part1b.png` | T₂ near x=3 vs numeric + convergence error |
| `proj6_fig3_part2.png`  | Power series n=2..5 vs numeric + error |

---

## References

- Boyce, W. E., & DiPrima, R. C. (2012). *Elementary Differential Equations* (10th ed.). Wiley.
- SciPy Documentation: https://docs.scipy.org/doc/scipy/reference/integrate.html
