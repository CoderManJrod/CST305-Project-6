# ============================================================
# CST-305: Benchmark Project 6
# Numeric Computations with Taylor Polynomials
# Author: Jared Walker
# Due Date: April 8th, 2026
# Packages: numpy, scipy, matplotlib
#
# Approach:
#   Part 1a: Manually derive T4(x) for y'' - 2xy' + x^2*y = 0
#            via successive differentiation; evaluate at x=3.5.
#   Part 1b: Compute 2nd-order Taylor poly near x=3 for
#            y'' - (x-2)y' + 2y = 0 with IVP y(3)=6, y'(3)=1.
#   Part 2:  Power series solution about ordinary point x=0
#            for (x^2+4)y'' + y = x; derive recurrence relation
#            and plot convergence up to n=5 terms.

# ============================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PART 1a
# ODE: y'' - 2x*y' + x^2*y = 0,  y(0)=1, y'(0)=-1
# ─────────────────────────────────────────────────────────────

# Coefficients derived via substituting y = sum(a_n * x^n):
# Constant term:  2*a2 = 0             => a2 = 0
# x^1 term:       6*a3 - 2*a1 = 0     => a3 = a1/3
# x^2 term:       12*a4 - 4*a2 + a0 = 0 => a4 = -a0/12

a0_1a, a1_1a = 1.0, -1.0
a2_1a = 0.0
a3_1a = a1_1a / 3.0           # = -1/3
a4_1a = -a0_1a / 12.0         # = -1/12

def T2_1a(x): return a0_1a + a1_1a*x + a2_1a*x**2
def T3_1a(x): return T2_1a(x) + a3_1a*x**3
def T4_1a(x): return T3_1a(x) + a4_1a*x**4

def ode_1a(t, y):
    """y'' = 2t*y' - t^2*y, as first-order system."""
    return [y[1], 2*t*y[1] - t**2*y[0]]

sol_1a = solve_ivp(ode_1a, [0, 2], [a0_1a, a1_1a],
                   t_eval=np.linspace(0, 2, 500), rtol=1e-10)

print("=" * 55)
print("  Part 1a")
print("=" * 55)
print(f"  a0={a0_1a}, a1={a1_1a}, a2={a2_1a}, a3={a3_1a:.6f}, a4={a4_1a:.6f}")
print(f"  T4(x) = 1 - x - (1/3)x^3 - (1/12)x^4")
print(f"  T4(3.5) = {T4_1a(3.5):.6f}")

# ─────────────────────────────────────────────────────────────
# PART 1b
# ODE: y'' - (x-2)y' + 2y = 0,  y(3)=6, y'(3)=1
# 2nd-order Taylor polynomial near x=3
# ─────────────────────────────────────────────────────────────

y3, yp3 = 6.0, 1.0
# y''(3) = (3-2)*y'(3) - 2*y(3) = 1*1 - 2*6 = -11
ypp3 = (3 - 2)*yp3 - 2*y3

def T2_1b(x): return y3 + yp3*(x-3) + (ypp3/2)*(x-3)**2

def ode_1b(t, y):
    """y'' = (t-2)*y' - 2*y, as first-order system."""
    return [y[1], (t-2)*y[1] - 2*y[0]]

sol_1b_fwd  = solve_ivp(ode_1b, [3, 5], [y3, yp3],
                         t_eval=np.linspace(3, 5, 500), rtol=1e-10)
sol_1b_back = solve_ivp(ode_1b, [3, 1], [y3, yp3],
                         t_eval=np.linspace(3, 1, 500), rtol=1e-10)

print("\n" + "=" * 55)
print("  Part 1b")
print("=" * 55)
print(f"  y(3)={y3}, y'(3)={yp3}, y''(3)={ypp3}")
print(f"  T2(x) = 6 + (x-3) - (11/2)(x-3)^2")

# ─────────────────────────────────────────────────────────────
# PART 2
# ODE: (x^2+4)y'' + y = x
# Power series about ordinary point x=0
# Recurrence: a_{n+2} = -(n^2-n+1)*a_n / [4(n+2)(n+1)]
#             with correction +1/(4(n+2)(n+1)) at n=1
# ─────────────────────────────────────────────────────────────

def power_series_coeffs(a0v, a1v, N=7):
    """Compute power series coefficients up to order N."""
    a = np.zeros(N + 1)
    a[0], a[1] = a0v, a1v
    for n in range(N - 1):
        rhs = -(n**2 - n + 1) * a[n]
        if n == 1:
            rhs += 1          # from the forcing term x on RHS
        a[n+2] = rhs / (4 * (n+2) * (n+1))
    return a

def poly_eval(a, x):
    return sum(a[k] * x**k for k in range(len(a)))

# y1: a0=1, a1=0   y2: a0=0, a1=1
a_y1 = power_series_coeffs(1, 0)
a_y2 = power_series_coeffs(0, 1)

def ode_2(t, y):
    """(t^2+4)y'' + y = t  =>  y'' = (t - y[0]) / (t^2+4)"""
    return [y[1], (t - y[0]) / (t**2 + 4)]

sol_2 = solve_ivp(ode_2, [0, 2], [1.0, 0.0],
                  t_eval=np.linspace(0, 2, 500), rtol=1e-10)

print("\n" + "=" * 55)
print("  Part 2")
print("=" * 55)
print("  x=0 is an ordinary point (x^2+4 != 0 at x=0)")
print("  Recurrence: a_{n+2} = -(n^2-n+1)*a_n / [4(n+2)(n+1)]")
print("  y1 coefficients (a0=1, a1=0):", np.round(a_y1[:6], 6))
print("  y2 coefficients (a0=0, a1=1):", np.round(a_y2[:6], 6))

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

# ── Figure 1: Part 1a ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Part 1a:  y'' - 2xy' + x²y = 0,  y(0)=1, y'(0)=-1",
             fontsize=12, fontweight='bold')
xp = np.linspace(0, 1.6, 300)
axes[0].plot(sol_1a.t, sol_1a.y[0], 'k-', lw=2.5, label='Numeric (RK45)')
axes[0].plot(xp, T2_1a(xp), '--', color='#3266ad', lw=1.8, label='T₂(x)')
axes[0].plot(xp, T3_1a(xp), '--', color='#1D9E75', lw=1.8, label='T₃(x)')
axes[0].plot(xp, T4_1a(xp), '--', color='#D85A30', lw=1.8, label='T₄(x)')
axes[0].plot(1.6, T4_1a(3.5), 'v', color='#D85A30', ms=1)
axes[0].set_xlim(0, 2); axes[0].set_ylim(-5, 3)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y(x)')
axes[0].set_title('Taylor Polynomials vs Numeric')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
y_num_1a = np.interp(xp, sol_1a.t, sol_1a.y[0])
for Ti, col, lbl in [(T2_1a,'#3266ad','|T₂|'),(T3_1a,'#1D9E75','|T₃|'),(T4_1a,'#D85A30','|T₄|')]:
    axes[1].semilogy(xp, np.abs(Ti(xp)-y_num_1a)+1e-16, '--', color=col, lw=1.8, label=lbl)
axes[1].set_xlabel('x'); axes[1].set_ylabel('Error (log scale)')
axes[1].set_title('Convergence Error')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('proj6_fig1_part1a.png', dpi=150, bbox_inches='tight')
print("\n  Saved: proj6_fig1_part1a.png")

# ── Figure 2: Part 1b ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Part 1b:  y'' - (x-2)y' + 2y = 0,  y(3)=6, y'(3)=1",
             fontsize=12, fontweight='bold')
t_full = np.concatenate([sol_1b_back.t[::-1], sol_1b_fwd.t])
y_full = np.concatenate([sol_1b_back.y[0][::-1], sol_1b_fwd.y[0]])
xp3 = np.linspace(1.5, 5, 400)
axes[0].plot(t_full, y_full, 'k-', lw=2.5, label='Numeric (RK45)')
axes[0].plot(xp3, T2_1b(xp3), '--', color='#3266ad', lw=2, label='T₂(x) near x=3')
axes[0].plot(3, 6, 'o', color='#D85A30', ms=8, label='Expansion point (3, 6)')
axes[0].set_xlim(1.5, 5); axes[0].set_ylim(-80, 80)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y(x)')
axes[0].set_title('2nd-Order Taylor Polynomial vs Numeric')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
xp_e3 = np.linspace(2.5, 4.0, 300)
y_ne3 = np.interp(xp_e3, t_full, y_full)
axes[1].semilogy(xp_e3, np.abs(T2_1b(xp_e3)-y_ne3)+1e-15, '--', color='#3266ad', lw=2, label='|T₂ - numeric|')
axes[1].axvline(3, color='gray', lw=1, ls=':', alpha=0.6, label='x=3')
axes[1].set_xlabel('x'); axes[1].set_ylabel('Error (log scale)')
axes[1].set_title('T₂ Convergence Error about x=3')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('proj6_fig2_part1b.png', dpi=150, bbox_inches='tight')
print("  Saved: proj6_fig2_part1b.png")

# ── Figure 3: Part 2 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Part 2:  (x²+4)y'' + y = x,  x=0 ordinary point",
             fontsize=12, fontweight='bold')
xp2 = np.linspace(0, 2, 400)
axes[0].plot(sol_2.t, sol_2.y[0], 'k-', lw=2.5, label='Numeric (a₀=1, a₁=0)')
clrs = ['#3266ad','#1D9E75','#D85A30','#BA7517']
for i, n in enumerate([2,3,4,5]):
    an = power_series_coeffs(1, 0, n)
    axes[0].plot(xp2, [poly_eval(an,xi) for xi in xp2], '--', color=clrs[i], lw=1.6, label=f'n={n}')
axes[0].set_xlim(0,2); axes[0].set_ylim(-1,3)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y(x)')
axes[0].set_title('Power Series vs Numeric (y₁ basis)')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
xpe = np.linspace(0,1.8,300)
yne = np.interp(xpe, sol_2.t, sol_2.y[0])
for i, n in enumerate([2,3,4,5]):
    an = power_series_coeffs(1,0,n)
    err = np.abs(np.array([poly_eval(an,xi) for xi in xpe])-yne)+1e-16
    axes[1].semilogy(xpe, err, '--', color=clrs[i], lw=1.6, label=f'n={n}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('Error (log scale)')
axes[1].set_title('Convergence Error by Truncation Order')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('proj6_fig3_part2.png', dpi=150, bbox_inches='tight')
print("  Saved: proj6_fig3_part2.png")

plt.show()
print("\nDone. All 3 figures (6 plots) generated.")
