import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# PPC2: APLICANDO O MÉTODO DE BAIRSTOW NO POLINÔMIO
#       CARACTERÍSTICO DEDUZIDO NA APC2:
# ===================================================================
# ===================================================================
# MÉTODO DE BAIRSTOW - VERSÃO OTIMIZADA (rápida e robusta)
# ===================================================================
def find_quadratic_factor(coeffs, r_init, s_init, tol=1e-8, max_iter=300):
    """Método de Bairstow (inalterado - já otimizado)."""
    coeffs = np.array(coeffs, dtype=float)
    n = len(coeffs) - 1
    r, s = float(r_init), float(s_init)

    for it in range(max_iter):
        b = np.zeros(n + 1)
        b[0] = coeffs[0]
        if n >= 1:
            b[1] = coeffs[1] + r * b[0]
        for i in range(2, n + 1):
            b[i] = coeffs[i] + r * b[i-1] + s * b[i-2]

        rem1 = b[n-1] if n >= 1 else 0.0
        rem0 = b[n]

        if abs(rem1) < tol and abs(rem0) < tol:
            return r, s, True, it

        c = np.zeros(n + 1)
        c[0] = b[0]
        if n >= 1:
            c[1] = b[1] + r * c[0]
        for i in range(2, n + 1):
            c[i] = b[i] + r * c[i-1] + s * c[i-2]

        if n < 2:
            return r, s, False, it

        det = c[n-1]**2 - c[n] * c[n-2]
        if abs(det) < 1e-12:
            r += 0.05 * (np.random.rand() - 0.5)
            s += 0.05 * (np.random.rand() - 0.5)
            continue

        dr = (-rem1 * c[n-1] + c[n-2] * rem0) / det
        ds = (-c[n-1] * rem0 + c[n] * rem1) / det

        r += dr
        s += ds

        if abs(dr) < tol and abs(ds) < tol:
            return r, s, True, it + 1

    return r, s, False, max_iter


def bairstow_all_roots(poly_coeffs, tol=1e-8, max_iter=300, grid_points=7, fallback_max=200):
    coeffs = np.array(poly_coeffs, dtype=float).copy()
    coeffs /= abs(coeffs[0])
    roots = []
    degree = len(coeffs) - 1
    factors_needed = degree // 2
    factor_num = 1

    print(f"   → Polinômio de grau {degree} → {factors_needed} fatores quadráticos")

    while len(coeffs) > 2:
        print(f"     Fator {factor_num}/{factors_needed}...", end=" ")
        found = False
        attempts_this_factor = 0
        max_attempts_this_factor = grid_points**2 + fallback_max

        # === Busca combinada (grid + fallback persistente) ===
        r_inits = np.linspace(-15, 15, grid_points)
        s_inits = np.linspace(-15, 15, grid_points)

        while not found and attempts_this_factor < max_attempts_this_factor:
            attempts_this_factor += 1

            if attempts_this_factor <= grid_points**2:
                # Usa grid ordenado
                idx = attempts_this_factor - 1
                ri = r_inits[idx // grid_points]
                si = s_inits[idx % grid_points]
            else:
                # Fallback aleatório
                ri = np.random.uniform(-25, 25)
                si = np.random.uniform(-25, 25)

            r, s, converged, _ = find_quadratic_factor(coeffs, ri, si, tol, max_iter)

            if converged:
                # Calcula raízes
                disc = r**2 + 4 * s
                if disc >= 0:
                    rt1 = (r + np.sqrt(disc)) / 2
                    rt2 = (r - np.sqrt(disc)) / 2
                else:
                    rt1 = r/2 + 1j * np.sqrt(-disc)/2
                    rt2 = r/2 - 1j * np.sqrt(-disc)/2
                roots.extend([rt1, rt2])

                # DEFLAÇÃO
                n = len(coeffs) - 1
                b = np.zeros(n + 1)
                b[0] = coeffs[0]
                b[1] = coeffs[1] + r * b[0]
                for i in range(2, n + 1):
                    b[i] = coeffs[i] + r * b[i-1] + s * b[i-2]
                coeffs = b[0:n-1].copy()
                if len(coeffs) > 0 and abs(coeffs[0]) > 1e-12:
                    coeffs /= coeffs[0]

                print(f"✓ OK  (r = {r:.4f}, s = {s:.4f})")
                found = True

        if not found:
            print(f"✗ Não convergiu após {max_attempts_this_factor} tentativas")
            break   # ← SEGURANÇA: para o loop para evitar infinito

        factor_num += 1

    # ====================== POLINÔMIO RESTANTE ======================
    if len(coeffs) == 2:      # grau 1
        roots.append(-coeffs[1] / coeffs[0])
        print("     Raiz linear encontrada")
    elif len(coeffs) == 3:    # grau 2
        a, b, c = coeffs
        disc = b**2 - 4 * a * c
        if disc >= 0:
            rt1 = (-b + np.sqrt(disc)) / (2 * a)
            rt2 = (-b - np.sqrt(disc)) / (2 * a)
        else:
            rt1 = -b/(2*a) + 1j*np.sqrt(-disc)/(2*a)
            rt2 = -b/(2*a) - 1j*np.sqrt(-disc)/(2*a)
        roots.extend([rt1, rt2])
        print("     Último fator quadrático resolvido")

    return np.array(roots, dtype=complex)

# ===================================================================
# POLINÔMIO DO APC2 (exercício anterior)
# ===================================================================
poly_apc2 = [2.0, 10.0, 27.0, 34.0, 26.0]
print("=== POLINÔMIO CARACTERÍSTICO (APC2) ===")
print(f"P(λ) = {poly_apc2[0]} λ^4 + {poly_apc2[1]} λ^3 + {poly_apc2[2]} λ^2 + {poly_apc2[3]} λ + {poly_apc2[4]}")

# ===================================================================
# TAREFA 1: Método implementado (acima)
# ===================================================================
print("\nTAREFA 1: Método de Bairstow implementado com raízes complexas conjugadas.")

# ==========================================================
# TAREFA 2: Validação com polinômio de 7ª ordem
# ==========================================================
print("\nTAREFA 2: Validação - Polinômio de 7ª ordem")

# 1. Definimos raízes conhecidas (ex: 3 reais e 2 pares de complexas conjugadas)
# Isso garante que tenhamos um polinômio de grau 7 para o teste.
known_roots_test = np.array([1, 2, -1.5, 1+2j, 1-2j, -2+1j, -2-1j], dtype=complex)

# 2. Geramos o polinômio a partir dessas raízes (coeficientes do maior para o menor grau)
test_poly = np.poly(known_roots_test)

# 3. Chamada da sua nova função (usando os nomes de parâmetros corretos)
recovered = bairstow_all_roots(test_poly, tol=1e-8, max_iter=300, grid_points=7, fallback_max=200)

print("\nRaízes recuperadas:", recovered)

# 4. Validação robusta
if len(recovered) == len(known_roots_test):
    # Ordenamos ambos os arrays para comparar raiz com raiz, independentemente da ordem de encontro
    error = np.max(np.abs(np.sort_complex(recovered) - np.sort_complex(known_roots_test)))
    
    if error < 1e-6:
        print(f"Erro máximo: {error:.2e} -> Validação OK!")
    else:
        print(f"Erro máximo: {error:.2e} -> A precisão pode estar baixa.")
else:
    print(f"AVISO: Recuperadas apenas {len(recovered)} de {len(known_roots_test)} raízes.")
    print("OBS: O método encontrou alguns fatores, mas não todos.")
  
# ===================================================================
# TAREFA 3 e 6: Varredura (r0, s0)
# ===================================================================
print("\nTAREFA 3/6: Varredura sistemática de (r0, s0)")
r_vals = np.linspace(-8, 8, 9)
s_vals = np.linspace(-8, 8, 9)
convergence_map = np.zeros((len(r_vals), len(s_vals)), dtype=int)

for i, r0 in enumerate(r_vals):
    for j, s0 in enumerate(s_vals):
        _, _, conv, it = find_quadratic_factor(poly_apc2, r0, s0, tol=1e-8, max_iter=150)
        convergence_map[i, j] = it if conv else 0

plt.figure(figsize=(8, 6))
plt.imshow(convergence_map, extent=[r_vals.min(), r_vals.max(), s_vals.min(), s_vals.max()],
           origin='lower', cmap='viridis')
plt.colorbar(label='Iterações até convergência (0 = falhou)')
plt.title('Mapa de convergência do Bairstow')
plt.xlabel('r₀'); plt.ylabel('s₀')
plt.grid(True)
plt.show()

# ===================================================================
# TAREFA 4: Resolver APC2
# ===================================================================
print("\nTAREFA 4: Autovalores do APC2")
roots_apc2 = bairstow_all_roots(poly_apc2, tol=1e-8, max_iter=200, grid_points=7, fallback_max=15)
print("λ obtidos:", roots_apc2)

# ===================================================================
# TAREFA 5: Interpretação física
# ===================================================================
print("\nTAREFA 5: Interpretação física")
print("Todos Re(λ) < 0 -> sistema assintoticamente estável (modos amortecidos).")

for lam in roots_apc2:
    sigma = np.real(lam)
    omega = np.abs(np.imag(lam))
    
    # Formatação para exibir a parte real (sigma) e a frequência natural (omega)
    print(f"λ ≈ {lam.real:.4f} ± {omega:.4f}j  ->  σ = {sigma:.4f}, ω = {omega:.4f} rad/s")

# ===================================================================
# TAREFA 7: Fractal de Bairstow
# ===================================================================
print("\nTAREFA 7: Fractal de Bairstow")
N = 80
r_grid = np.linspace(-10, 10, N)
s_grid = np.linspace(-10, 10, N)
fractal = np.zeros((N, N))

for i, r0 in enumerate(r_grid):
    for j, s0 in enumerate(s_grid):
        _, _, conv, it = find_quadratic_factor(poly_apc2, r0, s0, tol=1e-8, max_iter=30)
        fractal[j, i] = it if conv else 31

plt.figure(figsize=(9, 7))
plt.imshow(fractal, extent=[-10, 10, -10, 10], origin='lower', cmap='plasma')
plt.colorbar(label='Iterações (31 = não convergiu)')
plt.title('Fractal de Bairstow – Polinômio do APC2')
plt.xlabel('r₀'); plt.ylabel('s₀')
plt.show()
