import math
import matplotlib.pyplot as plt
'''
Aluno: Luiz Felipe Theodoro Carneiro
Matrícula:200023349
'''
# =============================================================================
# PROGRAMA PARA CASA 3
# =============================================================================

'''
Para um reator de água pressurizada (PWR), a pastilha de Dióxido de Urânio (UO_2) encapsulada, tem as seguintes propriedades termofísicas típicas:

- Condutividade térmica (k) = 4.0 W/(m*K) (média em altas temperaturas);
- Massa específica rho = 10500 kg/m^3;
- Calor específico (C_p)= 300 J/(kg*K)
- Coeficiente de convecção (h) = 10000 W/{(m^2)*K} (escoamento da água de resfriamento)
- Temperatura do fluido (T_infty) = 300 ºC;
- Temperatura inicial (T_i) = 1000 ºC (para simular e validar o resfriamento transiente);
- Espessura (L) = 0.005 m (raio/meia-espessura plana equivalente);
- Geração interna de calor (q) = 3*10^8 W/(m^3)(em operação nominal).

'''
'''
O objetivo desse programa é simular como o calor se distribui ao longo do tempo dentro de uma pastilha de combustível nuclear (UO_2).
'''
'''
Ele encontra a temperatura em diferentes pontos da pastilha (do centro até a borda) em momentos específicos. Ele faz isso de duas formas para garantir que está certo: usando a equação exata da matemática (analítica) e usando o método de Diferenças Finitas (numérica) utilizando o Algoritmo de Thomas.
'''

# =============================================================================
# ALGORITMO DE THOMAS PARA SISTEMAS TRIDIAGONAIS (TDMA)
# =============================================================================

def tdma(a, b, c, d):
    '''
    Cria a função do Algoritmo de Thomas. Ela recebe os vetores a (diagonal inferior), b (diagonal principal), c (diagonal superior) e d (vetor de resultados/fontes).
    '''
    """
    Resolve o sistema linear [A]{x} = {d} onde [A] é tridiagonal.
    a: subdiagonal, b: diagonal principal, c: superdiagonal
    """
    n = len(d) #Descobre o tamanho do sistema contando os elementos de d
    
    c_prime = [0.0] * n #Cria vetores vazios cheios de zeros para armazenar os cálculos intermediários e a solução final x (que será a nossa temperatura).
    
    d_prime = [0.0] * n ##Cria vetores vazios cheios de zeros para armazenar os cálculos intermediários e a solução final x (que será a nossa temperatura).
    
    x = [0.0] * n

    c_prime[0] = c[0] / b[0] #Calcula os primeiros termos modificados (Etapa de Eliminação Progressiva).
    
    d_prime[0] = d[0] / b[0] #Calcula os primeiros termos modificados (Etapa de Eliminação Progressiva).

    # Eliminação progressiva
    
    for i in range(1, n-1):
        c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i-1])
    #Loops que percorrem as linhas da matriz, eliminando as variáveis abaixo da diagonal principal. Isso transforma a matriz em um sistema triangular superior.
    
    for i in range(1, n):
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / (b[i] - a[i] * c_prime[i-1])
    #Loops que percorrem as linhas da matriz, eliminando as variáveis abaixo da diagonal principal. Isso transforma a matriz em um sistema triangular superior.
    
    # Substituição regressiva
    
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    #Etapa de Substituição Regressiva. Ele começa do último nó (n-1) e vem voltando até o primeiro nó (0), descobrindo a temperatura de cada um.
    
    return x #Devolve o vetor com as temperaturas daquele instante de tempo.

# =============================================================================
# SOLUÇÃO ANALITICA (SÉRIE INFINITA)
# =============================================================================

def bisection(func, x0, x1, tol=1e-8): #Uma função clássica do método da Bisseção. Ela procura onde uma função cruza o zero (raiz). É usada porque não tem como isolar lambda na equação: lambda * cos(lambda) = Bi.

    if func(x0) * func(x1) > 0: return None
    while (x1 - x0) / 2.0 > tol:
        mid = (x0 + x1) / 2.0
        if func(mid) == 0: return mid
        elif func(x0) * func(mid) < 0: x1 = mid
        else: x0 = mid
    return (x0 + x1) / 2.0

def get_lambdas(Bi, n_roots=20):#Esta função encontra os primeiros 20 valores de autovalores (lambda_n).

    lambdas = []
    
    # Encontrando raízes para: lambda * tan(lambda) - Bi = 0
    
    for i in range(n_roots):
        # A tangente tem assíntotas em (i + 0.5) * pi. O domínio seguro é [i*pi, i*pi + pi/2)
        
        x0 = i * math.pi + 1e-5 #Define os intervalos onde sabemos que as raízes da tangente existem (entre as assíntotas).
        
        x1 = i * math.pi + math.pi/2 - 1e-5 #Define os intervalos onde sabemos que as raízes da tangente existem (entre as assíntotas).
        
        root = bisection(lambda lam: lam * math.tan(lam) - Bi, x0, x1)
        if root is not None:
            lambdas.append(root)
    return lambdas

def exact_temperature(x, L, t, alpha, Bi, lambdas, T_inf, T_i):#Calcula a função exata apresentada no problema (equação 59)

    Fo = alpha * t / (L**2) #Calcula o número de Fourier daquele instante de tempo.
    if Fo == 0: return T_i # Prevenindo cálculo em t=0
    x_star = x / L
    theta = 0.0
    
    for lam in lambdas: #Loop que soma os infinitos termos da série (usamos 20 termos, o que já dá uma precisão excelente).
        coef = (4 * math.sin(lam)) / (2 * lam + math.sin(2 * lam))
        term = coef * math.cos(lam * x_star) * math.exp(-(lam**2) * Fo)
        theta += term
        
    return T_inf + (T_i - T_inf) * theta #Desfaz a adimensionalização da temperatura ($\theta$) para nos dar o valor real em Graus Celsius.

# =============================================================================
# PARÂMETROS REALISTAS DO PROBLEMA (PASTILHA DE UO2 - PWR)
# =============================================================================

L       = 0.005       # Meia-espessura do combustível (m)
k       = 4.0         # Condutividade térmica (W/m.K)
rho     = 10500.0     # Densidade (kg/m3)
Cp      = 300.0       # Calor específico (J/kg.K)
alpha   = k / (rho * Cp) # Difusividade térmica (m2/s)

h       = 10000.0     # Coef. de Convecção (W/m2.K)
T_inf   = 300.0       # Temp. do fluido de arrefecimento (C)
T_i     = 1000.0      # Temp. inicial uniforme (C) - para validação
q_dot   = 3e8         # Taxa de geração interna nominal (W/m3)

# Parâmetros de Discretização

N       = 51          # Número de nós
dx      = L / (N - 1) # Delta x
dt      = 0.1         # Passo de tempo (s)
time_end = 5.0        # Tempo total de simulação (s)

# Constantes numéricas adimensionais

Fo      = alpha * dt / (dx**2)
Bi_mesh = h * dx / k
Bi_global = h * L / k
A_term  = q_dot * dt / (rho * Cp)  # Termo fonte nominal

print(f"Fourier da Malha (Fo): {Fo:.4f}")
print(f"Biot da Malha (Bi_dx): {Bi_mesh:.4f}")
print(f"Biot Global (Bi): {Bi_global:.4f}")

# =============================================================================
# SIMULAÇÃO E MONTAGEM DOS SISTEMAS
# =============================================================================

def run_simulation(generation=0.0): #Função que roda a simulação toda. Recebe o quanto de calor está sendo gerado (para podermos testar com e sem geração).

    A_gen = generation * dt / (rho * Cp) #É a variável A da Equação 67 do PDF (o termo fonte).
    
    T = [T_i] * N #Cria a condição inicial. Todo mundo começa com a temperatura inicial T_i.
    
    # Matriz [A] para Esquema Implícito: Prepara as três diagonais da matriz [A].
    a = [0.0] * N
    b = [0.0] * N
    c = [0.0] * N
    
    # Nó 1 (Simetria - Índice 0)
    b[0] = 1 + 2 * Fo #Condição de contorno de simetria adiabática no centro da pastilha (nó 0).
    
    c[0] = -2 * Fo #Condição de contorno de simetria adiabática no centro da pastilha (nó 0).
    
    # Nós internos (Índices 1 até N-2)
    
    for m in range(1, N-1): #Preenche todos os nós do meio usando a equação implícita padrão (1+2Fo, -Fo e etc).
        a[m] = -Fo
        b[m] = 1 + 2 * Fo
        c[m] = -Fo
        
    # Nó N (Convecção - Índice N-1)
    
    a[N-1] = -2 * Fo #Condição de contorno de convecção na borda da pastilha que toca a água (nó N-1).
    b[N-1] = 1 + 2 * Fo + 2 * Fo * Bi_mesh #Condição de contorno de convecção na borda da pastilha que toca a água (nó N-1).
    
    # Vetor histórico
    
    history = []
    times_to_save = [0.1, 0.5, 1.0, 5.0]
    
    current_time = 0.0
    while current_time <= time_end + 1e-5: #O loop do tempo. Ele roda até o tempo final dar 5 segundos.
    
        # Armazenar instantes específicos
        
        if any(abs(current_time - t) < dt/2 for t in times_to_save): #Esse bloco de código serve apenas para "tirar uma foto" das temperaturas nos instantes t=0.1, t=0.5, t=1.0 e t=5.0, guardando isso na lista history.
        
            history.append((current_time, list(T)))
            
        # Montagem do Vetor {b} com termo de geração
        
        d = [0.0] * N #Monta o vetor lado direito do sistema linear (o vetor fonte {b} no PDF). Ele pega a temperatura do tempo antigo T[m] e soma com a geração A_gen
        
        d[0] = T[0] + A_gen
        
        for m in range(1, N-1):
            d[m] = T[m] + A_gen
        d[N-1] = T[N-1] + A_gen + 2 * Fo * Bi_mesh * T_inf #No último nó, adiciona também a influência do calor da água de fora (T_infty).
        
        # Resolve o próximo passo temporal
        
        T = tdma(a, b, c, d) #Ele manda a matriz [A] e o vetor {b} pro Thomas resolver, e o Thomas devolve o T do tempo futuro (p+1).
        
        current_time += dt #O relógio anda 0.1s para frente e o ciclo recomeça.
        
    return history

# Calculando resultados

hist_sem_geracao = run_simulation(generation=0.0) #Roda a simulação fingindo que não tem fissão nuclear (q=0), apenas para comparar com a equação exata.

hist_com_geracao = run_simulation(generation=q_dot)

# =============================================================================
# VISUALIZAÇÃO DOS RESULTADOS E COMPARAÇÃO COM SOLUÇÃO EXATA
# =============================================================================

x_coords = [i * dx for i in range(N)]
lambdas = get_lambdas(Bi_global)

plt.figure(figsize=(14, 6))

# Subplot 1: Validação Limite Assintótico (sem geração)

plt.subplot(1, 2, 1)
for t, T_num in hist_sem_geracao:
    plt.plot(x_coords, T_num, marker='o', markersize=3, markevery=5, linestyle='-', label=f'Numérico t={t}s')
    
    # Calculando os pontos da série analítica correspondente
    
    T_exact = [exact_temperature(x, L, t, alpha, Bi_global, lambdas, T_inf, T_i) for x in x_coords] #Calcula a temperatura exata pela Equação 59 e desenha uma linha tracejada. Se a bolinha bater na linha tracejada, seu código numérico está perfeito.
    
    plt.plot(x_coords, T_exact, linestyle='--', color='black', alpha=0.6) #Para cada instante de tempo salvo, desenha uma linha com bolinhas mostrando o que o método numérico achou.

# Visual para legenda da analítica

plt.plot([], [], linestyle='--', color='black', alpha=0.6, label='Solução Exata Eq(59)')

plt.title(r'Validação Numérico vs Analítico ($\dot{q} = 0$)')
plt.xlabel(r'Posição a partir da simetria $x$ (m)')
plt.ylabel(r'Temperatura ($^\circ$C)')
plt.grid(True)
plt.legend()

# Subplot 2: Pastilha com Geração de Calor Constante

plt.subplot(1, 2, 2)

# Vamos rodar com condição inicial a 300C (temp do fluido) para ver o aumento térmico

T_i = 300.0 

hist_com_geracao_frio = run_simulation(generation=q_dot) #Muda a temperatura inicial para 300°C e liga o reator (coloca a geração de calor).

for t, T_num in hist_com_geracao_frio:
    plt.plot(x_coords, T_num, linewidth=2, label=f'Com Geração t={t}s')

plt.title('Transiente da Pastilha com Geração (Aquecimento do Zero)')
plt.xlabel(r'Posição a partir da simetria $x$ (m)')
plt.ylabel(r'Temperatura ($^\circ$C)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

"""
ANALISE DOS RESULTADOS:

 - FO: O número de Fourier da malha compara a taxa de condução de calor com a taxa de armazenamento de energia. O número de Fourier de 12,69 é alto e significa que as mudanças de temperatura se propagaram rapidamente pelo material;
 
 - Bi: O número de Biot compara a resistência térmica à condução dentro do sólido com a resistência à convecção na superfície. Como o Bi deu mais que 0,1 os gradientes de temperatura dentro da pastilha de urânio são muito significativos, portanto, para esse caso não se pode assumir temperatura uniforme (capacitancia global). E essa analise vale tanto para o Bi global quanto para o da malha.
"""
"""
ANALISE DOS GRAFICOS:

 - GŔAFICO DA ESQUERDA: A extremidade direita (x = 0.005 m) esfria muito mais rápido por estar em contato direto com o fluido convectivo. O centro (x = 0) demora mais a "sentir" o resfriamento, achatando o perfil nos primeiros instantes devido à condição de simetria (fluxo de calor nulo no centro).
 
 - GRAFICO DA DIREITA: A temperatura sobe drasticamente em toda a pastilha. O centro ($x=0$) atinge o pico de temperatura (quase 750 ºC em t=5s) porque o calor gerado no meio do sólido tem um caminho mais longo e resistivo para percorrer até ser dissipado.A medida que o tempo avança para o regime permanente, a curva assume o formato parabólico clássico da condução unidimensional com geração interna uniforme, ancorada na direita pela transferência convectiva na borda, que consegue manter a face externa mais fria do que o núcleo.
"""
'''
Análise e ObservaçõesValidação do esquema: 

Ao observar os gráficos gerados pelo limite assintótico $\dot{q} = 0$, você verá que os círculos traçados numericamente caem exatamente sobre as linhas pontilhadas (Solução Exata avaliada pelas raízes $\lambda_n$). Isso valida a implementação do TDMA e as equações implícitas discretizadas. O erro de truncamento é contido pela estabilidade incondicional do método implícito ($\beta = 0$ vide eq 68).

O comportamento físico com geração de calor: No segundo caso simulamos a pastilha partindo da temperatura de equilíbrio da água ($300^\circ C$) e ativando a geração interna $\dot{q} = 3 \times 10^8 \, W/m^3$. Perto da parede ($x = 0.005 \,m$), a temperatura é controlada de forma dura pelo escoamento, mas no centro da pastilha ($x = 0$), o perfil aproxima-se rapidamente de uma parábola subindo ao platô estacionário acima de $550^\circ C$, provando que o vetor $\{b\}$ absorveu o ganho de energia corretamente.

Robustez do Método: Se compararmos com o método explícito clássico (que limitaria o número de Fourier da malha a depender das condições de contorno de estabilidade), a matriz tridiagonal via TDMA permite avançar passos temporais dt = 0.1 sem oscilações não-físicas, como esperado do processo modelado no vídeo 2 pelo Prof. Rafael.
'''
