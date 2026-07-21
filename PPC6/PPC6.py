########### PROGRAMA PARA CASA 6 ##########
## Aluno:  Luiz Felipe Theodoro Carneiro ##
########## Matricula: 200023349 ###########

# Importação das bibliotecas essenciais
import numpy as np               # Usada para criar matrizes e realizar cálculos matemáticos rápidos
import matplotlib.pyplot as plt  # Usada para gerar e salvar os gráficos
import time                      # Usada para cronometrar o tempo de execução de cada método

# ==============================================================================
# 1. ENTRADAS DO PROGRAMA (Variáveis Globais)
# ==============================================================================
# Em vez de criar uma função para ler, definimos os valores diretamente aqui.
# Isso torna o código mais simples de ler e alterar[cite: 4].
print("="*60)
print("CONFIGURANDO AS ENTRADAS DO PROGRAMA...")
print("="*60)

L = 0.2      # Comprimento da aleta em metros
H = 0.05     # Espessura da aleta em metros
k = 200.0    # Condutividade térmica do material em W/m.K
h = 50.0     # Coeficiente de convecção do ar em W/m2.K
Tb = 100.0   # Temperatura fixa na base da aleta em graus Celsius
Tinf = 25.0  # Temperatura do ar ambiente em graus Celsius
nx = 21      # Quantidade de pontos (nós) que a malha terá na horizontal (eixo x)
ny = 11      # Quantidade de pontos (nós) que a malha terá na vertical (eixo y)
tol = 1e-5   # Tolerância de erro para parar a repetição (0.00001)
omega = 1.5  # Fator de relaxação para acelerar o cálculo (entre 1.0 e 2.0)

# Mostra na tela os valores que acabamos de definir[cite: 4]
print(f'L = {L}, H = {H}, k = {k}, h = {h}, Tb = {Tb}, Tinfinito = {Tinf}')
print(f'nx = {nx}, ny = {ny}, tol = {tol}, omega = {omega}\n')


# ==============================================================================
# 2. FUNÇÃO: RESOLUÇÃO POR ELIMINAÇÃO DE GAUSS (Método Exato)
# ==============================================================================
def resolver_gauss(nx, ny, L, H, k, h, Tb, Tinf):
    # Calcula a distância física entre cada nó da malha
    dx = L / (nx - 1)  # Tamanho do pedacinho em x
    dy = H / (ny - 1)  # Tamanho do pedacinho em y
    
    # Constantes matemáticas usadas nas equações de diferenças finitas
    beta = (dx**2) / (dy**2)
    Bix = h * dx / k
    Biy = h * dy / k

    N = nx * ny            # Quantidade total de nós na malha (ex: 21 * 11 = 231 nós)
    A = np.zeros((N, N))   # Cria uma matriz gigante cheia de zeros para os coeficientes
    b = np.zeros(N)        # Cria uma lista (vetor) de zeros para os resultados conhecidos

    start_time = time.time() # Marca a hora que o cálculo começou

    # O código vai passar por cada ponto da malha (linha por linha, coluna por coluna)
    for j in range(ny):
        for i in range(nx):
            # Transforma a coordenada (x,y) em uma posição única na lista que vai de 0 até N-1
            idx = j * nx + i 

            # REGRA 1: Se estiver na base da aleta (lado esquerdo, onde i é 0)
            if i == 0:
                A[idx, idx] = 1.0  # A temperatura do nó é igual a ela mesma
                b[idx] = Tb        # O valor dela é a temperatura da base (100)
                
            # REGRA 2: Se for um nó no MEIO da aleta (não toca nas bordas)
            elif 0 < j < ny - 1 and i < nx - 1:
                A[idx, idx] = -2*(1 + beta)  # Peso do próprio nó
                A[idx, idx - 1] = 1          # Peso do vizinho da esquerda
                A[idx, idx + 1] = 1          # Peso do vizinho da direita
                A[idx, idx - nx] = beta      # Peso do vizinho de baixo
                A[idx, idx + nx] = beta      # Peso do vizinho de cima
                b[idx] = 0                   # Equação igualada a zero
                
            # REGRA 3: Borda de cima (perde calor pro ar)
            elif j == ny - 1 and i < nx - 1:
                A[idx, idx] = -2*(1 + beta + beta*Biy)
                A[idx, idx - 1] = 1
                A[idx, idx + 1] = 1
                A[idx, idx - nx] = 2*beta    # Pega calor dobrado de baixo para compensar o topo
                b[idx] = -2*beta*Biy*Tinf
                
            # REGRA 4: Borda de baixo (perde calor pro ar)
            elif j == 0 and i < nx - 1:
                A[idx, idx] = -2*(1 + beta + beta*Biy)
                A[idx, idx - 1] = 1
                A[idx, idx + 1] = 1
                A[idx, idx + nx] = 2*beta    # Pega calor dobrado de cima
                b[idx] = -2*beta*Biy*Tinf
                
            # REGRA 5: Ponta da aleta (lado direito, perde calor pro ar)
            elif 0 < j < ny - 1 and i == nx - 1:
                A[idx, idx] = -2*(1 + beta + Bix)
                A[idx, idx - 1] = 2          # Pega calor dobrado da esquerda
                A[idx, idx - nx] = beta
                A[idx, idx + nx] = beta
                b[idx] = -2*Bix*Tinf
                
            # REGRA 6: Quina de cima à direita (perde calor por cima e pela direita)
            elif j == ny - 1 and i == nx - 1:
                A[idx, idx] = -2*(1 + beta + Bix + beta*Biy)
                A[idx, idx - 1] = 2
                A[idx, idx - nx] = 2*beta
                b[idx] = -2*(Bix + beta*Biy)*Tinf
                
            # REGRA 7: Quina de baixo à direita (perde calor por baixo e pela direita)
            elif j == 0 and i == nx - 1:
                A[idx, idx] = -2*(1 + beta + Bix + beta*Biy)
                A[idx, idx - 1] = 2
                A[idx, idx + nx] = 2*beta
                b[idx] = -2*(Bix + beta*Biy)*Tinf

    # Pede ao Python (numpy) para resolver a matriz gigantesca "A * Temperaturas = b"
    T_vetor = np.linalg.solve(A, b)
    tempo = time.time() - start_time # Calcula quantos segundos demorou
    
    # A resposta é uma lista reta. Transformamos de volta num formato de malha (grade 2D)
    T_matriz = T_vetor.reshape((ny, nx))
    return T_matriz, tempo


# ==============================================================================
# 3. FUNÇÃO: RESOLUÇÃO POR MÉTODOS ITERATIVOS (Chute e Correção)
# ==============================================================================
# Essa função serve tanto para Gauss-Seidel (se omega for 1.0) 
# quanto para Relaxação SOR (se omega for maior que 1.0)
def resolver_iterativo(nx, ny, L, H, k, h, Tb, Tinf, tol, omega):
    dx = L / (nx - 1)
    dy = H / (ny - 1)
    beta = (dx**2) / (dy**2)
    Bix = h * dx / k
    Biy = h * dy / k

    # Cria uma malha inicial onde imaginamos que a aleta inteira está fria (25 graus)
    T = np.full((ny, nx), Tinf)
    
    # Mas sabemos que a base (coluna 0) está sempre quente (Tb)
    T[:, 0] = Tb 

    iteracoes = 0        # Contador de quantas vezes repetimos o cálculo
    erro = tol + 1.0     # Força o erro inicial a ser maior que a tolerância para entrar no loop
    start_time = time.time()

    # O código vai ficar repetindo até o erro ser bem pequenininho
    while erro > tol:
        erro_max = 0.0 # Reinicia o erro máximo desta rodada
        
        # Percorre a malha toda
        for j in range(ny):
            for i in range(1, nx): # Começa do 1 porque o 0 é a base e nunca muda
                
                T_antigo = T[j, i] # Salva a temperatura atual antes de mudá-la

                # Calcula a NOVA temperatura dependendo de onde o ponto está:
                
                # 1. No meio da aleta
                if i < nx - 1 and 0 < j < ny - 1:
                    T_novo = (T[j, i-1] + T[j, i+1] + beta*(T[j-1, i] + T[j+1, i])) / (2*(1+beta))
                # 2. Na borda de cima
                elif i < nx - 1 and j == ny - 1:
                    T_novo = (T[j, i-1] + T[j, i+1] + 2*beta*T[j-1, i] + 2*beta*Biy*Tinf) / (2*(1 + beta + beta*Biy))
                # 3. Na borda de baixo
                elif i < nx - 1 and j == 0:
                    T_novo = (T[j, i-1] + T[j, i+1] + 2*beta*T[j+1, i] + 2*beta*Biy*Tinf) / (2*(1 + beta + beta*Biy))
                # 4. Na ponta direita
                elif i == nx - 1 and 0 < j < ny - 1:
                    T_novo = (2*T[j, i-1] + beta*T[j-1, i] + beta*T[j+1, i] + 2*Bix*Tinf) / (2*(1 + beta + Bix))
                # 5. Na quina de cima direita
                elif i == nx - 1 and j == ny - 1:
                    T_novo = (2*T[j, i-1] + 2*beta*T[j-1, i] + 2*(Bix + beta*Biy)*Tinf) / (2*(1 + beta + Bix + beta*Biy))
                # 6. Na quina de baixo direita
                elif i == nx - 1 and j == 0:
                    T_novo = (2*T[j, i-1] + 2*beta*T[j+1, i] + 2*(Bix + beta*Biy)*Tinf) / (2*(1 + beta + Bix + beta*Biy))

                # Mistura o valor antigo com o novo usando o fator ômega (Relaxação)
                # Se omega for 1, ele pega só o T_novo (Gauss-Seidel normal)
                T[j, i] = omega * T_novo + (1 - omega) * T_antigo

                # Verifica o quanto a temperatura mudou em porcentagem
                if T[j, i] != 0:
                    mudanca = abs((T[j, i] - T_antigo) / T[j, i])
                    if mudanca > erro_max:
                        erro_max = mudanca

        erro = erro_max # Atualiza o erro geral
        iteracoes += 1  # Conta mais uma volta

    tempo = time.time() - start_time
    return T, iteracoes, erro, tempo


# ==============================================================================
# 4. FUNÇÃO: SOLUÇÃO ANALÍTICA 1D (Para comparação)
# ==============================================================================
def calcular_analitico_1d(x_array, L, H, k, h, Tb, Tinf):
    # Fórmula teórica da aleta 1D infinita
    m = np.sqrt((2 * h) / (k * H))
    
    T_analitico = np.zeros_like(x_array) # Cria uma lista vazia do mesmo tamanho
    
    # Calcula a temperatura teórica para cada posição x
    for idx, x in enumerate(x_array):
        numerador = np.cosh(m * (L - x)) + (h / (m * k)) * np.sinh(m * (L - x))
        denominador = np.cosh(m * L) + (h / (m * k)) * np.sinh(m * L)
        theta = numerador / denominador
        T_analitico[idx] = Tinf + theta * (Tb - Tinf)
        
    return T_analitico


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL DO PROGRAMA
# ==============================================================================
# Como retiramos o 'if __name__ == "__main__":', o código roda diretamente daqui 
# para baixo assim que você mandar rodar o arquivo.

print("Resolvendo pelo método de Eliminação de Gauss...")
T_gauss, t_gauss = resolver_gauss(nx, ny, L, H, k, h, Tb, Tinf)

print("Resolvendo pelo método Iterativo (Gauss-Seidel, ômega = 1.0)...")
T_gs, iter_gs, err_gs, t_gs = resolver_iterativo(nx, ny, L, H, k, h, Tb, Tinf, tol, 1.0)

print(f"Resolvendo pelo método Iterativo com Relaxação (ômega = {omega})...")
T_sor, iter_sor, err_sor, t_sor = resolver_iterativo(nx, ny, L, H, k, h, Tb, Tinf, tol, omega)

# Imprime os resultados bonitos na tela usando formatação básica
print("\n" + "="*65)
print("COMPARAÇÃO DE RESULTADOS")
print("="*65)
print(f"Método Gauss          | Tempo: {t_gauss:.6f} s")
print(f"Método Gauss-Seidel   | Iterações: {iter_gs} | Tempo: {t_gs:.6f} s")
print(f"Método Relaxação(SOR) | Iterações: {iter_sor} | Tempo: {t_sor:.6f} s")
print("="*65 + "\n")


# ==============================================================================
# 6. GERAÇÃO DE ARQUIVOS E GRÁFICOS (Baseado na solução de Relaxação)
# ==============================================================================
print("Gerando arquivo de texto 'resultados_temperatura.txt'...")
dx_malha = L / (nx - 1)
dy_malha = H / (ny - 1)

# Abre um arquivo e escreve as colunas x, y e T
with open('resultados_temperatura.txt', 'w') as arquivo:
    arquivo.write("x(m)\ty(m)\tT(C)\n")
    for j in range(ny):
        coord_y = j * dy_malha
        for i in range(nx):
            coord_x = i * dx_malha
            arquivo.write(f"{coord_x:.4f}\t{coord_y:.4f}\t{T_sor[j,i]:.4f}\n")

print("Gerando gráficos...")

# Prepara os eixos X e Y
vetor_x = np.linspace(0, L, nx)
vetor_y = np.linspace(0, H, ny)
Malha_X, Malha_Y = np.meshgrid(vetor_x, vetor_y)

# Gráfico 1: Mapa de Calor 2D
plt.figure(figsize=(8, 4))
plt.pcolormesh(Malha_X, Malha_Y, T_sor, cmap='jet', shading='auto')
plt.colorbar(label='Temperatura (°C)')
plt.title('Mapa de Temperatura 2D da Aleta')
plt.xlabel('Posição x (m)')
plt.ylabel('Posição y (m)')
plt.tight_layout()
plt.savefig('grafico_mapa_calor.png')

# Gráfico 2: Curvas Isotérmicas (Linhas de mesma temperatura)
plt.figure(figsize=(8, 4))
contornos = plt.contour(Malha_X, Malha_Y, T_sor, levels=15, cmap='jet')
plt.clabel(contornos, inline=True, fontsize=8)
plt.title('Contornos Isotérmicos')
plt.xlabel('Posição x (m)')
plt.ylabel('Posição y (m)')
plt.tight_layout()
plt.savefig('grafico_curvas_isotermicas.png')

# Gráfico 3: Comparação da linha do meio da aleta (2D numérico vs 1D teórico)
linha_do_meio = ny // 2 # Acha o índice que fica no centro do eixo y
T_numerico_meio = T_sor[linha_do_meio, :]
T_teorico_1d = calcular_analitico_1d(vetor_x, L, H, k, h, Tb, Tinf)

plt.figure(figsize=(8, 4))
plt.plot(vetor_x, T_numerico_meio, 'b-', label='Numérico 2D (Meio da aleta)')
plt.plot(vetor_x, T_teorico_1d, 'r--', label='Teórico 1D')
plt.title('Comparação Numérico x Teórico')
plt.xlabel('Comprimento x (m)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_comparacao_1d.png')

# Calcula a diferença (erro) entre a nossa simulação e a teoria
erro_percentual = np.mean(np.abs((T_numerico_meio - T_teorico_1d) / T_teorico_1d)) * 100
print(f"Erro percentual da simulação 2D em relação à teoria 1D: {erro_percentual:.2f}%")
print("Processo concluído! Gráficos e arquivo de texto foram salvos na mesma pasta do código.")
