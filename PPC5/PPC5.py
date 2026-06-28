import numpy as np
import matplotlib.pyplot as plt

###############################################
##           PROGRAMA PARA CASA 5            ##
## ALUNO: LUIZ FELIPE THEODORO CARNEIRO      ##
## MATRICULA: 200023349                      ##
###############################################
'''
O objetivo central do programa é resolver numericamente a Equação de Blasius, 
que é uma equação diferencial ordinária (EDO) não linear de terceira ordem. 
Essa equação descreve o perfil de velocidades e o comportamento do escoamento 
de um fluido em uma camada limite laminar sobre uma placa plana. Como as restrições
físicas do problema estão separadas (a velocidade é zero na parede e igual à 
do escoamento livre no infinito), trata-se de um Problema de Valor de Contorno (PVC). 
O código transforma matematicamente essa EDO complexa de 3ª ordem em um sistema 
equivalente de três EDOs de 1ª ordem para possibilitar a simulação computacional.
'''
def blasius_edo(eta, y):
    """
    Sistema de EDOs de 1ª ordem derivado da Equação de Blasius.
    y[0] = f
    y[1] = f'
    y[2] = f''
    """
    dy = np.zeros(3)              #cria um array (um vetor unidimensional matemático) com exatamente 3 posições, todas preenchidas com zeros.
    dy[0] = y[1]                  # df/d_eta = f' (Compartimento que vai receber o valor da primeira derivada: f')
    dy[1] = y[2]                  # d(f')/d_eta = f''(Compartimento que vai receber o valor da segunda derivada: f'')
    dy[2] = -0.5 * y[0] * y[2]    # d(f'')/d_eta = -0.5 * f * f'' (Compartimento que vai receber o valor da terceira derivada: f''')
    return dy

def runge_kutta_4(s, d_eta, eta_max):
    """
    Integra o sistema de Blasius usando Runge-Kutta de 4ª Ordem.
    Retorna o vetor de eta e a matriz de resultados y.
    """
    n_steps = int(eta_max / d_eta)
    eta = np.linspace(0, eta_max, n_steps + 1) 
    '''
    O linspace cria a "régua" contendo todos os pontos espaciais exatos desde a parede até o infinito. 
    Assim, o código sabe exatamente em quais coordenadas "eta" ele deve calcular a velocidade e o cisalhamento daquele fluido.
    '''
    y = np.zeros((n_steps + 1, 3))
    '''
    Chama o NumPy para criar uma estrutura preenchida com zeros; 
    "n_steps + 1": Representa a quantidade de linhas da tabela;
    3: Representa a quantidade de colunas da tabela. Cada coluna 
    é dedicada a uma das três variáveis que estamos resolvendo na EDO.
    '''
    
    # Condições Iniciais na parede (eta = 0)
    y[0, 0] = 0.0  # f(0) = 0
    y[0, 1] = 0.0  # f'(0) = 0
    y[0, 2] = s    # f''(0) = s (Chute do Método do Tiro)
    
    # Loop de integração RK4
    for i in range(n_steps):
        k1 = d_eta * blasius_edo(eta[i], y[i])
        k2 = d_eta * blasius_edo(eta[i] + d_eta / 2.0, y[i] + k1 / 2.0)
        k3 = d_eta * blasius_edo(eta[i] + d_eta / 2.0, y[i] + k2 / 2.0)
        k4 = d_eta * blasius_edo(eta[i] + d_eta, y[i] + k3)
        
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
    return eta, y

def error_function(s, d_eta, eta_max):
    """
    Calcula o erro na fronteira distante.
    O objetivo é que f'(eta_max) seja igual a 1.
    """
    eta, y = runge_kutta_4(s, d_eta, eta_max)
    f_prime_inf = y[-1, 1] # Pega o último valor calculado de f'
    return f_prime_inf - 1.0

def main():
    print("=== Solucionador da Equação de Blasius (Método do Tiro) ===\n")
    
    # 1. Solicitar parâmetros ao usuário
    try:
        s0 = float(input("Digite o valor inicial do chute s = f''(0) [ex: 0.1]: "))
        d_eta = float(input("Digite o passo de integração (Delta eta) [ex: 0.01]: "))
        eta_max = float(input("Digite o valor máximo de eta [ex: 10.0]: "))
        tol = float(input("Digite a tolerância de convergência [ex: 1e-6]: "))
        max_iter = int(input("Digite o número máximo de iterações [ex: 50]: "))
        
    except ValueError:
        print("Entrada inválida. Usando valores padrão recomendados...")
        s0 = 0.1
        d_eta = 0.01
        eta_max = 10.0
        tol = 1e-6
        max_iter = 50

    # 2. Método do Tiro (Usando o Método da Secante para atualizar o chute)
    print("\nIniciando iterações do Método do Tiro...")
    
    # Precisamos de um segundo chute para iniciar a secante. 
    # Damos uma pequena perturbação no chute inicial.
    s1 = s0 + 0.05 
    
    e0 = error_function(s0, d_eta, eta_max)
    e1 = error_function(s1, d_eta, eta_max)
    #erro aqui é a diferença entre a velocidade adimensional que o chute gerou na fronteira distante e a velocidade que deveria ser (que é 1).
    #Quando o erro fica menor que a tolerância, significa que você atingiu a precisão desejada e o laço é interrompido.
    
    iter_count = 0
    s_converged = s1
    
    while abs(e1) > tol and iter_count < max_iter: 
    #O valor absoluto do erro (e1) precisa ser maior que a tolerância estipulada (tol).
    #Se por algum motivo numérico o chute inicial for muito ruim e o método começar a divergir, essa trava impede que o computador fique preso num loop infinito, encerrando o processo ao atingir o limite estipulado.
        # Fórmula da Secante para encontrar a próxima raiz
        s_next = s1 - e1 * (s1 - s0) / (e1 - e0)
        
        # Atualização para a próxima iteração
        s0, e0 = s1, e1
        s1 = s_next
        e1 = error_function(s1, d_eta, eta_max)
        s_converged = s1
        iter_count += 1
        
        print(f"Iteração {iter_count:02d} | s = {s1:.6f} | Erro = {e1:.2e}")

    # 3. Execução final com o valor convergido
    eta, y = runge_kutta_4(s_converged, d_eta, eta_max) #O objetivo aqui não é guardar o perfil completo e correto do escoamento dentro da matriz y e o vetor de posições dentro de eta.
    f = y[:, 0]
    f_prime = y[:, 1]
    f_double_prime = y[:, 2]
    '''Essas três linhas acima usam um recurso poderoso do NumPy chamado slicing (fatiamento) para desmontar a matriz em vetores individuais, facilitando a criação dos gráficos e a gravação do arquivo Excel/CSV depois.
    '''
    f_prime_max = f_prime[-1]
    '''
    No Python, quando você acessa o índice -1 de um vetor ou lista, o computador entende que você quer pegar exatamente o último elemento daquela estrutura, de trás para frente.
    Como a variável f_prime guarda todos os valores da velocidade desde a parede até o infinito, pegar o [-1] significa capturar a velocidade exata no ponto mais distante da malha (o seu eta_max)
    '''
    # 4. Determinação de eta_99 (onde f' = 0.99)
    # Procuramos o primeiro índice onde f' passa de 0.99 e interpolamos linearmente para mais precisão
    idx_99 = np.where(f_prime >= 0.99)[0][0] #A função np.where varre todo o vetor de velocidades (f_prime) e retorna as posições (índices) onde o valor é maior ou igual a 0.99. Como queremos exatamente o primeiro momento em que isso acontece, o [0][0] extrai apenas o primeiro índice dessa lista.
    eta_a, eta_b = eta[idx_99-1], eta[idx_99] #Aqui o código captura a coordenada $\eta$ imediatamente antes de bater 0.99 (ponto A) e a coordenada $\eta$ imediatamente após passar de 0.99 (ponto B).
    fp_a, fp_b = f_prime[idx_99-1], f_prime[idx_99] #O mesmo é feito para capturar as velocidades exatas nesses dois pontos vizinhos.
    
    eta_99 = eta_a + (0.99 - fp_a) * (eta_b - eta_a) / (fp_b - fp_a)
    '''
    Para não depender da sorte de o passo cair exatamente em 0.99, o código traça uma reta imaginária entre o ponto A e o ponto B. Essa linha aplica a fórmula clássica de interpolação linear (derivada da equação da reta) para prever com alta precisão qual seria a coordenada exata se a velocidade fosse cravada em 0.99.
    '''
    
    # 5. Saída de Dados no Console
    valor_teorico = 0.332057
    '''
    Esse é o valor exato para $f''(0)$ que Blasius (e posteriormente Howarth, com mais precisão) encontrou no século XX resolvendo essa mesma equação.
    '''
    erro_percentual_s = abs(s_converged - valor_teorico) / valor_teorico * 100
    '''
    Essa é a fórmula padrão de erro relativo percentual. Ela calcula a diferença absoluta entre a tensão de cisalhamento que o seu método do tiro encontrou e o valor clássico, dividindo pelo valor clássico e multiplicando por 100 para transformar em porcentagem.
    '''
    
    print("\n" + "="*50)
    print("RESULTADOS FINAIS:")
    print("="*50)
    print(f"Número de iterações do Método do Tiro : {iter_count}")
    print(f"Erro final obtido na fronteira        : {abs(e1):.2e}")
    print(f"Valor de f'(eta_max)                  : {f_prime_max:.6f}")
    print(f"Valor convergido de f''(0)            : {s_converged:.6f}")
    print(f"Valor clássico da literatura f''(0)   : {valor_teorico:.6f}")
    print(f"Erro relativo de f''(0)               : {erro_percentual_s:.4f} %")
    print("-" * 50)
    print("ANÁLISE DA CAMADA LIMITE:")
    print(f"Coeficiente de atrito (Cf * sqrt(Rex)): {2 * s_converged:.6f}")
    print(f"Posição adimensional eta_99 (C_delta) : {eta_99:.4f}")
    print(f"Valor clássico de C_delta (literatura): 4.92")
    print("="*50)

    # 6. Salvar os resultados em arquivo txt/csv
    header = "eta, f(eta), f'(eta), f''(eta)"
    data_out = np.column_stack((eta, f, f_prime, f_double_prime))
    np.savetxt("resultados_blasius.csv", data_out, delimiter=",", header=header, comments='')
    print("\nOs perfis de similaridade foram salvos no arquivo 'resultados_blasius.csv'.")

    # 7. Geração dos Gráficos
    plt.figure(figsize=(10, 6))
    plt.plot(eta, f, label=r'$f(\eta)$ (Corrente)', color='blue')
    plt.plot(eta, f_prime, label=r"$f'(\eta) = u/U_\infty$ (Velocidade)", color='red')
    plt.plot(eta, f_double_prime, label=r"$f''(\eta)$ (Cisalhamento)", color='green')
    
    # Linhas de referência para visualização da convergência e eta_99
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(eta_99, color='orange', linestyle=':', label=rf'$\eta_{{99}} \approx {eta_99:.2f}$')
    
    plt.title('Solução por Similaridade de Blasius (Camada Limite Laminar)')
    plt.xlabel(r'$\eta$ (Variável de Similaridade)')
    plt.ylabel('Valores Adimensionais')
    plt.xlim(0, eta_max)
    plt.ylim(0, max(f[-1], 1.2)) # Ajuste dinâmico do eixo Y para ver bem f(eta)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('perfis_blasius.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

