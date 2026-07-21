# Atividade para Casa 6 (APC6)

O APC6 tem como objetivo estudar analiticamente o problema bidimensional de condução de calor em regime permanente em uma aleta retangular de seção transversal constante, governado pela equação de Laplace.

Como a aleta possui temperatura prescrita na base e troca convectiva com o ambiente nas demais superfícies, trata-se de um Problema de Valor de Contorno (PVC). A atividade aplica o Método das Diferenças Finitas (MDF) para discretizar manualmente a equação governante em uma malha uniforme de 4x4 nós.

O desenvolvimento inclui a formulação algébrica para os nós internos, de superfície (Robin) e de canto, culminando na montagem explícita do sistema linear matricial associado. O estudo também aborda qualitativamente a resolução desse sistema por Eliminação de Gauss, Gauss-Seidel (Liebmann) e Método de Relaxação, além de requerer um fluxograma da sequência de cálculo numérico.

O resultado final consolida a modelagem e a estrutura do problema, preparando a base algébrica necessária para a sua posterior implementação e simulação computacional.
