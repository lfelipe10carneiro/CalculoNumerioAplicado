# Programa para Casa 6 (PPC6)

O PPC6 tem como objetivo resolver numericamente o problema bidimensional de condução de calor em uma aleta retangular, utilizando o Método das Diferenças Finitas para integrar a equação de Laplace.

Partindo das restrições físicas modeladas (temperatura fixa na base e convecção nas superfícies livres), o programa cria uma malha bidimensional e monta computacionalmente o sistema linear de equações equivalente ao Problema de Valor de Contorno (PVC).

A solução explora e compara três métodos numéricos distintos para a resolução do sistema: Eliminação de Gauss (método exato), Método de Liebmann (Gauss-Seidel) e Método de Liebmann com Relaxação (SOR). O código avalia o tempo de execução, a velocidade de convergência em função do fator de relaxação (ômega) e realiza um estudo de refinamento de malha. O perfil de temperatura na linha central da aleta simulada em 2D também é validado contra a solução analítica 1D clássica.

Após o algoritmo iterar, resolver o sistema e convergir para a resposta correta, ele consolida os dados e entrega os resultados em três formatos diferentes:

- **1. Saída de Dados no Console (Terminal):** Exibe o tempo computacional de execução de cada método numérico testado e o número de iterações necessárias nos métodos de Liebmann. Apresenta também o cálculo numérico do erro percentual da simulação bidimensional em relação à teoria clássica unidimensional.
- **2. Geração de Arquivo de Dados (.txt):** O programa exporta a matriz completa de resultados, iterando sobre a malha espacial e criando o arquivo `resultados_temperatura.txt`. O arquivo contém três colunas organizadas: as coordenadas espaciais x e y, seguidas da solução de temperatura T calculada para aquele nó.
- **3. Visualização Gráfica (.png):** O algoritmo salva automaticamente uma cópia em alta resolução de três visualizações térmicas diferentes na mesma pasta do código: um gráfico de contornos isotérmicos (`grafico_curvas_isotermicas.png`), um mapa de calor bidimensional da aleta (`grafico_mapa_calor.png`) e um gráfico comparativo das curvas numéricas da linha central e analíticas 1D ao longo do comprimento x (`grafico_comparacao_1d.png`).
