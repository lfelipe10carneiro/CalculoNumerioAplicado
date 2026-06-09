# Programa para casa 2

O PPC2 tem como objetivo resolver o problema já proposto no ApC 2, de forma computacional, utilizando o método de Bairstow.

Para isso, o método de Bairstow segue a seguinte lógica:
	- Dividir o polinômio f(x) por um fator quadrático genérico;
	- Utilizar Newton-Raphson para ajustar iterativamente os valores de r e s até que o resto dessa divisão seja igual a zero;
	- Extrair um par de raízes (reais ou complexas) aplicando a fórmula quadrática sobre D(x);
	- Aplicando a deflação polinomial no polinômio resultante;
	- Por fim, repetir o mesmo processo até encontrar todas as raízes.
	
Seguindo a lógica acima, o programa desenvolvido em python procurou ser capaz de resolver o problema proposto na atividade. Keep chill!

