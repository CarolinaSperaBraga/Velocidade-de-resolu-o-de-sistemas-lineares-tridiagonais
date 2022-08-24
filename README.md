# Velocidade de resolucão de sistemas lineares tridiagonais

O objetivo desta atividade foi resolver sistemas lineares da forma $A \cdot x$ = b e comparar a velocidade de processamento de vários métodos para uma classe especial de sistemas lineares, cuja matriz é simétrica positiva definida (SPD) e tridiagonal.

As matrizes utilizadas nesta atividade são diagonais estritamente dominantes, ou seja, o coeficiente da diagonal em cada equação deve ser maior que a soma dos valores absolutos dos outros coeficientes da equação.

Os métodos avaliados foram os seguintes:
* Usando diretamente o comando <em>linalg.solve()</em> do <em>Python</em>;
* Decomposição LU;
* Decomposição de Cholesky;
* Eliminação de Gauss sem pivoteamento;
* Método de Gauss-Jacobi;
* Método de Gauss-Seidel.

Sendo os quatro primeiros métodos diretos e os dois últimos métodos iterativos. Os métodos de Gauss-Jacobi e Gauss-Seidel foram implementados com uma tolerância inferior a 10^(-8) como critério de parada, e  avaliados também para uma tolerância de 10^(-2).

As matrizes tridiagonais SPD foram padronizadas e avaliadas nas dimensões 64, 128, 256, 512, 1024, 2048, 4096, 8192 e 16384.

A solução do sistema Ax = b para as decomposições LU e de Cholesky foram calculadas diretamente pelas funções <em>lu_solve</em> e <em>cho_solve</em>, respectivamente, mas os códigos para decompor primeiro a matriz A em LU e A em HH^T e em seguida aplicar substituições progressivas e regressivas para encontrar o valor de x também se encontram no arquivo <em>Velocidade de resolução de sistemas lineares tridiagonais.py</em>. No mesmo arquivo também se encontra uma solução para ambas as decomposições através do comando <em>dot()</em>. O código utilizado de fato foi apenas a solução mais simples.  
