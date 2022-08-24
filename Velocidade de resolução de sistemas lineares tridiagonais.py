#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
SME0892 - Cálculo Numérico para Estatística
Trabalho #1 - Velocidade de resolução de sistemas lineares tridiagonais

-Nome: Carolina Spera Braga - Número USP: 7161740
"""

# Bibliotecas utilizadas
import numpy as np 
import time
import scipy
import scipy.linalg  
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve, lu
from matplotlib import pyplot as plt

#####################################################################################


# Matriz tridiagonal SPD

# Defina uma dimensao para sua matriz
n = 16384# exemplo de valor

# Defina valores que irao compor a estrutura tridiagonal ( Os valores nas duas ...
#linhas seguintes são exemplos )
a = -1 # valor para posições adjacentes a diagonal principal
d = 8 # valor para posições da diagonal principal (d>2*a)

# Gera um vetor de elementos unitários
e = np.ones(n)
# Constroi matriz aleatória tridiagonal e SPD de dimensão n
A = np.diag(d*e) + np.diag(a*e[range(n-1) ] , 1 ) + np.diag(a*e[range(n-1) ] , -1 )
# Gera b : vetor lado−direito do sistema Ax=b
b = (d+2*a)*e 

print(A)


#####################################################################################


# Método 1 (direto)
# linalg.solve()

# Resolução do sistema Ax=b diretamente pela função np.linalg.solve()
start = time.perf_counter()
m1 = np.linalg.solve(A,b)
print(m1)

# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)  


#####################################################################################

# Método 2 (direto)
# Decomposição LU 

# Obtenção das matrizes triangulares L e U e da matriz de permutação P
P, L, U = scipy.linalg.lu(A)

# Resolução do sistema pela Decomposição LU
start = time.perf_counter()
lu, piv = lu_factor(A)
dlu = lu_solve((lu, piv), b)
print(dlu)

# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)

#####################################################################################

# Método 3 (direto)
# Decomposição de Cholesky

# Obtenção das matrizes L e U da decomposição de Cholesky
L = scipy.linalg.cholesky(A, lower=True)
U = scipy.linalg.cholesky(A, lower=False)

# Resolução do sistema Ax=b pela decomposição de Cholesky
start = time.perf_counter()
c, low = cho_factor(A)
dc = cho_solve((c, low), b)

print(dc)

# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)

#####################################################################################

# Método 4 (direto)
# Eliminação de Gauss

# Definição da função Eliminacao_de_Gauss()
def Eliminacao_de_Gauss(a,b):
    n = len(b)
    # Fase da eliminação
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                # Definimos λ não nulo
                lam = a [i,k]/a[k,k]
                # Calculamos a nova linha da matriz
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                # Atualizamos o vetor b
                b[i] = b[i] - lam*b[k]
                # Aplicamos a substituição regressiva
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    
    return b


# Resolução do sistema pela Eliminação de Gauss
start = time.perf_counter()
eg = Eliminacao_de_Gauss(A,b)
print(eg)

# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)

#####################################################################################

# Método 5 (iterativo)
# Método de Gauss-Jacobi, tol = 0.00000001

# Definição da função GaussJacobi()
def GaussJacobi(A, b, tolerancia=0.00000001, iteracoes_max=10000):
    
    x = np.zeros_like(b, dtype=np.double)
    T = A - np.diag(np.diagonal(A))
    
    for k in range(iteracoes_max):
        x_inicio = x.copy()
        # Define o sistema linear, isolando x, para resolver o sistema 
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        
        # Imposição do limite da tolerância
        if np.linalg.norm(x - x_inicio, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerancia:
            break
    return x


# Resolução do sistema pelo método de Gauss-Jacobi
start = time.perf_counter()
mj = GaussJacobi(A,b,0.00000001,iteracoes_max=10000)
print(mj)

# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start

print ("Tempo de execução: ", tempo)

#####################################################################################

# Método 6 (iterativo)
# Método de Gauss-Seidel, tol = 0.00000001

# Definição da função GaussSeidel()
def GaussSeidel(A, b, tol = 0.00000001, iteracoes_max=10000):
    
    x = np.zeros_like(b, dtype=np.double)
    
    # Iterações
    for k in range(iteracoes_max):
        
        x_inicio  = x.copy()
        
        # Loop sobre as colunas
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_inicio[(i+1):])) / A[i ,i]
            
        # Imposição do limite de tolerância
        if np.linalg.norm(x - x_inicio, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tol:
            break
            
    return x


# Resolução do sistema pelo método de Gauss-Seidel
start = time.perf_counter()    
gs = GaussSeidel(A,b,0.00000001,10000)
print(gs)


# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)

#####################################################################################

# Valor da ordem de custo computacional 

# Valores das ordens das matrizes
N = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Tempos médios calculados para cada método e cada ordem da matriz nxn

# Tempo médio para o método 1 - linalg.solve()
T1 = [0.0001921046499910517,0.0006898578650066156,0.0013864797150017693,0.008874163679993217,     0.03197338856999522,0.20201188537993403,1.2769707248000668,7.641260387000147,     64.40243209340005]

# Tempo médio para o método 2 - Decomposição LU 
T2 = [0.0012132573899998533,0.00242663662000723,0.005673358900021412,0.014560512780017234,     0.03268148394997297,0.18766518054006157,1.0113703232997977,7.655640923199826,     65.81264940740002]

# Tempo médio para o método 3 - Decomposição de Cholesky
T3 = [0.00018077013997299217,0.0003805997000290517,0.0012748461799947108,0.005028755940029442,     0.026828742230013632,0.1399928940799873,0.6463830618999964,4.323501099600071,     31.655622658599896]

# Tempo médio para o método 4 - EliminTempo médio de resolução do sistema Ax=b para diferentes ordens da matriz para os métodos iterativosação de Gauss
T4 = [0.0006778641900018556,0.0018419723400029397,0.006294993719980085,0.02512737528001253,     0.10253752347998671,0.4373132398900179,1.9326951844999711,8.213907051399838,     41.090669211999554]

# Tempo médio para o método 5 - Método de Gauss-Jacobi com limite de tolerância 10^{-8}
T5 = [0.0003070184400030485,0.0009031618100061678,0.0014913821499976622,0.005344070600012856,      0.017952960649988654,0.07722907196998677,0.28699872609998694,0.9679337144999636,      3.6405114109999888]

# Tempo médio para o método 6 - Método de Gauss-Seidel com limite de tolerância 10^{-8}
T6 = [0.0018850548200134653,0.0036309085800189677,0.007568721470011042,0.016508049189999385,      0.039207016900002147,0.09635848425999484,0.2664243551400068,0.7956683589999557,      5.039535914399972]

# Tempo médio para o método 5 - Método de Gauss-Jacobi com limite de tolerância 10^{-2}
T5_tol = [0.0002379016999839223,0.0006279738998273387,0.0010351422200983507,0.0014772176000406034,      0.013801567259979492,0.04613186570994003,0.17457884327995998,0.5399478850002197,      1.910320652280243]

# Tempo médio para o método 6 - Método de Gauss-Seidel com limite de tolerância 10^{-2}
T6_tol = [0.0014626008998675389,0.001702594299968041,0.002939601870057231,0.0061849438801436915,      0.015512678840023,0.0391521626400754,0.10197405670003717,0.3168618640997011,      1.9288013095200223]



# Plano 4 : Cálculo baseado nos quatro últimos pares ordenados
m = len(N)
ordem_custo1 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T1[m-4:m])),1)
ordem_custo2 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T2[m-4:m])),1)
ordem_custo3 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T3[m-4:m])),1)
ordem_custo4 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T4[m-4:m])),1)
ordem_custo5 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T5[m-4:m])),1)
ordem_custo6 = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T6[m-4:m])),1)
ordem_custo5tol = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T5_tol[m-4:m])),1)
ordem_custo6tol = np.polyfit(np.log(np.array(N[m-4:m])),np.log(np.array(T6_tol[m-4:m])),1)

print("O valor da ordem de custo computacional foi de: \n")
print(ordem_custo1[0], "para o Método 1 - linalg.solve()")
print(ordem_custo2[0], "para o Método 2 - Decomposição LU ")
print(ordem_custo3[0], "para o Método 3 - Decomposição de Cholesky")
print(ordem_custo4[0], "para o Método 4 - Eliminação de Gauss")
print(ordem_custo5[0], "para o Método 5 - Método de Gauss-Jacobi com limite de tolerância 10^{-8}")
print(ordem_custo6[0], "para o Método 6 - Método de Gauss-Seidel com limite de tolerância 10^{-8}")
print(ordem_custo5tol[0], "para o Método 5 - Método de Gauss-Jacobi com limite de tolerância 10^{-2}")
print(ordem_custo6tol[0], "para o Método 6 - Método de Gauss-Seidel com limite de tolerância 10^{-2}")


#####################################################################################
#####################################################################################


"""Pergunta 2: Seguindo a tendência de crescimento observada de cada método, 
estime quanto tempo cada método demoraria para resolver 
um sistema desses com 1 milhão de incógnitas no seu computador? E com 1 bilhão de incógnitas?"""

#  N = 1 milhão
"""
linalg.solve(): 338996.3697353939 s =~ 3.92 dias
Decomposição LU: 349844.0846355921 s =~ 4.05 dias
Decomposição de Cholesky: 161079.2439461761 s =~ 1.86 dias
Eliminação de Gauss: 176986.5722257182 s =~ 49.16 h =~ 2.05 dias
Método de Gauss-Jacobi com limite de tolerância 10^{-8}: 212.13160851319358 s =~ 3.54 minutos
Método de Gauss-Seidel com limite de tolerância 10^{-8}: 282.65001967085493 s =~ 4.71 minutos
Método de Gauss-Jacobi com limite de tolerância 10^{-2}: 111.90146980587357 s =~ 1.87 minutos
Método de Gauss-Seidel com limite de tolerância 10^{-2}: 108.38082680796816 s =~1.81 minutos
"""

#  N = 1 bilhão
"""
linalg.solve(): 340722624026.168 s =~ 10796.85 anos
Decomposição LU: 351665863039.3879 s =~ 11143.62 anos
Decomposição de Cholesky: 161829479651.0073 s =~ 5128.07 anos
Eliminação de Gauss: 177403317702.246 s =~ 5621.57 anos
Método de Gauss-Jacobi com limite de tolerância 10^{-8}: 212347.60915898334 s =~ 2.46 dias
Método de Gauss-Seidel com limite de tolerância 10^{-8}: 282981.67175194784 s =~ 3.28 dias
Método de Gauss-Jacobi com limite de tolerância 10^{-2}: 112009.69587011427 s =~ 1.3 dias
Método de Gauss-Seidel com limite de tolerância 10^{-2}: 108506.70252930386 s =~ 1.26 dias
"""

# Encontramos a função polinomial aproximada para cada método
model = np.polyfit(N, T6, 1.87)
predict = np.poly1d(model)
# Fazemos a previsão aproximada do valor do tempo para determinado N
x_value = 1000000000 # Valor de N
predict(x_value) # Valor do tempo


#####################################################################################


"""Códigos para substituições progressivas e regressivas"""

# Função para substituição progressiva
def sub_progressiva(L,b):
    #L: matriz triangular inferior
    #b: termo independente
    #x: vetor solução
    n = len(b)
    x = np.zeros([n,1])


    x[0] = b[0]/L[0, 0]
    C = np.zeros((n,n))
    for i in range(1,n):
        bb = 0
        for j in range(n):
            bb += L[i, j]*x[j]

        C[i, i] = b[i] - bb
        x[i] = C[i, i]/L[i, i]

    return x
    
    
# Função para substituição regressiva    
def sub_regressiva(U,b):
    #U: matriz triangular superior
    #b: termo independente
    #x: vetor solução
    n = len(b)
    x = np.zeros([n,1])


    x[n-1] = b[n-1]/U[n-1, n-1]
    C = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += U[i, j]*x[j]

        C[i, i] = b[i] - bb
        x[i] = C[i, i]/U[i, i]

    return x     


"""Outros códigos testados para a Decomposição LU + substituições progressivas e regressivas,
e Decomposição LU + função .dot()"""

# Decomposição LU
p, L, U = lu(A)
b = n*[0]

# Aplicação de substituições progressivas e regressivas para obtenção de x
y = sub_progressiva(L,b)
x = sub_regressiva(U,y)

# Aplicação da função .dot() para obtenção de x
y1 = np.linalg.inv(L).dot(b)
x1 = np.linalg.inv(U).dot(y)


"""Outros códigos testados para a Decomposição de Cholesky + substituições progressivas e regressivas,
e Decomposição de Cholesky + função .dot()"""

# Decomposição de Cholesky
H = np.linalg.cholesky(A)
HT = np.transpose(H)
b = n*[0]

# Aplicação de substituições progressivas e regressivas para obtenção de x
y = sub_progressiva(H,b)
x = sub_regressiva(HT,y)

# Aplicação da função .dot() para obtenção de x
y1 = np.linalg.inv(H).dot(b)
x1 = np.linalg.inv(HT).dot(y)


#####################################################################################
#####################################################################################


"""Outros códigos testados para os métodos de Gauss-Jacobi e Gauss-Seidel, mas que apresentaram 
pior desempenho.
Aqui estão, inclusive, os códigos passados em aula em Matlab convertidos para Python."""

# Definição da função GaussSeidel2()
def GaussSeidel2(A, b, tolerancia=0.00000001, iteracoes_max=1000):
    
    x = np.zeros_like(b, dtype=np.double)
    T = A - np.diag(np.diagonal(A))
    
    for i in range(iteracoes_max):
        x0 = x.copy()
        for j in range(n):
            x0[j] = x[j]
        # Define o sistema linear, isolando x, para resolver o sistema 
            x[:] = (b - np.dot(T, x0)) / np.diagonal(A)
        
        # Imposição do limite da tolerância
        if np.linalg.norm(x - x0, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerancia:
            break
    return x

    
# Resolução do sistema pelo método de Gauss-Seidel
start = time.perf_counter()    
gs = GaussSeidel2(A, b, tolerancia=0.00000001, iteracoes_max=1000)
print(gs)


# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)   


################################################################################

# Definição da função GaussSeidel3()
def GaussSeidel3(A, b, x, N, tol):
    
    iteracoes_max = 1000
    # Define o chute inicial x0
    x0 = [0.0 for i in range(N)]
    l = []
    
    for i in range(iteracoes_max):
        for j in range(N):
            x0[j] = x[j]
            soma = 0.0
            for k in range(N):
                if (k != j):
                    # Organização do sistema linear para isolar o x
                    soma = soma + A[j][k] * x[k]
            # Define o sistema linear com as substituições adequadas para resolver o sistema        
            x[j] = (b[j] - soma) / A[j][j]
        dn = 0.0
        on = 0.0
        for j in range(N):
            dn = dn + abs(x[j] - x0[j])
            on = on + abs(x0[j])  
        if on == 0.0:
            on = 1.0
        norm = dn / on
        # Imposição do limite da tolerância        
        if (norm < tol) and i != 0:
            for j in range(N - 1):
#                 print(x[j], ",", end="")
                l.append(x[j])
            print(l)
#             print(x[N - 1], "]. Levou", i + 1, "iterações.")
            return 
#     print("Não converge.")

    
# Resolução do sistema pelo método de Gauss-Seidel
x = n*[0]
start = time.perf_counter()    
gs = GaussSeidel3(A, b, x, n, 0.00000001)


# Contagem do tempo de resolução do sistema Ax=b em segundos
end = time.perf_counter()
tempo = end - start
print ("Tempo de execução: ", tempo)


################################################################################
# Função Gauss-Jacobi dada em aula

def Gauss_Jacobi(A,b,x0,tol):

    D = np.diag((np.diagonal(A)))
    c = np.eye(n) - np.linalg.inv(D) * A
    g = np.linalg.inv(D) * b
    kmax = 10000
    k = 0
    
    while np.linalg.norm(b - A * x0) > tol and k in range(kmax):
        k += 1
        x0 = c*x0 + g
    x = x0
    print(np.diagonal(x))
    
x0 = n*[0]    
start = time.perf_counter()
Gauss_Jacobi(A,b,x0,0.00000001)    

end = time.perf_counter()
tempo = end - start

print ("Tempo de execução: ", tempo)


################################################################################
# Função Gauss-Seidel dada em aula

def Gauss_Seidel(A,b,x0,tol):
    L = np.tril(A)
    R = np.triu(A,1)
    c = -np.linalg.inv(L) * R
    g = np.linalg.inv(L) * b
    kmax = 10000
    k = 0
    
    while np.linalg.norm(b - A*x0) > tol and k < kmax:
        k += 1
        x0 = c*x0 + g
    x = x0
    print(np.diagonal(x))

start = time.perf_counter()
Gauss_Seidel(A,b,x0,0.00000001)    

end = time.perf_counter()
tempo = end - start

print ("Tempo de execução: ", tempo)

