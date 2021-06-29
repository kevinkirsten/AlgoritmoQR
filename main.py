import numpy as np
import math
import matplotlib.pyplot as plt


def c(alfa, beta):
    return alfa / (math.sqrt((alfa ** 2) + (beta ** 2)))


def s(alfa, beta):
    return -beta / (math.sqrt((alfa ** 2) + (beta ** 2)))


def cria_q(ck, sk):
    return [[ck, -sk], [sk, ck]]


def pegar_par_de_linhas(linha_inicio, matriz):
    """
    :param matriz: [[1, 2, 0], [0, 0, 0], [2, 3, 4]]
    :param linha_inicio: 0
    :return: [[1, 2, 0], [0, 0, 0]]
    """
    return np.array([matriz[linha_inicio], matriz[linha_inicio + 1]])


def pegar_par_de_colunas(coluna_inicio, matriz):
    """
    :param matriz: [[1, 2, 0], [0, 0, 0], [2, 3, 4]]
    :param coluna_inicio: 0
    :return: [[1, 2], [0, 0], [2, 3]]
    """
    matriz = np.array(matriz)
    return (matriz[:, coluna_inicio:coluna_inicio + 2]).tolist()


def substitui_par_de_linhas(matriz_original, idx_linha, linhas):
    matriz_original[idx_linha] = linhas[0]
    matriz_original[idx_linha + 1] = linhas[1]
    return matriz_original


def substitui_par_de_colunas(matriz_original, idx_coluna, colunas):
    if type(matriz_original) is not np.ndarray:
        matriz_original = np.array(matriz_original)
    if type(colunas) is not np.ndarray:
        colunas = np.array(colunas)

    matriz_original[:, idx_coluna] = colunas[:, 0]
    matriz_original[:, idx_coluna + 1] = colunas[:, 1]
    return matriz_original.tolist()


def rotacoes_de_givens_linhas(matriz):
    Q = []  # matriz Q de cada iteração

    for k in range(len(matriz) - 1):
        QA = []
        alfa = matriz[k][k]      # encontra alfa
        beta = matriz[k + 1][k]  # encontra beta
        ck = c(alfa, beta)       # calcula ck
        sk = s(alfa, beta)       # calcula sk
        Qk = cria_q(ck, sk)      # encontra a matriz Qk
        Q.append(Qk)             # armazena a matriz Qk
        A_aux = pegar_par_de_linhas(k, matriz)  # matriz auxiliar com linhas i e i+1
        QA = np.matmul(Qk, A_aux)  # multiplicação de Qk por a_aux
        matriz = substitui_par_de_linhas(matriz, k, QA.tolist())  # atualiza a matriz original

    return Q, arredondada(matriz)


def rotacoes_de_givens_colunas(R, Q):
    for k in range(len(R) - 1):
        Q_T = np.transpose(Q[k])  # transpõe a matriz Q_k
        A_aux = pegar_par_de_colunas(k, R)  # pega os pares de colunas da matriz R
        RQ_T = np.matmul(A_aux, Q_T)  # multiplica A_aux por Q_k transposto
        R = substitui_par_de_colunas(R, k, RQ_T)  # atualiza a matriz original

    return arredondada(R)


def calcula_autovetores(V, Q):
    for k in range(len(Q)):
        Q_T = np.transpose(Q[k])  # transpõe a matriz Q_k
        A_aux = pegar_par_de_colunas(k, V)  # pega os pares de colunas da matriz V
        VQ_T = np.matmul(A_aux, Q_T)  # multiplica A_aux por Q_k transposto
        V = substitui_par_de_colunas(V, k, VQ_T)  # atualiza a matriz original
    return V


def convergiu(ultima_linha_matriz, criterio_parada):
    penultimo_elemento = ultima_linha_matriz[len(ultima_linha_matriz) - 2]
    if abs(penultimo_elemento) < criterio_parada:
        return True
    return False


def remove_ultima_linha_e_coluna(matriz):
    if type(matriz) is not np.ndarray:
        matriz = np.array(matriz)
    return matriz[0:-1, 0:-1]


def atualiza_matriz(matriz_final, matriz_menor):
    for linha in range(len(matriz_menor)):
        for coluna, elemento in enumerate(matriz_menor[linha]):
            matriz_final[linha][coluna] = matriz_menor[linha][coluna]
    return matriz_final


def arredondada(matriz):
    for linha in range(len(matriz)):
        for coluna in range(len(matriz)):
            numero = abs(matriz[linha][coluna])
            if numero < (10**(-10)):
                matriz[linha][coluna] = 0
    return matriz


def algoritmo_QR(matriz, criterio_parada):
    tamanho_da_matriz_inicial = len(matriz)
    V = np.identity(tamanho_da_matriz_inicial)
    ultima_linha = len(matriz) - 1
    k = 0
    matriz_de_autovalores = matriz  # matriz final com os resultados convergidos
    Q = [None] * (len(matriz) - 1)  # [None, None, None]
    # diagonal da matriz_aux são os autovalores

    while len(matriz) > 2:
        while not convergiu(matriz[ultima_linha], criterio_parada):
            Q, matriz = rotacoes_de_givens_linhas(matriz)
            matriz = rotacoes_de_givens_colunas(matriz, Q)  # R[-1] é a última matriz R da última iteração
            V = calcula_autovetores(V, Q)
            k += 1

        matriz_de_autovalores = atualiza_matriz(matriz_de_autovalores, matriz)
        matriz = remove_ultima_linha_e_coluna(matriz)
        ultima_linha = len(matriz) - 1

    iteracoes = k
    autovalores = matriz_de_autovalores
    # autovetores = V
    # autovalores = matriz_de_autovalores.diagonal()
    # autovetores = V.diagonal()
    autovetores = 0
    return autovalores, autovetores, iteracoes


def cria_matriz_enunciado(tamanho):
    matriz = []
    for linha in range(tamanho):
        linha_aux = []
        for coluna in range(tamanho):
            if linha == coluna:
                linha_aux.append(2)
            elif linha == coluna - 1:
                linha_aux.append(-1)
            elif linha == coluna + 1:
                linha_aux.append(-1)
            else:
                linha_aux.append(0)
        matriz.append(linha_aux)
    return matriz


def main():
    # matriz = [[4, 3, 0], [3, 4, 3], [0, 3, 4]]
    criterio_parada = 10**(-6)
    matriz = cria_matriz_enunciado(4)
    autovalores, autovetores, iteracoes = algoritmo_QR(matriz, criterio_parada)

    # print('autovalores')
    print(np.array(autovalores))
    # print('autovetores')
    # print(autovetores)
    print('iterações: {}'.format(iteracoes))


if __name__ == '__main__':
    main()
