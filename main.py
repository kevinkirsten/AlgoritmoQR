"""
 Autovalores e Autovetores de Matrizes
Tridiagonais Simétricas - O Algoritmo QR
    EP1 - MAP3121
Alunos:
    Kevin Kirsten Lucas     - NªUSP 10853306 - Turma 01
    Robson Rezende da Silva - NªUSP 11263404 - Turma 01
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


class Configuracoes:
    PRECISAO = 10
    CRITERIO_PARADA = 10 ** (-6)
    MASSA = 2
    TEMPO_TOTAL = 10
    ESCALA = 0.0025
    DEPURACAO = False
    DEPURACAO_MODE = 0
    CRITERIO_ARREDONDAMENTO = 10 ** (-10)


def c(alfa, beta):
    return alfa / (np.sqrt((alfa ** 2) + (beta ** 2)))


def s(alfa, beta):
    return -beta / (np.sqrt((alfa ** 2) + (beta ** 2)))


def cria_q(ck, sk):
    return np.array([[ck, -sk], [sk, ck]])


def pega_par_de_linhas(linha_inicio, matriz):
    """
    :param matriz: [[1, 2, 0], [0, 0, 0], [2, 3, 4]]
    :param linha_inicio: 0
    :return: [[1, 2, 0], [0, 0, 0]]
    """
    return np.array([matriz[linha_inicio], matriz[linha_inicio + 1]])


def pega_par_de_colunas(coluna_inicio, matriz):
    """
    :param matriz: [[1, 2, 0], [0, 0, 0], [2, 3, 4]]
    :param coluna_inicio: 0
    :return: [[1, 2], [0, 0], [2, 3]]
    """
    matriz = np.array(matriz)
    return matriz[:, coluna_inicio:coluna_inicio + 2]


def substitui_par_de_linhas(matriz_original, idx_linha, matriz_linhas):
    if type(matriz_linhas) is np.ndarray:
        matriz_linhas = matriz_linhas.tolist()

    matriz_original[idx_linha] = matriz_linhas[0]
    matriz_original[idx_linha + 1] = matriz_linhas[1]

    return np.array(matriz_original)


def substitui_par_de_colunas(matriz_original, idx_coluna, colunas):
    if type(matriz_original) is not np.ndarray:
        matriz_original = np.array(matriz_original)
    if type(colunas) is not np.ndarray:
        colunas = np.array(colunas)

    matriz_original[:, idx_coluna] = colunas[:, 0]
    matriz_original[:, idx_coluna + 1] = colunas[:, 1]
    return matriz_original


def print_rotacoes_de_givens_linhas(iteracao, matriz_linhas, matriz_Q, matriz_QA, matriz_atualizada):
    print()
    print('>> {} par de linhas'.format(iteracao + 1))
    print(matriz_linhas)
    print('>> matriz Q_{}'.format(iteracao + 1))
    print(matriz_Q)
    print('>> multiplicação de Q_{} por A'.format(iteracao + 1))
    print(matriz_QA)
    print('>> matriz R atualizada')
    print(matriz_atualizada)


def rotacoes_de_givens_linhas(matriz):
    Q = []  # matriz Q de cada iteração
    if Configuracoes.DEPURACAO:
        print()
        print('##### rotações de givens nas linhas da matriz #####')

    for k in range(len(matriz) - 1):
        QA = []                                          # zera a matriz QA
        alfa = matriz[k][k]                              # encontra alfa
        beta = matriz[k + 1][k]                          # encontra beta
        ck = c(alfa, beta)                               # calcula ck
        sk = s(alfa, beta)                               # calcula sk
        Qk = cria_q(ck, sk)                              # encontra a matriz Qk
        Q.append(Qk)                                     # armazena a matriz Qk
        A_aux = pega_par_de_linhas(k, matriz)           # matriz auxiliar com linhas i e i+1
        QA = np.matmul(Qk, A_aux)                        # multiplicação de Qk por a_aux
        matriz = substitui_par_de_linhas(matriz, k, QA)  # atualiza a matriz original
        matriz = arredonda(matriz)                       # troca todos os números menores que 10^(-10) por 0

        if Configuracoes.DEPURACAO:
            print_rotacoes_de_givens_linhas(k, A_aux, Qk, QA, matriz)

    return Q, matriz


def print_rotacoes_de_givens_colunas(iteracao, matriz_linhas, matriz_Q_transposta, matriz_RQ, matriz_atualizada):
    print()
    print('>> {} par de colunas'.format(iteracao + 1))
    print(matriz_linhas)
    print('>> matriz Q_{} transposta'.format(iteracao + 1))
    print(matriz_Q_transposta)
    print('>> multiplicação de R por Q_{}'.format((iteracao + 1)))
    print(matriz_RQ)
    print('>> matriz R atualizada')
    print(matriz_atualizada)


def rotacoes_de_givens_colunas(R, Q):
    if Configuracoes.DEPURACAO:
        print()
        print('##### rotações de givens nas colunas da matriz #####')
    for k in range(len(R) - 1):
        Q_T = np.transpose(Q[k])                  # transpõe a matriz Q_k
        A_aux = pega_par_de_colunas(k, R)        # pega os pares de colunas da matriz R
        RQ_T = np.matmul(A_aux, Q_T)              # multiplica A_aux por Q_k transposto
        R = substitui_par_de_colunas(R, k, RQ_T)  # atualiza a matriz original
        R = arredonda(R)                          # troca todos os números menores que 10^(-10) por 0

        if Configuracoes.DEPURACAO:
            print_rotacoes_de_givens_colunas(k, A_aux, Q_T, RQ_T, R)
    return R


def princ_calcula_autovetores(iteracao, matriz_colunas, matriz_Q_transposta, matriz_VQ, matriz_atualizada):
    print()
    print('>> {} par de colunas'.format(iteracao+1))
    print(matriz_colunas)
    print('>> matriz Q_{} transposta'.format(iteracao+1))
    print(matriz_Q_transposta)
    print('>> multiplicação de V por Q_{}'.format((iteracao+1)))
    print(matriz_VQ)
    print('>> matriz V atualizada')
    print(np.array(matriz_atualizada))


def calcula_autovetores(V, Q):
    if Configuracoes.DEPURACAO:
        print()
        print('##### calculo dos autovetores #####')
    for k in range(len(Q)):
        Q_T = np.transpose(Q[k])                  # transpõe a matriz Q_k
        A_aux = pega_par_de_colunas(k, V)         # pega os pares de colunas da matriz V
        VQ_T = np.matmul(A_aux, Q_T)              # multiplica A_aux por Q_k transposto
        V = substitui_par_de_colunas(V, k, VQ_T)  # atualiza a matriz original

        if Configuracoes.DEPURACAO:
            princ_calcula_autovetores(k, A_aux, Q_T, VQ_T, V)
    return V


def convergiu(ultima_linha_matriz):
    # pega ultima linha e elemento n-1
    penultimo_elemento = ultima_linha_matriz[len(ultima_linha_matriz) - 2]
    if abs(penultimo_elemento) < Configuracoes.CRITERIO_PARADA:
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


def arredonda(matriz):
    for linha in range(len(matriz)):
        for coluna in range(len(matriz)):
            numero = abs(matriz[linha][coluna])
            if numero < Configuracoes.CRITERIO_ARREDONDAMENTO:
                matriz[linha][coluna] = 0
    return matriz


def algoritmo_QR_sem_deslocamentos(matriz):
    if Configuracoes.DEPURACAO:
        print()
        print('########## algoritmo QR ##########')
    tamanho_da_matriz_inicial = len(matriz)
    V = np.identity(tamanho_da_matriz_inicial)  # cria matriz identidade do tamanho da matriz original
    ultima_linha = len(matriz) - 1
    k = 0
    matriz_de_autovalores = matriz  # matriz final com os resultados convergidos
    Q = [None] * (len(matriz) - 1)  # [None, None, None]

    while len(matriz) > 1:
        while not convergiu(matriz[ultima_linha]):
            if Configuracoes.DEPURACAO:
                print()
                print('##########  {} iteração ##########'.format(k + 1))
            Q, matriz = rotacoes_de_givens_linhas(matriz)
            matriz = rotacoes_de_givens_colunas(matriz, Q)  # R[-1] é a última matriz R da última iteração
            V = calcula_autovetores(V, Q)
            k += 1

        matriz_de_autovalores = atualiza_matriz(matriz_de_autovalores, matriz)
        matriz = remove_ultima_linha_e_coluna(matriz)
        ultima_linha = len(matriz) - 1

    iteracoes = k
    autovalores = (np.array(matriz_de_autovalores).diagonal()).tolist()
    autovetores = V
    return autovalores, autovetores, iteracoes


def algoritmo_QR_com_deslocamento(matriz):
    if Configuracoes.DEPURACAO:
        print()
        print('########## algoritmo QR com deslocamento ##########')

    tamanho_da_matriz_inicial = len(matriz)
    V = np.identity(tamanho_da_matriz_inicial)  # cria matriz identidade do tamanho da matriz original
    ultima_linha = len(matriz) - 1
    k = 0
    matriz_de_autovalores = matriz  # matriz final com os resultados convergidos
    Q = [None] * (len(matriz) - 1)  # [None, None, None]

    while len(matriz) > 1:
        while not convergiu(matriz[ultima_linha]):
            if Configuracoes.DEPURACAO:
                print()
                print('##########        {} iteração            ##########'.format(k + 1))
            mi = calcula_mi(matriz, k)
            matriz = subtrai_da_diagonal_principal(matriz, mi)  # subtrai mi da diagonal principal
            Q, matriz = rotacoes_de_givens_linhas(matriz)
            matriz = rotacoes_de_givens_colunas(matriz, Q)      # R[-1] é a última matriz R da última iteração
            matriz = soma_na_diagonal_principal(matriz, mi)     # soma mi na diagonal principal
            V = calcula_autovetores(V, Q)
            k += 1

        matriz_de_autovalores = atualiza_matriz(matriz_de_autovalores, matriz)
        matriz = remove_ultima_linha_e_coluna(matriz)
        ultima_linha = len(matriz) - 1

    iteracoes = k
    autovalores = (np.array(matriz_de_autovalores).diagonal()).tolist()
    autovetores = V
    return autovalores, autovetores, iteracoes


def cria_matriz_letra_a(tamanho):
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


def subtrai_da_diagonal_principal(matriz, valor):
    dimensao_matriz = len(matriz)
    for i in range(dimensao_matriz):
        matriz[i][i] = matriz[i][i] - valor

    return matriz


def soma_na_diagonal_principal(matriz, valor):
    dimensao_matriz = len(matriz)
    for i in range(dimensao_matriz):
        matriz[i][i] = matriz[i][i] + valor

    return matriz


def sgn(d):
    if d >= 0:
        return 1
    else:
        return -1


def calcula_mi(matriz, k):
    if k == 0:
        return 0
    beta_n_menos_1 = matriz[len(matriz) - 1][len(matriz) - 2]             # encontra beta_n_menos_1
    alfa_n = matriz[len(matriz) - 1][len(matriz) - 1]                     # encontra alfa_n
    alfa_n_menos_1 = matriz[len(matriz) - 2][len(matriz) - 2]             # encontra alfa_n_menos_1
    d = (alfa_n_menos_1 - alfa_n) / 2                                     # calculo de d de acordo com o enunciado
    mi = alfa_n + d - sgn(d) * np.sqrt((d ** 2) + (beta_n_menos_1 ** 2))  # calculo de mi
    return mi


def calcula_ks(molas, b):
    kis = []
    if b:
        for i in range(1, molas + 1):
            k = (40 + 2 * i)
            kis.append(k)
    else:
        for i in range(1, molas + 1):
            k = (40 + 2 * ((-1) ** i))
            kis.append(k)
    return kis


def gera_matriz_ks(qtd_massas, b):

    molas = qtd_massas + 1
    k_molas = calcula_ks(molas, b)

    matriz = []
    for linha in range(qtd_massas):
        linha_aux = []
        for coluna in range(qtd_massas):
            if linha == coluna:
                k = (k_molas[linha] + k_molas[linha + 1]) / Configuracoes.MASSA
                linha_aux.append(k)
            elif linha == coluna - 1:
                k = (-k_molas[coluna]) / Configuracoes.MASSA
                linha_aux.append(k)
            elif linha == coluna + 1:
                k = (-k_molas[coluna + 1]) / Configuracoes.MASSA
                linha_aux.append(k)
            else:
                linha_aux.append(0)
        matriz.append(linha_aux)

    return np.array(matriz)


def autovetores_analitico(matriz):
    tamanho_matriz = len(matriz)
    autovetores = np.zeros((tamanho_matriz, tamanho_matriz))

    for linha in range(tamanho_matriz):
        for coluna in range(tamanho_matriz):
            j = coluna + 1
            i = linha + 1
            autovetores[linha][coluna] = np.sin((i * j * np.pi) / (tamanho_matriz + 1))

    # inverte a ordem dos autovetores para se assemelhar com os autovetores do algoritmo QR
    autovetores = np.fliplr(autovetores)
    return autovetores


def autovalores_analitico(matriz):
    tamanho_matriz = len(matriz)
    autovalores = []

    # para uma matriz 4x4 -> j varia de 1 até 4
    for j in range(1, tamanho_matriz + 1):
        autovalor = 2 * (1 - np.cos((j * np.pi) / (tamanho_matriz + 1)))
        autovalores.append(autovalor)

    # inverte a ordem dos autovalores para se assemelhar com os autovalores do algoritmo QR
    autovalores.reverse()
    return autovalores


def calcula_ys_t(t, y_0, autovalores):
    # para um dado tempo t, calcula y_1, y_2, ..., y_len(autovalores) e retorna uma lista
    lista_y_t = []

    for i in range(len(autovalores)):
        omega_0 = np.sqrt(autovalores[i])
        y = y_0[i] * np.cos(omega_0 * t)
        lista_y_t.append(y)

    return lista_y_t


def calculos_b_c(matriz, posicoes_iniciais):

    autovalores_k, autovetores_k, iteracoes_k = algoritmo_QR_com_deslocamento(matriz)

    if len(posicoes_iniciais) == 0:
        X_0_maior_frequencia = autovetores_k[:, 0]
        posicoes_iniciais = X_0_maior_frequencia

    autovetores_k_T = np.transpose(autovetores_k)
    y_0 = np.matmul(autovetores_k_T, posicoes_iniciais)

    frequencias = []
    lista_xi = []
    t = np.arange(0, Configuracoes.TEMPO_TOTAL, Configuracoes.ESCALA)
    # minor_ticks = np.arange(0, Configuracoes.TEMPO_TOTAL, Configuracoes.ESCALA/2)

    for linha in range(len(autovetores_k)):
        xi = 0
        for elemento in range(len(autovetores_k)):
            omega_0 = np.sqrt(autovalores_k[elemento])
            xi += y_0[elemento] * autovetores_k[linha][elemento] * np.cos(omega_0 * t)
        frequencias.append(omega_0)
        lista_xi.append(xi)
        titulo = 'Posição das massas em função do tempo'
        plt.plot(t, xi, label='Massa {}'.format(linha + 1))
        plt.title(titulo, fontdict=None, loc='center', pad=None)
        plt.grid(color='gainsboro', which='major', axis='both')
        plt.grid(color='gainsboro', which='minor', axis='both')
        plt.legend(loc='lower right')
        plt.minorticks_on()

    plt.ylabel('Posição [m]')
    plt.xlabel('Tempo [s]')
    plt.savefig('{}_massas.png'.format(linha + 1), dpi=300)
    plt.show()
    plt.clf()

    for i in range(len(lista_xi)):
        plt.plot(t, lista_xi[i])
        titulo = 'Posição da massa {} em função do tempo'.format(i + 1)
        plt.title(titulo, fontdict=None, loc='center', pad=None)
        plt.grid(color='gainsboro', which='major', axis='both')
        plt.grid(color='gainsboro', which='minor', axis='both')
        plt.legend(loc='lower right')
        plt.ylabel('Posição [m]')
        plt.xlabel('Tempo [s]')
        plt.minorticks_on()
        plt.savefig('massa_{}.png'.format(i + 1), dpi=300)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    print()
    print('Modos de vibração')
    print(autovetores_k)
    print()
    print('Frequências de vibração')
    print(frequencias)


def seleciona_tamanho_matriz_personalizado():
    tamanho_matriz = int(input('Digite um inteiro para a dimensão da matriz NxN: '))
    if tamanho_matriz == 0:
        print('A dimensão da matriz não pode ser zero!')
        seleciona_tamanho_matriz_personalizado()
    elif tamanho_matriz == 1:
        print('A dimensão da matriz não pode ser um!')
        seleciona_tamanho_matriz_personalizado()
    return tamanho_matriz


def seleciona_tamanho_matriz():
    print()
    print('Escolha a dimensão da matriz com diagonal principal 2 e subdiagonais -1:')
    print('1)  4x4')
    print('2)  8x8')
    print('3) 16x26')
    print('4) 32x32')
    print('5) Valor personalizado')

    opcao = int(input('Digite sua opção: '))

    if opcao == 5:
        tamanho_matriz = seleciona_tamanho_matriz_personalizado()
        return tamanho_matriz
    elif opcao == 1:
        return 4
    elif opcao == 2:
        return 8
    elif opcao == 3:
        return 16
    elif opcao == 4:
        return 32
    print('Escolha entre uma das opções disponíveis!')
    seleciona_tamanho_matriz()


def seleciona_versao_algoritmo():
    print()
    print('Selecione um dos algoritmos QR:')
    print('1) Com deslocamento espectral')
    print('2) Sem deslocamento espectral')
    algoritmo = int(input('Digite sua opção: '))
    if algoritmo == 1 or algoritmo == 2:
        return algoritmo
    print('Escolha uma das opções disponíveis!')
    seleciona_versao_algoritmo()


def letra_a():
    print()
    print('##########################################################')
    print('##                     Letra a)                         ##')
    print('##########################################################')

    tamanho_matriz = seleciona_tamanho_matriz()
    matriz = cria_matriz_letra_a(tamanho_matriz)
    algoritmo = seleciona_versao_algoritmo()

    print()
    print('Matriz original {}x{}'.format(tamanho_matriz, tamanho_matriz))
    print(np.array(matriz))
    print()
    print('Calculando ...')

    if algoritmo == 1:
        autovalores, autovetores, iteracoes = algoritmo_QR_com_deslocamento(matriz)
    else:
        autovalores, autovetores, iteracoes = algoritmo_QR_sem_deslocamentos(matriz)

    matriz_autovalores_analitico = autovalores_analitico(matriz)
    matriz_autovetores_analitico = autovetores_analitico(matriz)

    print()
    if algoritmo == 1:
        print('Autovalores com algoritmo QR com deslocamentos')
    else:
        print('Autovalores com algoritmo QR sem deslocamentos')
    print(autovalores)

    print()
    print('Autovalores calculados analiticamente')
    print(matriz_autovalores_analitico)

    print()
    if algoritmo == 1:
        print('Autovetores com algoritmo QR com deslocamentos')
    else:
        print('Autovetores com algoritmo QR sem deslocamentos')
    print(autovetores)

    print()
    print('Autovetores calculados analiticamente')
    print(matriz_autovetores_analitico)

    print()
    print('iterações: {}'.format(iteracoes))


def escolhe_posicoes_iniciais_b():
    print()
    print('Escolha uma das opções iniciais disponíveis')
    print('1) X(0) = [-2, -3, -1, -3, -1]')
    print('2) X(0) = [1, 10, -4, 3, -2]')
    print('3) X(0) correspondente ao modo de maior frequência')
    opcao = int(input('Digite sua opção: '))
    if opcao == 1:
        return [-2, -3, -1, -3, -1]
    elif opcao == 2:
        return [1, 10, -4, 3, -2]
    elif opcao == 3:
        return []

    print('Escolha uma das opções disponíveis!')
    escolhe_posicoes_iniciais_b()


def escolhe_posicoes_iniciais_c():
    print()
    print('Escolha uma das opções iniciais disponíveis')
    print('1) X(0) = [-2, -3, -1, -3, -1, -2, -3, -1, -3, -1]')
    print('2) X(0) = [ 1, 10, -4,  3, -2,  1, 10, -4,  3, -2]')
    print('3) X(0) correspondente ao modo de maior frequência')
    opcao = int(input('Digite sua opção: '))
    if opcao == 1:
        return [-2, -3, -1, -3, -1, -2, -3, -1, -3, -1]
    elif opcao == 2:
        return [1, 10, -4, 3, -2, 1, 10, -4, 3, -2]
    elif opcao == 3:
        return []

    print('Escolha uma das opções disponíveis!')
    escolhe_posicoes_iniciais_c()


def letra_b():
    print()
    print('##########################################################')
    print('##                     Letra b)                         ##')
    print('##########################################################')

    qtd_massas = 5
    matriz_k = gera_matriz_ks(qtd_massas, True)
    posicoes_iniciais = escolhe_posicoes_iniciais_b()

    calculos_b_c(matriz_k, posicoes_iniciais)



def letra_c():
    print()
    print('##########################################################')
    print('##                     Letra c)                         ##')
    print('##########################################################')

    qtd_massas = 10

    matriz_k = gera_matriz_ks(qtd_massas, False)
    posicoes_iniciais = escolhe_posicoes_iniciais_c()

    calculos_b_c(matriz_k, posicoes_iniciais)


def seleciona_letra_da_tarefa():
    print()
    print('Selecione uma das Tarefas, digite a letra desejada (a, b ou c): ')
    print('Digite 1 para a letra A')
    print('Digite 2 para a letra B')
    print('Digite 3 para a letra C')
    opcao = int(input('Digite sua opção: '))

    if opcao == 1 or opcao == 2 or opcao == 3:
        return opcao
    seleciona_letra_da_tarefa()


def main():
    # configurações numpy
    np.set_printoptions(precision=Configuracoes.PRECISAO)
    np.set_printoptions(threshold=sys.maxsize)

    tarefa = seleciona_letra_da_tarefa()
    if tarefa == 1:
        letra_a()
    elif tarefa == 2:
        letra_b()
    elif tarefa == 3:
        letra_c()

    continuar = input('Digite 1 para rodar novamente o programa ou qualquer tecla para encerrar: ')

    if continuar == '1':
        main()


if __name__ == '__main__':
    main()
