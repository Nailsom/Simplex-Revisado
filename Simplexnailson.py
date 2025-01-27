import numpy as np

def solver_simplex(cr, R, b):
    """
    Resolve um problema de programação linear usando o Simplex Revisado.

    Parâmetros:
        cr (list): Coeficientes da função objetivo.
        R (list): Restrições de igualdade.
        b (list): Lado direito das restrições.

    Retorna:
        dict: Resultado da solução, status e detalhes.
    """
    n_vars = len(cr)
    n_restricoes = len(b)

    # Índices das variáveis básicas e não básicas
    indices_B = list(range(n_vars - n_restricoes, n_vars))
    indices_N = list(range(n_vars - n_restricoes))

    A = np.array(R, dtype=float)
    B = A[:, indices_B]
    c_B = np.array(cr)[indices_B]

    iteracao = 0
    while True:
        iteracao += 1

        # Verifica se a matriz B é invertível
        if np.linalg.det(B) < 1e-10:
            return {"status": "Matriz Singular", "iteracao": iteracao, "mensagem": "Matriz de base não invertível."}

        B_inv = np.linalg.inv(B)
        x_B = np.dot(B_inv, b)

        # Verifica se a solução é viável
        if any(x < 0 for x in x_B):
            return {"status": "Inviável", "iteracao": iteracao}

        # Calcula os custos reduzidos
        N = A[:, indices_N]
        c_N = np.array(cr)[indices_N]
        y = np.dot(c_B, B_inv)
        custos_reduzidos = c_N - np.dot(y, N)

        # Verifica se a solução é ótima
        if all(custos_reduzidos >= 0):
            x = np.zeros(n_vars)
            x[indices_B] = x_B
            valor_objetivo = np.dot(cr, x)
            return {"solucao": x, "valor_objetivo": valor_objetivo, "status": "Ótima", "iteracao": iteracao}

        # Verifica se o problema é ilimitado
        direcao = np.dot(B_inv, A[:, indices_N[np.argmin(custos_reduzidos)]])
        if all(direcao <= 0):
            return {"status": "Ilimitado", "iteracao": iteracao, "mensagem": "Problema ilimitado."}

        # Determina a variável de entrada
        var_entrada = indices_N[np.argmin(custos_reduzidos)]

        # Calcula a direção de entrada
        direcao = np.dot(B_inv, A[:, var_entrada])

        # Verifica se o problema é ilimitado novamente
        if all(direcao <= 0):
            return {"status": "Ilimitado", "iteracao": iteracao, "mensagem": "Problema ilimitado."}

        # Determina a variável de saída usando a razão mínima
        razoes = [x_B[i] / direcao[i] if direcao[i] > 0 else np.inf for i in range(len(direcao))]
        var_saida = indices_B[np.argmin(razoes)]

        # Atualiza a base
        indices_B[indices_B.index(var_saida)] = var_entrada
        indices_N[indices_N.index(var_entrada)] = var_saida
        B = A[:, indices_B]
        c_B = np.array(cr)[indices_B]


def preparar_entrada(R, cr, limites):
    """
    Prepara os dados para o Simplex Revisado, adicionando variáveis de folga.

    Parâmetros:
        R (list): Restrições.
        cr (list): Coeficientes da função objetivo.
        limites (list): Limites das variáveis.

    Retorna:
        tuple: Matrizes e limites ajustados.
    """
    n_restricoes = len(R)
    matriz_identidade = np.eye(n_restricoes)

    # Adiciona variáveis de folga
    R = np.hstack((R, matriz_identidade))

    # Ajusta os coeficientes da função objetivo
    cr += [0] * n_restricoes

    # Ajusta os limites das variáveis
    limites += [(0.0, np.inf)] * n_restricoes

    return R, cr, limites


# Exemplo de uso
R = [
    [1, 2, -3],
    [-2, 0, 3],
    [1, 1, 0],
]
b = [10, 15, 8]
cr = [-2, -3, -4]
limites = [(0, np.inf), (0, np.inf), (0, np.inf)]

# Prepara os dados de entrada
R, cr, limites = preparar_entrada(R, cr, limites)

# Resolve o problema
resultado = solver_simplex(cr, R, b)

# Exibe o resultado
if resultado["status"] == "Ótima":
    print(f'Solução ótima encontrada após {resultado["iteracao"]} iterações:')
    print(f'Solução: {resultado["solucao"]}')
    print(f'Valor ótimo da função objetivo: {resultado["valor_objetivo"]}')
elif resultado["status"] == "Ilimitado":
    print(f'O problema é ilimitado. Iteração {resultado["iteracao"]}: {resultado["mensagem"]}')
elif resultado["status"] == "Inviável":
    print(f'O problema é inviável. Iteração {resultado["iteracao"]}.')
elif resultado["status"] == "Matriz Singular":
    print(f'Matriz de base singular. Iteração {resultado["iteracao"]}: {resultado["mensagem"]}')
