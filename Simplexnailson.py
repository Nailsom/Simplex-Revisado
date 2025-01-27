import numpy as np


def solver(c: list, A_eq: list, b_eq: list) -> dict:
    """
    Implementação do Algoritmo Simplex Revisado para resolver problemas de programação linear.
    Minimizar: c^T x
    Sujeito a: A_eq x = b_eq, bounds.

    Parâmetros:
        c: Coeficientes da função objetivo.
        A_eq: Matriz de restrições de igualdade.
        b_eq: Lado direito das restrições de igualdade.

    Retorna:
        Um dicionário contendo a solução, o status e outros detalhes relevantes.
    """
    # Inicializa as variáveis
    num_vars = len(c)
    num_constraints = len(b_eq)

    # Identifica os índices iniciais de base (variáveis de folga)
    B_indices = list(range(num_vars - num_constraints, num_vars))
    N_indices = list(range(num_vars - num_constraints))

    A = np.array(A_eq, dtype=float)
    B = A[:, B_indices]  # Matriz de base inicial
    c_B = np.array(c)[B_indices]
    x_B = None  # Solução básica inicial

    iteration = 0
    while True:
        iteration += 1

        # Calcula a solução básica para a base atual
        B_inv = np.linalg.inv(B)
        x_B = np.dot(B_inv, b_eq)

        # Verifica inviabilidade
        if any(x < 0 for x in x_B):
            return {"status": "Infeasible", "iteration": iteration}

        # Calcula os custos reduzidos
        N = A[:, N_indices]
        c_N = np.array(c)[N_indices]
        y = np.dot(c_B, B_inv)
        reduced_costs = c_N - np.dot(y, N)

        # Verifica otimalidade (se todos os custos reduzidos >= 0)
        if all(reduced_costs >= 0):
            x = np.zeros(num_vars)
            x[B_indices] = x_B
            objective_value = np.dot(c, x)
            return {
                "solution": x,
                "objective_value": objective_value,
                "status": "Optimal",
                "iteration": iteration,
            }

        # Determina a variável de entrada
        entering_index = np.argmin(reduced_costs)
        entering_var = N_indices[entering_index]

        # Calcula o vetor de direção
        direction = np.dot(B_inv, A[:, entering_var])

        # Verifica ilimitabilidade
        if all(direction <= 0):
            return {"status": "Unbounded", "iteration": iteration}

        # Determina a variável de saída usando a razão mínima
        ratios = [
            x_B[i] / direction[i] if direction[i] > 0 else np.inf
            for i in range(len(direction))
        ]
        leaving_index = np.argmin(ratios)
        leaving_var = B_indices[leaving_index]

        # Atualiza a base
        B_indices[leaving_index] = entering_var
        N_indices[entering_index] = leaving_var
        B = A[:, B_indices]
        c_B = np.array(c)[B_indices]


def prepara_input(A_eq: list, c: list, bounds: list) -> tuple:
    """
    Prepara a entrada para o Algoritmo Simplex Revisado adicionando variáveis de folga 
    e convertendo para a forma padrão.

    Parâmetros:
        A_eq: Matriz de restrições de igualdade.
        c: Coeficientes da função objetivo.
        bounds: Limites das variáveis (inferior e superior).

    Retorna:
        Uma tupla contendo as matrizes A_eq, c e bounds modificadas.
    """
    num_constraints = len(A_eq)

    # Adiciona variáveis de folga para converter para a forma padrão
    identidade = np.eye(num_constraints)
    A_eq = np.hstack((A_eq, identidade))

    # Adiciona coeficientes zero para as variáveis de folga na função objetivo
    c = c + [0] * num_constraints

    # Expande os limites com (0.0, inf)
    bounds = bounds + [(0.0, np.inf)] * num_constraints

    return A_eq, c, bounds


# Exemplo de uso
A_eq = [
    [1, 2, -3],
    [-2, 0, 3],
    [1, 1, 0],
]
b_eq = [10, 15, 8]
c = [-2, -3, -4]
bounds = [(0, np.inf), (0, np.inf), (0, np.inf)]

# Prepara a entrada
A_eq, c, bounds = prepara_input(A_eq, c, bounds)

# Resolve o problema
result = solver(c, A_eq, b_eq)

# Mostra os resultados
if result["status"] == "Optimal":
    print(f'Solução ótima encontrada em {result["iteration"]} iterações:')
    print(f'Solução: {result["solution"]}')
    print(f'Valor ótimo da função objetivo: {result["objective_value"]}')
elif result["status"] == "Unbounded":
    print(f'O problema é ilimitado. Detectado na iteração {result["iteration"]}.')
elif result["status"] == "Infeasible":
    print(f'O problema é inviável. Detectado na iteração {result["iteration"]}.')
