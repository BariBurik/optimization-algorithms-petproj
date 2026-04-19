import numpy as np
import matplotlib.pyplot as plt

from functions import f, df, ddf


# =========================
# Оптимизаторы с траекторией
# =========================

def grad_desc_path(df, learning_rate, iterations, w_0):
    w = w_0.astype(float).copy()
    path = [w.copy()]

    for _ in range(iterations):
        g = df(w)
        w = w - learning_rate * g
        path.append(w.copy())

    return np.array(path)


def momentum_path(df, learning_rate, iterations, w_0, alpha=0.1):
    w = w_0.astype(float).copy()
    path = [w.copy()]
    v = np.zeros_like(w)

    for _ in range(iterations):
        g = df(w)
        v = alpha * v + learning_rate * g
        w = w - v
        path.append(w.copy())

    return np.array(path)


def newton_path(df, ddf, learning_rate, iterations, w_0, eps=1e-6, max_step=1.0):
    w = w_0.astype(float).copy()
    path = [w.copy()]

    for _ in range(iterations):
        g = df(w)
        hess = ddf(w)
        hess_stable = hess + np.eye(len(w)) * eps

        try:
            step = np.linalg.solve(hess_stable, g)
        except np.linalg.LinAlgError:
            break

        # ограничение слишком большого шага
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step = step / step_norm * max_step

        w = w - learning_rate * step
        path.append(w.copy())

        # остановка при почти нулевом градиенте
        if np.linalg.norm(g) < 1e-6:
            break

    return np.array(path)


# =========================
# Вспомогательные функции
# =========================

def distance_to_local_min(w):
    return np.linalg.norm(w - np.array([0.0, 0.0]))


def path_values(path):
    return np.array([f(w) for w in path])


def score_path(path):
    """
    Чем меньше score, тем лучше:
    - ближе к (0,0)
    - меньше значение функции
    """
    final_point = path[-1]
    dist = distance_to_local_min(final_point)
    value = abs(f(final_point))
    return dist + 0.3 * value


def evaluate_start_point(start, gd_params, momentum_params, newton_params):
    gd_path = grad_desc_path(df, **gd_params, w_0=start)
    momentum_res = momentum_path(df, **momentum_params, w_0=start)
    newton_res = newton_path(df, ddf, **newton_params, w_0=start)

    total_score = (
        score_path(gd_path) +
        score_path(momentum_res) +
        score_path(newton_res)
    )

    return {
        "start": start,
        "gd": gd_path,
        "momentum": momentum_res,
        "newton": newton_res,
        "score": total_score
    }


def print_method_result(name, path):
    final_point = path[-1]
    print(f"{name}:")
    print(f"  конечная точка: {final_point}")
    print(f"  f(w) = {f(final_point):.6f}")
    print(f"  расстояние до (0,0) = {distance_to_local_min(final_point):.6f}")


# =========================
# Графики
# =========================

def plot_contour(best_result):
    gd_path = best_result["gd"]
    momentum_res = best_result["momentum"]
    newton_res = best_result["newton"]
    start = best_result["start"]

    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 2 * X * Y**2

    plt.figure(figsize=(10, 7))
    plt.contour(X, Y, Z, levels=30)

    plt.plot(gd_path[:, 0], gd_path[:, 1], 'o-', label='Gradient Descent')
    plt.plot(momentum_res[:, 0], momentum_res[:, 1], 'o-', label='Momentum')
    plt.plot(newton_res[:, 0], newton_res[:, 1], 'o-', label='Newton')

    plt.scatter(start[0], start[1], s=120, marker='x', label='Best random start')
    plt.scatter(0, 0, s=150, marker='*', label='Local minimum (0,0)')

    plt.title('Траектории методов оптимизации')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


def plot_convergence(best_result):
    gd_values = path_values(best_result["gd"])
    momentum_values = path_values(best_result["momentum"])
    newton_values = path_values(best_result["newton"])

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(gd_values)), gd_values, marker='o', label='Gradient Descent')
    plt.plot(range(len(momentum_values)), momentum_values, marker='o', label='Momentum')
    plt.plot(range(len(newton_values)), newton_values, marker='o', label='Newton')

    plt.title('Сходимость методов: f(w) по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('f(w)')
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# Основная программа
# =========================

def main():
    np.random.seed(42)

    num_random_points = 5

    gd_params = {
        "learning_rate": 0.001,
        "iterations": 2000
    }

    momentum_params = {
        "learning_rate": 0.001,
        "iterations": 1000,
        "alpha": 0.8
    }

    newton_params = {
        "learning_rate": 1,
        "iterations": 10,
        "eps": 1e-6,
        "max_step": 1.0
    }

    random_starts = [np.random.random(2) for _ in range(num_random_points)]

    print("Случайные стартовые точки:")
    for i, point in enumerate(random_starts, 1):
        print(f"{i}) {point}")

    results = []
    print("\nПроверка 5 случайных точек")
    print("=" * 60)

    for i, start in enumerate(random_starts, 1):
        result = evaluate_start_point(start, gd_params, momentum_params, newton_params)
        results.append(result)

        print(f"\nТочка {i}: {start}")
        print(f"Суммарная оценка: {result['score']:.6f}")
        print_method_result("Gradient Descent", result["gd"])
        print_method_result("Momentum", result["momentum"])
        print_method_result("Newton", result["newton"])

    best_result = min(results, key=lambda r: r["score"])

    print("\n" + "=" * 60)
    print("ЛУЧШАЯ СТАРТОВАЯ ТОЧКА")
    print("=" * 60)
    print(f"Старт: {best_result['start']}")
    print(f"Итоговая суммарная оценка: {best_result['score']:.6f}\n")

    print_method_result("Gradient Descent", best_result["gd"])
    print()
    print_method_result("Momentum", best_result["momentum"])
    print()
    print_method_result("Newton", best_result["newton"])

    plot_contour(best_result)
    plot_convergence(best_result)


if __name__ == "__main__":
    main()