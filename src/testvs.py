import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Необходим для 3D визуализации
import numpy as np

def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            # Преобразуем строку в список чисел
            coordinates = list(map(float, line.strip().split()))
            points.append(coordinates)
    return points

def read_weights(filename):
    weights = []
    b = 0
    with open(filename, 'r') as file:
        for line in file:
            if "Прямая Хо-Кашьяпа" in line:
                next_line = next(file).strip()
                weight_strs = next_line.split('=')[0].replace('y = ', '').split('+')
                weights = [float(w.split('*')[0].strip()) for w in weight_strs if '*' in w]
                b = float(next_line.split('=')[1].strip())
                break
    return weights, b

def visualize_points_and_line(points, dimensions, weights, b):
    points = np.array(points)
    if dimensions == 2:
        plt.scatter(points[:, 0], points[:, 1], label='Точки')
        x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
        y = -(weights[0] * x + b) / weights[1]
        plt.plot(x, y, color='red', label='Прямая Хо-Кашьяпа')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Визуализация точек и прямой Хо-Кашьяпа')
        plt.grid(True)
        plt.legend()
        plt.show()
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Точки')
        x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
        y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)
        X, Y = np.meshgrid(x, y)
        Z = -(weights[0] * X + weights[1] * Y + b) / weights[2]
        ax.plot_surface(X, Y, Z, color='red', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Визуализация точек и поверхности Хо-Кашьяпа')
        plt.legend()
        plt.show()
    else:
        print(f"Визуализация для {dimensions}-мерных данных не поддерживается напрямую.")
        plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'Проекция {dimensions}-мерных точек на 2D плоскость')
        plt.grid(True)
        plt.show()

def main():
    input_filename = 'output.txt'  # Имя файла с точками

    # Для получения количества измерений можно прочитать исходный файл или задать вручную
    dimensions = int(input("Введите количество измерений (2 или 3): "))

    # Читаем точки из файла
    points = read_points(input_filename)

    # Читаем веса и значение b из файла
    weights, b = read_weights(input_filename)

    # Проверяем, что точки имеют правильную размерность
    if not points or len(points[0]) != dimensions:
        print("Ошибка: Размерность точек не совпадает с указанной.")
        return

    # Визуализируем точки и прямую Хо-Кашьяпа
    visualize_points_and_line(points, dimensions, weights, b)

if __name__ == "__main__":
    main()