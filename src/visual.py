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
    return points33

def visualize_points(points, dimensions):
    points = np.array(points)
    if dimensions == 2:
        plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Визуализация точек')
        plt.grid(True)
        plt.show()
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Визуализация точек')
        plt.show()
    else:
        print(f"Визуализация для {dimensions}-мерных данных не поддерживается напрямую.")
        # Можно визуализировать первые 2 или 3 измерения
        plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'Проекция {dimensions}-мерных точек на 2D плоскость')
        plt.grid(True)
        plt.show()

def main():
    input_filename = 'output.txt'  # Имя файла с точками
    # Замените 'output.txt' на имя вашего файла, если оно другое

    # Для получения количества измерений можно прочитать исходный файл или задать вручную
    dimensions = int(input("Введите количество измерений (2 или 3): "))

    # Читаем точки из файла
    points = read_points(input_filename)

    # Проверяем, что точки имеют правильную размерность
    if not points or len(points[0]) != dimensions:
        print("Ошибка: Размерность точек не совпадает с указанной.")
        return

    # Визуализируем точки
    visualize_points(points, dimensions)

if __name__ == "__main__":
    main()