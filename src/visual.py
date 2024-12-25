import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Необходим для 3D визуализации
import numpy as np
import os

def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            # Преобразуем строку в список чисел
            # Предполагаем, что последним числом может быть метка. Если она есть, отбрасываем её.
            values = line.strip().split()
            if len(values) <= 1:
                print(f"Пропускаем некорректную строку: {line}")
                continue
            # Если у вас только координаты без меток, используйте:
            coords = list(map(float, values[: -1])) if len(values) > 2 else list(map(float, values))
            points.append(coords)
    return points

def read_centers(filename):
    centers = []
    if not os.path.exists(filename):
        print(f"Файл с центрами кластеров '{filename}' не найден.")
        return centers
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            centers.append(coords)
    return centers

def visualize_points(points, centers, dimensions):
    points = np.array(points)
    centers = np.array(centers)

    if dimensions == 2:
        plt.figure(figsize=(8,6))
        plt.scatter(points[:, 0], points[:, 1], c='b', label='Точки')
        if len(centers) > 0:
            plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='X', s=100, label='Центры кластеров')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Визуализация точек и центров кластеров')
        plt.grid(True)
        plt.legend()
        plt.show()

    elif dimensions == 3:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Точки')
        if len(centers) > 0:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', marker='X', s=100, label='Центры кластеров')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Визуализация точек и центров кластеров')
        ax.legend()
        plt.show()
    else:
        print(f"Визуализация для {dimensions}-мерных данных не поддерживается напрямую.")
        # Можно визуализировать только первые 2 измерения
        plt.figure(figsize=(8,6))
        plt.scatter(points[:, 0], points[:, 1], c='b', label='Точки')
        if len(centers) > 0:
            plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='X', s=100, label='Центры кластеров')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'Проекция {dimensions}-мерных точек на 2D плоскость')
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    input_filename = 'docs/output.txt'      # Имя файла с точками

    # Проверяем наличие файла с точками
    if not os.path.exists(input_filename):
        print(f"Файл с точками '{input_filename}' не найден. Пожалуйста, убедитесь, что вы сгенерировали его с помощью C++ программы.")
        return
    
    try:
        centr = int(input("Введите вариант кластеризации для центров (1-forel или 2-isodata): "))
    except ValueError:
        print("Необходимо ввести варант кластеризации!")
        return
    
    # Запрашиваем центры кластеров
    if centr == 1:
        centers_filename = 'docs/forel_centers.txt'   # Имя файла с центрами кластеров
    elif centr == 2:
        centers_filename = 'docs/isodata_centers.txt'   # Имя файла с центрами кластеров
    else:
        print("Ошибка: нет такого варианта!")
        return

    # Запрашиваем количество измерений
    try:
        dimensions = int(input("Введите количество измерений (2 или 3): "))
    except ValueError:
        print("Ошибка: необходимо ввести целое число (2 или 3).")
        return

    if dimensions not in [2, 3]:
        print("Ошибка: поддерживаются только 2 или 3 измерения.")
        return

    # Читаем точки из файла
    points = read_points(input_filename)

    # Проверяем, что точки имеют правильную размерность
    if not points:
        print("Ошибка: файл с точками пуст или не содержит допустимых данных.")
        return

    if len(points[0]) != dimensions:
        print(f"Ошибка: Размерность точек ({len(points[0])}) не совпадает с указанной ({dimensions}).")
        return

    # Читаем центры кластеров (если файл есть)
    centers = read_centers(centers_filename)
    if centers and len(centers[0]) != dimensions:
        print(f"Внимание: размерность центров кластеров ({len(centers[0])}) не совпадает с размерностью точек ({dimensions}). Центры не будут отображены.")
        centers = []

    # Визуализируем точки и центры
    visualize_points(points, centers, dimensions)

if __name__ == "__main__":
    main()
