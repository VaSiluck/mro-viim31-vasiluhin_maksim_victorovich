#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>

// Структура для хранения точки в n-мерном пространстве
struct Point {
    std::vector<int> coordinates;
};

// Функция для чтения файла
void readInputFile(const std::string& filename, int& dimensions, int& regionsCount, std::vector<std::pair<Point, Point>>& regions) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Не удалось открыть файл для чтения: " << filename << std::endl;
        exit(1);
    }

    infile >> dimensions >> regionsCount;

    for (int i = 0; i < regionsCount; ++i) {
        Point minPoint, maxPoint;
        minPoint.coordinates.resize(dimensions);
        maxPoint.coordinates.resize(dimensions);

        // Читаем минимальные координаты области
        for (int d = 0; d < dimensions; ++d) {
            infile >> minPoint.coordinates[d];
        }

        // Читаем максимальные координаты области
        for (int d = 0; d < dimensions; ++d) {
            infile >> maxPoint.coordinates[d];
        }

        regions.push_back({ minPoint, maxPoint });
    }

    infile.close();
}

// Функция для генерации случайных точек внутри заданной области
std::vector<Point> generateRandomPointsInRegion(const Point& minPoint, const Point& maxPoint, int dimensions, int pointsCount) {
    std::vector<Point> points(pointsCount);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int d = 0; d < dimensions; ++d) {
        std::uniform_real_distribution<> dis(minPoint.coordinates[d], maxPoint.coordinates[d]);
        for (int i = 0; i < pointsCount; ++i) {
            if (points[i].coordinates.size() < dimensions)
                points[i].coordinates.resize(dimensions);
            points[i].coordinates[d] = dis(gen);
        }
    }

    return points;
}

// Главная функция
int main() {
    setlocale(LC_ALL, "Russian");
    std::string inputFilename = "input.txt";
    std::string outputFilename = "output.txt";

    int dimensions;
    int regionsCount;
    std::vector<std::pair<Point, Point>> regions; // Пары минимальных и максимальных точек для каждой области

    // Читаем входной файл
    readInputFile(inputFilename, dimensions, regionsCount, regions);

    // Открываем выходной файл
    std::ofstream outfile(outputFilename);
    if (!outfile) {
        std::cerr << "Не удалось открыть файл для записи: " << outputFilename << std::endl;
        exit(1);
    }

    // Для каждой области генерируем случайные точки
    for (int i = 0; i < regionsCount; ++i) {
        // Генерируем случайное количество точек (например, от 10 до 50)
        int pointsCount = rand() % 50 + 10;

        std::vector<Point> randomPoints = generateRandomPointsInRegion(
            regions[i].first, regions[i].second, dimensions, pointsCount);

        // Записываем точки в файл
        for (const auto& point : randomPoints) {
            for (int d = 0; d < dimensions; ++d) {
                outfile << point.coordinates[d] << ' ';
            }
            outfile << '\n';
        }
    }

    outfile.close();

    std::cout << "Точки успешно сгенерированы и записаны в файл " << outputFilename << std::endl;

    return 0;
}