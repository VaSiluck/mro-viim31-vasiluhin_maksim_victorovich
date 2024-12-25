#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>

// Структура для хранения точки в n-мерном пространстве
struct Point {
    std::vector<int> coordinates;
};

class RegionHandler {
public:
    void readInputFile(const std::string& filename, int& dimensions, int& regionsCount, std::vector<std::pair<Point, Point>>& regions, std::vector<std::pair<Point, int>>& trainingData) {
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

            // Добавляем данные для обучения Хо-Кашьяпа
            trainingData.push_back({ minPoint, 1 });
            trainingData.push_back({ maxPoint, -1 });
        }

        infile.close();
    }

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
};

class HoKashyap {
public:
    void hoKashyapAlgorithm(const std::vector<std::pair<Point, int>>& data, std::vector<double>& weights, double& b, double learningRate, int maxIterations) {
        int n = data.size();
        int dimensions = data[0].first.coordinates.size();
        weights.resize(dimensions, 0);
        b = 0;

        std::vector<double> error(n, 1.0);
        int iteration = 0;

        while (iteration < maxIterations) {
            bool allClassifiedCorrectly = true;

            for (int i = 0; i < n; ++i) {
                double sum = 0;
                for (int d = 0; d < dimensions; ++d) {
                    sum += weights[d] * data[i].first.coordinates[d];
                }
                sum += b;
                sum *= data[i].second;

                if (sum <= 0) {
                    allClassifiedCorrectly = false;
                    for (int d = 0; d < dimensions; ++d) {
                        weights[d] += learningRate * data[i].second * data[i].first.coordinates[d];
                    }
                    b += learningRate * data[i].second;
                }
            }

            if (allClassifiedCorrectly) {
                break;
            }

            iteration++;
        }

        if (iteration == maxIterations) {
            std::cout << "Алгоритм не сошелся за " << maxIterations << " итераций." << std::endl;
        } else {
            std::cout << "Алгоритм сошелся за " << iteration << " итераций." << std::endl;
        }
    }

    void writeLineToFile(const std::string& filename, const std::vector<double>& weights, double b) {
        std::ofstream outfile(filename, std::ios::app);
        if (!outfile) {
            std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
            exit(1);
        }

        outfile << "Прямая Хо-Кашьяпа: " << std::endl;
        outfile << "y = ";
        for (size_t i = 0; i < weights.size(); ++i) {
            outfile << weights[i] << " * x" << i;
            if (i < weights.size() - 1) {
                outfile << " + ";
            }
        }
        outfile << " + " << b << " = 0" << std::endl;

        outfile.close();
    }
};

// Главная функция
int main() {
    setlocale(LC_ALL, "Russian");
    std::string inputFilename = "input.txt";
    std::string outputFilename = "output.txt";

    int dimensions;
    int regionsCount;
    std::vector<std::pair<Point, Point>> regions; // Пары минимальных и максимальных точек для каждой области
    std::vector<std::pair<Point, int>> trainingData; // Данные для обучения Хо-Кашьяпа

    RegionHandler regionHandler;
    // Читаем входной файл
    regionHandler.readInputFile(inputFilename, dimensions, regionsCount, regions, trainingData);

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

        std::vector<Point> randomPoints = regionHandler.generateRandomPointsInRegion(
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

    // Пример работы алгоритма Хо-Кашьяпа с данными из файла
    std::vector<double> weights;
    double b;
    double learningRate = 0.1;
    int maxIterations = 1000;

    HoKashyap hoKashyap;
    hoKashyap.hoKashyapAlgorithm(trainingData, weights, b, learningRate, maxIterations);
    hoKashyap.writeLineToFile(outputFilename, weights, b);

    std::cout << "Весовые коэффициенты: ";
    for (const auto& w : weights) {
        std::cout << w << ' ';
    }
    std::cout << "\nb: " << b << std::endl;

    std::cout << "Точки успешно сгенерированы и записаны в файл " << outputFilename << std::endl;

    return 0;
}