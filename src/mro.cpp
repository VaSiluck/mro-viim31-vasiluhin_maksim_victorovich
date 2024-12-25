#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <locale>
#include <algorithm>
#include <filesystem>

// Класс для хранения точки в n-мерном пространстве
class Point {
public:
    std::vector<double> coordinates;

    // Конструктор: инициализируем точку с заданной размерностью
    Point(int dimensions = 0) : coordinates(dimensions) {}

    // Вычисление Евклидова расстояния до другой точки
    double euclideanDistance(const Point& other) const {
        double sum = 0.0;
        for (size_t d = 0; d < coordinates.size(); ++d) {
            double diff = coordinates[d] - other.coordinates[d];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    // Вычисление Манхэттенского расстояния до другой точки (не используется в данном коде, но оставлено для демонстрации)
    double manhattanDistance(const Point& other) const {
        double sum = 0.0;
        for (size_t d = 0; d < coordinates.size(); ++d) {
            sum += fabs(coordinates[d] - other.coordinates[d]);
        }
        return sum;
    }
};

// Класс для хранения точки с меткой класса
class LabeledPoint {
public:
    Point point;
    int label; // Метка: +1 или -1

    // Конструктор: инициализируем точку и её метку
    LabeledPoint(const Point& p = Point(), int l = 0) : point(p), label(l) {}
};

// Класс для работы с данными: чтение областей, генерация точек, присваивание меток, запись в файл
class DataSet {
public:
    int dimensions; // Размерность пространства
    int regionsCount; // Количество областей
    std::vector<std::pair<Point, Point>> regions; // Вектор пар (minPoint, maxPoint)
    std::vector<LabeledPoint> dataset; // Датасет с метками

    // Чтение параметров из файла
    void readInputFile(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Не удалось открыть файл для чтения: " << filename << std::endl;
            exit(1);
        }

        // Формат входного файла: dimensions, regionsCount, затем для каждой области minPoint и maxPoint
        infile >> dimensions >> regionsCount;

        for (int i = 0; i < regionsCount; ++i) {
            Point minPoint(dimensions), maxPoint(dimensions);
            for (int d = 0; d < dimensions; ++d) {
                infile >> minPoint.coordinates[d];
            }
            for (int d = 0; d < dimensions; ++d) {
                infile >> maxPoint.coordinates[d];
            }
            regions.push_back({ minPoint, maxPoint });
        }

        infile.close();
    }

    // Генерация случайных точек в заданных областях и запись их в выходной файл
    void generateRandomPoints(const std::string& outputFilename) {
        std::ofstream outfile(outputFilename);
        if (!outfile) {
            std::cerr << "Не удалось открыть файл для записи: " << outputFilename << std::endl;
            exit(1);
        }

        // Для каждой области генерируем точки
        for (int i = 0; i < regionsCount; ++i) {
            // Первой области присваиваем метку +1, остальным -1
            int label = (i == 0) ? 1 : -1;

            // Генерируем количество точек от 50 до 100
            int pointsCount = rand() % 51 + 50;

            // Генерируем точки равномерно в пределах minPoint и maxPoint
            std::vector<Point> randomPoints = generateRandomPointsInRegion(
                regions[i].first, regions[i].second, dimensions, pointsCount);

            // Записываем точки в файл и датасет
            for (const auto& point : randomPoints) {
                for (int d = 0; d < dimensions; ++d) {
                    outfile << point.coordinates[d] << ' ';
                }
                outfile << label << '\n';

                dataset.push_back({ point, label });
            }
        }

        outfile.close();
        std::cout << "Точки успешно сгенерированы и записаны в файл " << outputFilename << std::endl;
    }

private:
    // Генерация случайных точек внутри заданной области
    std::vector<Point> generateRandomPointsInRegion(const Point& minPoint, const Point& maxPoint, int dimensions, int pointsCount) {
        std::vector<Point> points(pointsCount, Point(dimensions));
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int d = 0; d < dimensions; ++d) {
            std::uniform_real_distribution<> dis(minPoint.coordinates[d], maxPoint.coordinates[d]);
            for (int i = 0; i < pointsCount; ++i) {
                points[i].coordinates[d] = dis(gen);
            }
        }

        return points;
    }
};

// Класс для работы с матрицами (используется алгоритмом Хо-Кашьяпа)
class Matrix {
public:
    typedef std::vector<std::vector<double>> MatrixData;
    MatrixData data;

    Matrix() {}
    Matrix(int rows, int cols, double init = 0.0) : data(rows, std::vector<double>(cols, init)) {}

    // Транспонирование матрицы
    static Matrix transpose(const Matrix& M) {
        int rows = (int)M.data.size();
        int cols = (int)M.data[0].size();
        Matrix M_T(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                M_T.data[j][i] = M.data[i][j];
        return M_T;
    }

    // Умножение матриц
    static Matrix multiply(const Matrix& A, const Matrix& B) {
        int rowsA = (int)A.data.size();
        int colsA = (int)A.data[0].size();
        int rowsB = (int)B.data.size();
        int colsB = (int)B.data[0].size();

        if (colsA != rowsB) {
            std::cerr << "Размеры матриц не совпадают для умножения." << std::endl;
            exit(1);
        }

        Matrix C(rowsA, colsB, 0.0);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j)
                for (int k = 0; k < colsA; ++k)
                    C.data[i][j] += A.data[i][k] * B.data[k][j];

        return C;
    }

    // Обратная матрица (для небольших матриц)
    static Matrix inverse(const Matrix& M) {
        int n = (int)M.data.size();
        Matrix A = M;
        Matrix I(n, n);
        for (int i = 0; i < n; ++i)
            I.data[i][i] = 1.0;

        // Метод Гаусса для обращения матрицы
        for (int i = 0; i < n; ++i) {
            double pivot = A.data[i][i];
            if (fabs(pivot) < 1e-12) {
                std::cerr << "Матрица вырождена и не может быть обращена." << std::endl;
                exit(1);
            }
            for (int j = 0; j < n; ++j) {
                A.data[i][j] /= pivot;
                I.data[i][j] /= pivot;
            }
            for (int k = 0; k < n; ++k) {
                if (k == i) continue;
                double factor = A.data[k][i];
                for (int j = 0; j < n; ++j) {
                    A.data[k][j] -= factor * A.data[i][j];
                    I.data[k][j] -= factor * I.data[i][j];
                }
            }
        }
        return I;
    }
};

// Алгоритм Хо-Кашьяпа для линейной классификации
class HoKashyapAlgorithm {
public:
    std::vector<double> w; // Вектор весов

    void train(const std::vector<LabeledPoint>& dataset) {
        int N = (int)dataset.size();
        int D = (int)dataset[0].point.coordinates.size() + 1; // Добавляем размерность для смещения (bias)

        Matrix A(N, D);
        std::vector<double> b_vec(N, 1.0);
        std::vector<double> e(N, 0.0);
        w.resize(D, 0.0);

        // Формируем матрицу A согласно алгоритму Хо-Кашьяпа
        for (int i = 0; i < N; ++i) {
            int y_i = dataset[i].label;
            for (int d = 0; d < D - 1; ++d) {
                A.data[i][d] = y_i * dataset[i].point.coordinates[d];
            }
            A.data[i][D - 1] = y_i * 1.0;
        }

        double delta = 0.1;
        int max_iter = 1000;
        double epsilon = 1e-3;

        for (int iter = 0; iter < max_iter; ++iter) {
            Matrix A_T = Matrix::transpose(A);
            Matrix A_T_A = Matrix::multiply(A_T, A);
            Matrix A_T_A_inv = Matrix::inverse(A_T_A);
            Matrix A_pinv = Matrix::multiply(A_T_A_inv, A_T);

            // Вычисляем веса w
            Matrix b_matrix(N, 1);
            for (int i = 0; i < N; ++i) {
                b_matrix.data[i][0] = b_vec[i];
            }
            Matrix w_matrix = Matrix::multiply(A_pinv, b_matrix);
            for (int d = 0; d < D; ++d) {
                w[d] = w_matrix.data[d][0];
            }

            // Вычисляем ошибку e
            for (int i = 0; i < N; ++i) {
                double sum = 0.0;
                for (int d = 0; d < D; ++d) {
                    sum += A.data[i][d] * w[d];
                }
                e[i] = sum - b_vec[i];
            }

            // Обновляем b_vec
            bool all_b_positive = true;
            double e_norm = 0.0;
            for (int i = 0; i < N; ++i) {
                double adjustment = delta * (e[i] + fabs(e[i])) / 2.0;
                b_vec[i] += adjustment;
                e_norm += e[i] * e[i];
                if (b_vec[i] <= 0)
                    all_b_positive = false;
            }
            e_norm = sqrt(e_norm);

            if (all_b_positive && e_norm < epsilon) {
                std::cout << "Алгоритм Хо-Кашьяпа сошелся за " << iter + 1 << " итераций." << std::endl;
                break;
            }
            if (iter == max_iter - 1) {
                std::cout << "Алгоритм Хо-Кашьяпа не сошелся за максимальное количество итераций." << std::endl;
            }
        }
    }

    // Сохранение вектора весов
    void saveWeights(const std::string& filename) {
        std::ofstream weightsFile(filename);
        if (weightsFile.is_open()) {
            for (double weight : w) {
                weightsFile << weight << " ";
            }
            weightsFile << std::endl;
            weightsFile.close();
            std::cout << "Вектор весов (Хо-Кашьяпа) записан в файл " << filename << std::endl;
        } else {
            std::cerr << "Не удалось открыть файл " << filename << " для записи." << std::endl;
        }
    }

    // Предсказание метки класса для точки
    int predict(const Point& point) {
        double sum = 0.0;
        for (size_t d = 0; d < point.coordinates.size(); ++d) {
            sum += w[d] * point.coordinates[d];
        }
        sum += w[w.size() - 1]; // Добавляем смещение
        return (sum >= 0) ? 1 : -1;
    }
};

// Класс для алгоритмов кластеризации (FOREL и упрощенный ISODATA)
class ClusteringAlgorithms {
public:
    // Алгоритм FOREL: находит центры кластеров, основываясь на радиусе R
    std::vector<Point> FOREL(const std::vector<Point>& points, double R) {
        std::vector<Point> remaining = points;
        std::vector<Point> clusterCenters;

        while (!remaining.empty()) {
            Point currentCenter = remaining[rand() % remaining.size()];

            while (true) {
                std::vector<Point> cluster;
                // Находим все точки в радиусе R
                for (auto& p : remaining) {
                    if (currentCenter.euclideanDistance(p) <= R) {
                        cluster.push_back(p);
                    }
                }

                if (cluster.empty()) break;

                // Считаем новый центр
                Point newCenter((int)currentCenter.coordinates.size());
                for (auto& cp : cluster) {
                    for (size_t d = 0; d < newCenter.coordinates.size(); ++d) {
                        newCenter.coordinates[d] += cp.coordinates[d];
                    }
                }
                for (size_t d = 0; d < newCenter.coordinates.size(); ++d) {
                    newCenter.coordinates[d] /= cluster.size();
                }

                double dist = currentCenter.euclideanDistance(newCenter);
                currentCenter = newCenter;
                // Если изменение центра мало, фиксируем кластер
                if (dist < 1e-7) {
                    for (auto& cp : cluster) {
                        removePoint(remaining, cp);
                    }
                    clusterCenters.push_back(currentCenter);
                    break;
                }
            }
        }

        return clusterCenters;
    }

    // Упрощенный ISODATA: начинаем с K кластеров, назначаем точки, пересчитываем центры
    std::vector<Point> ISODATA(std::vector<Point> points, int K, int maxIter = 100) {
        if ((int)points.size() < K) {
            std::cerr << "Число точек меньше количества кластеров." << std::endl;
            return {};
        }

        // Инициализируем центры кластеров случайно
        std::vector<Point> centers;
        for (int i = 0; i < K; ++i) {
            centers.push_back(points[rand() % points.size()]);
        }

        for (int iter = 0; iter < maxIter; ++iter) {
            // Распределяем точки по ближайшим центрам
            std::vector<std::vector<Point>> clusters(K);
            for (auto& p : points) {
                int nearest = findNearestCenter(p, centers);
                clusters[nearest].push_back(p);
            }

            bool changed = false;
            // Пересчитываем центры
            for (int i = 0; i < K; ++i) {
                if (clusters[i].empty()) {
                    centers[i] = points[rand() % points.size()];
                    changed = true;
                    continue;
                }

                Point newCenter((int)points[0].coordinates.size());
                for (auto& cp : clusters[i]) {
                    for (size_t d = 0; d < newCenter.coordinates.size(); ++d) {
                        newCenter.coordinates[d] += cp.coordinates[d];
                    }
                }
                for (size_t d = 0; d < newCenter.coordinates.size(); ++d) {
                    newCenter.coordinates[d] /= clusters[i].size();
                }

                double dist = centers[i].euclideanDistance(newCenter);
                if (dist > 1e-7) {
                    changed = true;
                    centers[i] = newCenter;
                }
            }

            if (!changed) break;
        }

        return centers;
    }

private:
    // Удаление точки из вектора
    void removePoint(std::vector<Point>& points, const Point& p) {
        points.erase(std::remove_if(points.begin(), points.end(), [&](const Point& pt) {
            if (pt.coordinates.size() != p.coordinates.size()) return false;
            for (size_t i = 0; i < p.coordinates.size(); ++i) {
                if (fabs(pt.coordinates[i] - p.coordinates[i]) > 1e-12) {
                    return false;
                }
            }
            return true;
        }), points.end());
    }

    // Нахождение ближайшего центра для точки
    int findNearestCenter(const Point& p, const std::vector<Point>& centers) {
        int index = 0;
        double minDist = p.euclideanDistance(centers[0]);
        for (int i = 1; i < (int)centers.size(); ++i) {
            double dist = p.euclideanDistance(centers[i]);
            if (dist < minDist) {
                minDist = dist;
                index = i;
            }
        }
        return index;
    }
};

// Простой перцептрон для линейной классификации
class Perceptron {
public:
    std::vector<double> weights; 
    double learningRate;
    int maxIter;

    Perceptron(double lr = 0.01, int maxIt = 1000)
        : learningRate(lr), maxIter(maxIt) {}

    // Инициализируем веса случайными малыми значениями
    void initWeights(int dimension) {
        weights.resize(dimension + 1); // +1 для смещения
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = ((double)rand() / RAND_MAX) * 0.01 - 0.005; 
        }
    }

    // Обучение перцептрона по правилу Розенблатта
    void train(const std::vector<LabeledPoint>& dataset) {
        int D = (int)dataset[0].point.coordinates.size();
        initWeights(D);

        for (int iter = 0; iter < maxIter; ++iter) {
            int errors = 0;
            for (auto& lp : dataset) {
                int predicted = predict(lp.point);
                int error = lp.label - predicted;
                if (error != 0) {
                    // Обновляем веса
                    for (int d = 0; d < D; ++d) {
                        weights[d] += learningRate * error * lp.point.coordinates[d];
                    }
                    weights[D] += learningRate * error; // обновляем смещение
                    errors++;
                }
            }
            if (errors == 0) {
                std::cout << "Перцептрон сошелся за " << iter + 1 << " итераций." << std::endl;
                break;
            }
            if (iter == maxIter - 1) {
                std::cout << "Перцептрон не сошелся за максимальное количество итераций." << std::endl;
            }
        }
    }

    int predict(const Point& p) const {
        double sum = 0.0;
        for (size_t d = 0; d < p.coordinates.size(); ++d) {
            sum += weights[d] * p.coordinates[d];
        }
        sum += weights[weights.size() - 1]; // смещение
        return (sum >= 0) ? 1 : -1;
    }
};

int main() {
    setlocale(LC_ALL, "Russian");
    srand((unsigned)time(0)); // Инициализируем ГПСЧ

    try {
        std::filesystem::current_path("../"); 
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Ошибка изменения рабочей директории: " << e.what() << "\n";
        return 1;
    }

    std::string inputFilename = "docs\\input.txt";
    std::string outputFilename = "docs\\output.txt";

    DataSet dataSet;
    dataSet.readInputFile(inputFilename);

    // Проверяем количество областей
    if (dataSet.regionsCount < 2) {
        std::cerr << "Для работы алгоритма необходимо минимум две области." << std::endl;
        return 1;
    }

    // Генерируем данные
    dataSet.generateRandomPoints(outputFilename);

    // Алгоритм Хо-Кашьяпа
    HoKashyapAlgorithm classifier;
    classifier.train(dataSet.dataset);

    // Выводим параметры разделяющей гиперплоскости Хо-Кашьяпа
    std::cout << "Полученный вектор весов (Хо-Кашьяпа) w: ";
    for (size_t i = 0; i < classifier.w.size() - 1; ++i) {
        std::cout << classifier.w[i] << ' ';
    }
    std::cout << "\nСмещение (b): " << classifier.w[classifier.w.size() - 1] << std::endl;

    // Оценка точности Хо-Кашьяпа
    {
        int correct = 0;
        for (const auto& lp : dataSet.dataset) {
            int predicted_label = classifier.predict(lp.point);
            if (predicted_label == lp.label)
                ++correct;
        }

        double accuracy = (double)correct / dataSet.dataset.size() * 100.0;
        std::cout << "Точность на обучающей выборке (Хо-Кашьяпа): " << accuracy << "%" << std::endl;
    }

    // Сохраняем веса Хо-Кашьяпа
    classifier.saveWeights("docs/weights.txt");

    // Обучаем перцептрон
    Perceptron perceptron(0.01, 1000);
    perceptron.train(dataSet.dataset);

    // Оценка точности перцептрона
    {
        int correct = 0;
        for (const auto& lp : dataSet.dataset) {
            int predicted_label = perceptron.predict(lp.point);
            if (predicted_label == lp.label)
                ++correct;
        }

        double accuracy = (double)correct / dataSet.dataset.size() * 100.0;
        std::cout << "Точность перцептрона: " << accuracy << "%" << std::endl;
    }

    // Записываем предсказания перцептрона в файл
    {
        std::ofstream predFile("docs\\perceptron_predictions.txt");
        if (!predFile.is_open()) {
            std::cerr << "Не удалось открыть файл perceptron_predictions.txt для записи." << std::endl;
            return 1;
        }

        for (auto& lp : dataSet.dataset) {
            for (double coord : lp.point.coordinates) {
                predFile << coord << " ";
            }
            int predicted_label = perceptron.predict(lp.point);
            predFile << predicted_label << "\n";
        }

        predFile.close();
        std::cout << "Предсказания перцептрона записаны в файл perceptron_predictions.txt" << std::endl;
    }

    // Применяем кластеризацию
    ClusteringAlgorithms clustering;
    std::vector<Point> points;
    for (auto& lp : dataSet.dataset) {
        points.push_back(lp.point);
    }

    // FOREL
    std::vector<Point> forelCenters = clustering.FOREL(points, 120);
    std::cout << "FOREL найдено кластеров: " << forelCenters.size() << std::endl;

    // Запишем центры FOREL в файл
    {
        std::ofstream forelFile("docs\\forel_centers.txt");
        if (forelFile.is_open()) {
            for (auto& c : forelCenters) {
                for (double coord : c.coordinates) {
                    forelFile << coord << " ";
                }
                forelFile << "\n";
            }
            forelFile.close();
            std::cout << "Центры кластеров FOREL записаны в forel_centers.txt" << std::endl;
        } else {
            std::cerr << "Не удалось открыть файл forel_centers.txt для записи." << std::endl;
        }
    }

    // ISODATA с K=3
    std::vector<Point> isodataCenters = clustering.ISODATA(points, 4);
    std::cout << "ISODATA найдено кластеров: " << isodataCenters.size() << std::endl;

    // Запишем центры ISODATA в файл
    {
        std::ofstream isoFile("docs\\isodata_centers.txt");
        if (isoFile.is_open()) {
            for (auto& c : isodataCenters) {
                for (double coord : c.coordinates) {
                    isoFile << coord << " ";
                }
                isoFile << "\n";
            }
            isoFile.close();
            std::cout << "Центры кластеров ISODATA записаны в isodata_centers.txt" << std::endl;
        } else {
            std::cerr << "Не удалось открыть файл isodata_centers.txt для записи." << std::endl;
        }
    }

    return 0;
}
