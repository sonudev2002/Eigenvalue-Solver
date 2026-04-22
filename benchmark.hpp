#pragma once
#include "matrix.hpp"
#include <string>

struct BenchmarkData {
    std::string algorithm;
    double time_ms;
    int iterations;
    double error;
};

class Benchmark {
public:
    static void runComparisonDashboard(const Matrix& A);
};