#include "cli.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <random>

void runBuiltInTests() {
    std::cout << "Running Internal Test Suite...\n";

    int N = 1000;
    Matrix A(N, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double val = dist(rng);
            A.at(i,j) = val;
            A.at(j,i) = val;
        }
    }
    std::cout << "Generated random symmetric matrix A of size " << N << "x" << N << "\n";
    Benchmark::runComparisonDashboard(A);
}

int main(int argc, char** argv) {
    try {
        AppArgs args = CLI::parse(argc, argv);
        if (args.run_tests) runBuiltInTests();
        else CLI::execute(args);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}