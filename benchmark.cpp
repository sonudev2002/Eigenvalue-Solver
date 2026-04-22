#include "benchmark.hpp"
#include "solver.hpp"
#include "pca.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>   // 🔥 REQUIRED for std::min

void Benchmark::runComparisonDashboard(const Matrix& A) {
    std::cout << "\nInput Matrix Size: " << A.rows() << "x" << A.cols() << "\n";

    BenchmarkData data_power{"Power", 0, 0, 0};
    BenchmarkData data_qr{"QR", 0, 0, 0};
    BenchmarkData data_lanczos{"Lanczos", 0, 0, 0};

    // 🔥 RANDOM INIT (fixed)
    std::vector<double> v_init(A.cols());
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : v_init) x = dist(gen);

    SolverConfig cfg;

    SolverConfig cfg_lanczos;
    cfg_lanczos.max_iterations = 100;
    cfg_lanczos.epsilon = 1e-10;

    auto t_start_all = std::chrono::high_resolution_clock::now();

    PowerIterResult power_res;
    QRResult qr_res;
    LanczosResult lanczos_res;

    std::thread t1([&]() {
        auto t0 = std::chrono::high_resolution_clock::now();
        power_res = EigenSolver::powerIteration(A, v_init, cfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        data_power.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        data_power.iterations = power_res.iterations;
        data_power.error = EigenSolver::computeResidual(A, power_res.eigenvector, power_res.eigenvalue);
    });

    std::thread t2([&]() {
        auto t0 = std::chrono::high_resolution_clock::now();
        qr_res = EigenSolver::shiftedQR(A, cfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        data_qr.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        data_qr.iterations = qr_res.iterations;
        data_qr.error = EigenSolver::computeResidual(A, qr_res.eigenvectors.getColumn(0), qr_res.eigenvalues[0]);
    });

    std::thread t3([&]() {
        auto t0 = std::chrono::high_resolution_clock::now();
        lanczos_res = EigenSolver::lanczos(A, cfg_lanczos);
        auto t1 = std::chrono::high_resolution_clock::now();
        data_lanczos.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        data_lanczos.iterations = lanczos_res.iterations;
        data_lanczos.error = EigenSolver::computeResidual(A, lanczos_res.eigenvector, lanczos_res.eigenvalue);
    });

    t1.join(); t2.join(); t3.join();

    auto t_end_all = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(t_end_all - t_start_all).count();

    // ================= PCA =================
    auto t0_pca = std::chrono::high_resolution_clock::now();
    auto pca_res = PCA::run(A, 3);
    auto t1_pca = std::chrono::high_resolution_clock::now();
    double pca_time = std::chrono::duration<double, std::milli>(t1_pca - t0_pca).count();

    // ================= Throughput =================
    int runs = 10;
    auto t0_thr = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) EigenSolver::lanczos(A, cfg_lanczos);
    auto t1_thr = std::chrono::high_resolution_clock::now();
    double throughput = runs / std::chrono::duration<double>(t1_thr - t0_thr).count();

    // ================= DASHBOARD =================
    std::cout << "\n======================================================\n";
    std::cout << "  BENCHMARK DASHBOARD\n";
    std::cout << "======================================================\n";
    std::cout << std::left << std::setw(15) << "Algorithm"
              << std::setw(15) << "Time(ms)"
              << std::setw(15) << "Iterations"
              << "Error ||Av-λv||\n";
    std::cout << "------------------------------------------------------\n";

    auto print_row = [](const BenchmarkData& d) {
        std::cout << std::left << std::setw(15) << d.algorithm
                  << std::setw(15) << std::fixed << std::setprecision(3) << d.time_ms
                  << std::setw(15) << d.iterations
                  << std::scientific << std::setprecision(3) << d.error << "\n";
    };

    print_row(data_power);
    print_row(data_lanczos);
    print_row(data_qr);

    std::cout << "------------------------------------------------------\n";
    std::cout << "PCA Time: " << std::fixed << std::setprecision(3) << pca_time << " ms\n";
    std::cout << "Throughput (Lanczos): " << std::fixed << std::setprecision(3) << throughput << " runs/sec\n";
    std::cout << "Total Wall Time: " << total_time << " ms\n";
    std::cout << "======================================================\n";

    // ================= FINAL RESULTS =================
    std::cout << "\n===== FINAL RESULTS =====\n";

    std::cout << "\n[Power Iteration]\n";
    std::cout << "Eigenvalue: " << power_res.eigenvalue << "\n";
    std::cout << "Iterations: " << power_res.iterations << "\n";
    std::cout << "Eigenvector (first 5 values): ";
    for (int i = 0; i < 5 && i < power_res.eigenvector.size(); i++)
        std::cout << power_res.eigenvector[i] << " ";
    std::cout << "\n";

    std::cout << "\n[Lanczos]\n";
std::cout << "Eigenvalue: " << lanczos_res.eigenvalue << "\n";
std::cout << "Iterations: " << lanczos_res.iterations << "\n";

std::cout << "Eigenvector (first 5 values): ";
for (int i = 0; i < 5 && i < lanczos_res.eigenvector.size(); i++)
    std::cout << lanczos_res.eigenvector[i] << " ";
std::cout << "\n";

    

    std::cout << "\n[Shifted QR]\n";
    std::cout << "Iterations: " << qr_res.iterations << "\n";
    std::cout << "Top Eigenvalue: " << qr_res.eigenvalues[0] << "\n";

    std::cout << "\n[PCA]\n";
    std::cout << "Top Eigenvalues:\n";
    for (int i = 0; i < 3 && i < pca_res.explained_variance.size(); i++)
        std::cout << pca_res.explained_variance[i] << "\n";

    // ================= SUMMARY =================
    std::cout << "\n===== SUMMARY =====\n";

    double min_time = std::min({data_power.time_ms, data_qr.time_ms, data_lanczos.time_ms});
    if (min_time == data_lanczos.time_ms)
        std::cout << "Fastest Method: Lanczos\n";
    else if (min_time == data_power.time_ms)
        std::cout << "Fastest Method: Power Iteration\n";
    else
        std::cout << "Fastest Method: Shifted QR\n";

    double min_error = std::min({data_power.error, data_qr.error, data_lanczos.error});
    if (min_error == data_qr.error)
        std::cout << "Most Accurate: Shifted QR\n";
    else if (min_error == data_lanczos.error)
        std::cout << "Most Accurate: Lanczos\n";
    else
        std::cout << "Most Accurate: Power Iteration\n";

    std::cout << "Best Practical Choice: Lanczos\n";

    std::cout << "\nUse Cases:\n";
    std::cout << "- Power → simple cases (large spectral gap)\n";
    std::cout << "- Lanczos → large matrices, dominant eigenvalues\n";
    std::cout << "- QR → full eigen spectrum (high accuracy)\n";

    std::cout << "======================================================\n\n";
}