#pragma once
#include "matrix.hpp"

struct SolverConfig {
    double epsilon        = 1e-10;
    int    max_iterations = 2000;
    bool   verbose        = false;
};

struct PowerIterResult {
    double              eigenvalue;
    std::vector<double> eigenvector;
    int                 iterations;
    bool                converged;
};

struct QRResult {
    std::vector<double> eigenvalues;
    Matrix              eigenvectors; 
    int                 iterations;
    bool                converged;
    
    void sortDescending();
};
struct LanczosResult {
    double              eigenvalue;
    std::vector<double> eigenvector;
    int                 iterations;
    bool                converged;
};

class EigenSolver {
public:
    static PowerIterResult powerIteration(const Matrix& A, std::vector<double> v_init, const SolverConfig& cfg = SolverConfig{});
    static QRResult shiftedQR(const Matrix& A, const SolverConfig& cfg = SolverConfig{});
    static double computeResidual(const Matrix& A, const std::vector<double>& v, double lambda);
    static LanczosResult lanczos(const Matrix& A, const SolverConfig& cfg = SolverConfig{});

private:
    static std::pair<Matrix, Matrix> toHessenberg(const Matrix& A);
    static std::pair<Matrix, Matrix> qrDecompose(const Matrix& A);
    
    // Optimized Dispatchers
    static QRResult symmetricQR(const Matrix& A, const SolverConfig& cfg);
    static QRResult generalQR(const Matrix& A, const SolverConfig& cfg);
};