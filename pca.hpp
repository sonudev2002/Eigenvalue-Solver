#pragma once
#include "matrix.hpp"
#include <vector>

struct PCAResult {
    Matrix projected_data;
    Matrix principal_components; // columns are top-k eigenvectors
    std::vector<double> explained_variance;
};

class PCA {
public:
    // data: rows = samples, cols = features
    static PCAResult run(const Matrix& data, int k);
};