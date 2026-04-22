#include "pca.hpp"
#include "solver.hpp"
#include <stdexcept>
#include <numeric>

PCAResult PCA::run(const Matrix& data, int k) {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    if (k > n_features) throw std::invalid_argument("k cannot be greater than number of features.");

    // 1. Mean Centering
    std::vector<double> mean(n_features, 0.0);
    for (int r = 0; r < n_samples; ++r)
        for (int c = 0; c < n_features; ++c)
            mean[c] += data.at(r, c);
            
    for (int c = 0; c < n_features; ++c) mean[c] /= n_samples;

    Matrix centered = data;
    for (int r = 0; r < n_samples; ++r)
        for (int c = 0; c < n_features; ++c)
            centered.at(r, c) -= mean[c];

    // 2. Covariance Matrix (X^T * X / (n-1))
    Matrix cov = centered.transpose() * centered;
    double scale = 1.0 / (n_samples - 1);
    for (int r = 0; r < cov.rows(); ++r)
        for (int c = 0; c < cov.cols(); ++c)
            cov.at(r, c) *= scale;

    // 3. Eigendecomposition
    SolverConfig cfg; cfg.max_iterations = 3000;
    QRResult eig = EigenSolver::shiftedQR(cov, cfg); // Already sorts descending internally

    // 4. Extract Top K Components
    Matrix W(n_features, k);
    std::vector<double> variances(k);
    for (int i = 0; i < k; ++i) {
        W.setColumn(i, eig.eigenvectors.getColumn(i));
        variances[i] = eig.eigenvalues[i];
    }

    // 5. Project Data
    PCAResult result;
    result.projected_data = centered * W;
    result.principal_components = W;
    result.explained_variance = variances;

    return result;
}