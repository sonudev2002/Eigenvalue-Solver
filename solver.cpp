#include "solver.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

static inline double sign(double x) { return (x >= 0.0) ? 1.0 : -1.0; }

double EigenSolver::computeResidual(const Matrix& A, const std::vector<double>& v, double lambda) {
    auto Av = A.multiply(v);
    auto diff = VecUtil::subtract(Av, VecUtil::scale(v, lambda));
    return VecUtil::norm(diff);
}

void QRResult::sortDescending() {
    int n = eigenvalues.size();
    std::vector<std::pair<double, int>> indices(n);
    for(int i = 0; i < n; ++i) indices[i] = {eigenvalues[i], i};
    
    // Sort ALGEBRAICALLY (highest positive to lowest negative), not by magnitude.
    // This is mathematically required for accurate PCA and variance tracking.
    std::sort(indices.begin(), indices.end(), [](const auto& a, const auto& b){
        return a.first > b.first; 
    });
    
    std::vector<double> new_eigenvalues(n);
    Matrix new_eigenvectors(eigenvectors.rows(), n);
    for(int i = 0; i < n; ++i) {
        new_eigenvalues[i] = indices[i].first;
        new_eigenvectors.setColumn(i, eigenvectors.getColumn(indices[i].second));
    }
    eigenvalues = new_eigenvalues;
    eigenvectors = new_eigenvectors;
}

PowerIterResult EigenSolver::powerIteration(const Matrix& A, std::vector<double> v_init, const SolverConfig& cfg) {
    const int n = A.rows();
    PowerIterResult result{0.0, v_init, 0, false};
    std::vector<double> v = VecUtil::normalize(v_init);
    
    std::vector<double> w(n);
    double lambda = 0.0;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
    ++result.iterations;

    // 1. Multiply
    w = A.multiply(v);

    // 2. Normalize FIRST (critical)
    double w_norm = VecUtil::norm(w);
    if (w_norm < 1e-15) break;

    std::vector<double> v_new = VecUtil::scale(w, 1.0 / w_norm);

    // 3. Rayleigh quotient using NEW vector
    lambda = VecUtil::dot(v_new, A.multiply(v_new));

    // 4. Residual using NEW vector
    double residual = computeResidual(A, v_new, lambda);

    if (residual < cfg.epsilon) {
        result.converged = true;
        v = v_new;
        break;
    }

    // 5. Update
    v = v_new;
}
    
    result.eigenvalue = lambda;
    result.eigenvector = v;
    return result;
}

std::pair<Matrix, Matrix> EigenSolver::toHessenberg(const Matrix& A) {
    const int n = A.rows();
    Matrix H = A;
    Matrix Q = Matrix::identity(n);
    
    std::vector<double> x, u;
    for (int k = 0; k < n - 2; ++k) {
        int m = n - k - 1;
        x.resize(m); u.resize(m);
        for (int i = 0; i < m; ++i) x[i] = H.at(k + 1 + i, k);

        double x_norm = VecUtil::norm(x);
        if (x_norm < 1e-15) continue;

        u = x; u[0] += sign(x[0]) * x_norm;
        double u_norm = VecUtil::norm(u);
        if (u_norm < 1e-15) continue;
        
        u = VecUtil::scale(u, 1.0 / u_norm);

        for (int c = k; c < n; ++c) {
            double dot = 0.0;
            for (int i = 0; i < m; ++i) dot += u[i] * H.at(k + 1 + i, c);
            for (int i = 0; i < m; ++i) H.at(k + 1 + i, c) -= 2.0 * dot * u[i];
        }

        for (int r = 0; r < n; ++r) {
            double dot = 0.0;
            for (int i = 0; i < m; ++i) dot += H.at(r, k + 1 + i) * u[i];
            for (int i = 0; i < m; ++i) H.at(r, k + 1 + i) -= 2.0 * dot * u[i];
        }

        for (int r = 0; r < n; ++r) {
            double dot = 0.0;
            for (int i = 0; i < m; ++i) dot += Q.at(r, k + 1 + i) * u[i];
            for (int i = 0; i < m; ++i) Q.at(r, k + 1 + i) -= 2.0 * dot * u[i];
        }
    }
    return {H, Q};
}

QRResult EigenSolver::shiftedQR(const Matrix& A, const SolverConfig& cfg) {
    bool is_symmetric = true;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (std::abs(A.at(i, j) - A.at(j, i)) > 1e-9) {
                is_symmetric = false;
                break;
            }
        }
        if (!is_symmetric) break;
    }

    if (is_symmetric) return symmetricQR(A, cfg);
    return generalQR(A, cfg);
}

QRResult EigenSolver::symmetricQR(const Matrix& A, const SolverConfig& cfg) {
    int n = A.rows();
    QRResult result;
    result.converged = true;
    result.iterations = 0;

    auto [H, Q_h] = toHessenberg(A);
    
    std::vector<double> d(n), e(n, 0.0);
    for(int i = 0; i < n; ++i) {
        d[i] = H.at(i, i);
        if (i < n - 1) e[i] = H.at(i + 1, i);
    }

    std::vector<double> c_arr(n), s_arr(n);

    for (int l = 0; l < n; l++) {
        int iter = 0;
        while (true) {
            int m;
            for (m = l; m < n - 1; m++) {
                double dd = std::abs(d[m]) + std::abs(d[m + 1]);
                if (std::abs(e[m]) + dd == dd) break; 
            }
            if (m == l) break; 

            if (++iter >= cfg.max_iterations) {
                result.converged = false;
                break;
            }
            result.iterations++;

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;
            int i, idx = 0;
            
            for (i = m - 1; i >= l; i--) {
                double f = s * e[i];
                double b = c * e[i];
                if (std::abs(f) >= std::abs(g)) {
                    c = g / f; r = std::hypot(c, 1.0);
                    e[i + 1] = f * r; s = 1.0 / r; c *= s;
                } else {
                    s = f / g; r = std::hypot(s, 1.0);
                    e[i + 1] = g * r; c = 1.0 / r; s *= c;
                }
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                
                c_arr[idx] = c;
                s_arr[idx] = s;
                idx++;
            }
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = 0.0;

            for (int k = 0; k < n; k++) {
                int local_idx = 0;
                for (i = m - 1; i >= l; i--) {
                    double c_val = c_arr[local_idx];
                    double s_val = s_arr[local_idx];
                    local_idx++;
                    
                    double f_q = Q_h.at(k, i + 1);
                    Q_h.at(k, i + 1) = s_val * Q_h.at(k, i) + c_val * f_q;
                    Q_h.at(k, i)     = c_val * Q_h.at(k, i) - s_val * f_q;
                }
            }
        }
    }

    result.eigenvalues = d;
    result.eigenvectors = Q_h;
    result.sortDescending();
    return result;
}

QRResult EigenSolver::generalQR(const Matrix& A, const SolverConfig& cfg) {
    const int n = A.rows();
    QRResult result;
    result.converged = true;
    result.iterations = 0;

    auto [H, Q_h] = toHessenberg(A);
    Matrix Q_acc = Matrix::identity(n);
    std::vector<double> c(n, 0.0), s(n, 0.0);

    int bottom = n - 1;
    while (bottom > 0) {
        
        // 1. Scan upwards to find the top of the active block
        int top = bottom;
        while (top > 0) {
            double sub = std::abs(H.at(top, top - 1));
            double scale = std::abs(H.at(top - 1, top - 1)) + std::abs(H.at(top, top));
            if (sub <= cfg.epsilon * (scale + 1e-12)) {
                H.at(top, top - 1) = 0.0; 
                break;
            }
            top--;
        }

        // Deflation complete for this element
        if (top == bottom) {
            bottom--;
            continue;
        }

        int iter = 0;
        while (iter < cfg.max_iterations) {
            ++iter;
            ++result.iterations;

            // 2. Wilkinson shift using active block corner
            double a_val = H.at(bottom-1, bottom-1), b_val = H.at(bottom-1, bottom);
            double c_val = H.at(bottom, bottom-1),   d_val = H.at(bottom, bottom);
            
            double delta = (a_val - d_val) / 2.0;
            double sgn = (delta >= 0) ? 1.0 : -1.0;
            double denom = std::abs(delta) + std::sqrt(delta * delta + b_val * c_val);
            double mu = (denom == 0.0) ? d_val : d_val - sgn * (b_val * c_val) / denom;

            for (int i = top; i <= bottom; ++i) H.at(i, i) -= mu;

            // 3. QR Sweep ONLY within [top, bottom] bounds
            for (int i = top; i < bottom; ++i) {
                double x = H.at(i, i), y = H.at(i + 1, i);
                if (std::abs(y) < 1e-15) { c[i] = 1.0; s[i] = 0.0; }
                else {
                    double r_val = std::hypot(x, y);
                    c[i] = x / r_val; s[i] = -y / r_val;
                }
                
                // Left Givens
                for (int j = i; j < n; ++j) {
                    double h1 = H.at(i, j), h2 = H.at(i + 1, j);
                    H.at(i, j)     = c[i] * h1 - s[i] * h2;
                    H.at(i + 1, j) = s[i] * h1 + c[i] * h2;
                }
            }

            // Right Givens (Cache-friendly row iteration)
            for (int j = 0; j < n; ++j) {
                for (int i = top; i < bottom; ++i) {
                    if (j <= i + 1) { 
                        double h1 = H.at(j, i), h2 = H.at(j, i + 1);
                        H.at(j, i)     = c[i] * h1 - s[i] * h2;
                        H.at(j, i + 1) = s[i] * h1 + c[i] * h2;
                    }
                }
            }

            for (int i = top; i <= bottom; ++i) H.at(i, i) += mu;

            // Accumulate Q
            for (int j = 0; j < n; ++j) {
                for (int i = top; i < bottom; ++i) {
                    double q1 = Q_acc.at(j, i), q2 = Q_acc.at(j, i + 1);
                    Q_acc.at(j, i)     = c[i] * q1 - s[i] * q2;
                    Q_acc.at(j, i + 1) = s[i] * q1 + c[i] * q2;
                }
            }
            
            // Re-check deflation at the bottom
            double sub_check = std::abs(H.at(bottom, bottom - 1));
            double scale_check = std::abs(H.at(bottom - 1, bottom - 1)) + std::abs(H.at(bottom, bottom));
            if (sub_check <= cfg.epsilon * (scale_check + 1e-12)) {
                H.at(bottom, bottom - 1) = 0.0;
                break;
            }
        }
        
        if(iter >= cfg.max_iterations) {
            result.converged = false;
            break;
        }
        bottom--;
    }
    
    result.eigenvalues = H.diagonal();
    result.eigenvectors = Q_h * Q_acc;
    result.sortDescending();
    return result;
}


LanczosResult EigenSolver::lanczos(const Matrix& A, const SolverConfig& cfg) {
    const int n = A.rows();
    
    // m is the size of our Krylov subspace. Usually, 30-50 iterations 
    // is enough to find the dominant eigenvalue even for 10,000x10,000 matrices.
    int m = std::min(n, cfg.max_iterations); 

    std::vector<double> alpha(m, 0.0);
    std::vector<double> beta(m, 0.0);
    
    // V stores our orthogonal basis vectors (columns)
    Matrix V(n, m); 

    // Initial random unit vector
    std::vector<double> v_curr(n, 1.0); 
    v_curr = VecUtil::normalize(v_curr);
    V.setColumn(0, v_curr);

    // Pre-allocate to avoid heap allocations in the hot loop
    std::vector<double> v_prev(n, 0.0);
    std::vector<double> w(n, 0.0);

    int k = 0; // Tracks actual subspace dimension reached

    for (int j = 0; j < m; ++j) {
        k = j + 1;

        // 1. Matrix-Vector Multiply (The only heavy operation)
        w = A.multiply(v_curr);

        // 2. Orthogonalize against the previous vector
        if (j > 0) {
            for (int i = 0; i < n; ++i) {
                w[i] -= beta[j - 1] * v_prev[i];
            }
        }

        // 3. Compute diagonal element of T
        alpha[j] = VecUtil::dot(w, v_curr);

        // 4. Orthogonalize against the current vector
        for (int i = 0; i < n; ++i) {
            w[i] -= alpha[j] * v_curr[i];
        }

        // Optional but recommended for floating-point stability:
        // Local re-orthogonalization to strictly kill ghost projections
        double proj = VecUtil::dot(w, v_curr);
        for (int i = 0; i < n; ++i) {
            w[i] -= proj * v_curr[i];
        }

        // 5. Compute sub-diagonal element of T
        beta[j] = VecUtil::norm(w);

        // Check for early Krylov subspace exhaustion (invariant subspace found)
        if (beta[j] < 1e-12 || j == m - 1) {
            break;
        }

        // 6. Advance the vectors
        v_prev = v_curr;
        for (int i = 0; i < n; ++i) {
            v_curr[i] = w[i] / beta[j];
        }
        V.setColumn(j + 1, v_curr);
    }

    // ============================================================
    // Solve the minimal m x m Tridiagonal Matrix T
    // ============================================================
    Matrix T(k, k, 0.0);
    for (int i = 0; i < k; ++i) {
        T.at(i, i) = alpha[i];
        if (i < k - 1) {
            T.at(i + 1, i) = beta[i];
            T.at(i, i + 1) = beta[i];
        }
    }

    SolverConfig t_cfg;
    t_cfg.epsilon = 1e-12;
    t_cfg.max_iterations = 1000;
    
    // Leverage our existing optimized Implicit QL solver!
    QRResult t_res = symmetricQR(T, t_cfg); 

    LanczosResult result;
    // symmetricQR already sorts algebraically, so [0] is the true dominant eigenvalue
    result.eigenvalue = t_res.eigenvalues[0]; 
    result.iterations = k;
    result.converged = true;

    // ============================================================
    // Project the T-eigenvector back to the original N-dimensional space
    // v_original = V * v_T
    // ============================================================
    std::vector<double> y = t_res.eigenvectors.getColumn(0);
    std::vector<double> dom_vec(n, 0.0);
    
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            dom_vec[i] += V.at(i, j) * y[j];
        }
    }
    
    result.eigenvector = VecUtil::normalize(dom_vec);

    return result;
}