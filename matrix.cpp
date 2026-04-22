#include "matrix.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("Dimensions must be >= 0.");
}

Matrix Matrix::identity(int n) {
    Matrix I(n, n, 0.0);
    for (int i = 0; i < n; ++i) I.at(i, i) = 1.0;
    return I;
}

Matrix Matrix::fromVector(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return Matrix();
    int rows = data.size();
    int cols = data[0].size();
    Matrix M(rows, cols);
    for (int r = 0; r < rows; ++r) {
        if ((int)data[r].size() != cols) throw std::invalid_argument("Ragged matrix.");
        for (int c = 0; c < cols; ++c) M.at(r, c) = data[r][c];
    }
    return M;
}

Matrix Matrix::fromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open file: " + filename);
    int rows, cols;
    if (!(file >> rows >> cols)) throw std::runtime_error("Invalid file format.");
    Matrix M(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        if (!(file >> M.data_[i])) throw std::runtime_error("Not enough data in file.");
    }
    return M;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) result.data_[i] = data_[i] + rhs.data_[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) result.data_[i] = data_[i] - rhs.data_[i];
    return result;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_) throw std::invalid_argument("Size mismatch in multiply.");
    Matrix result(rows_, rhs.cols_, 0.0);
    for (int r = 0; r < rows_; ++r) {
        for (int k = 0; k < cols_; ++k) {
            double a_rk = at(r, k);
            if (a_rk == 0.0) continue;
            for (int c = 0; c < rhs.cols_; ++c) result.at(r, c) += a_rk * rhs.at(k, c);
        }
    }
    return result;
}

std::vector<double> Matrix::multiply(const std::vector<double>& v) const {
    std::vector<double> result(rows_, 0.0);
    for (int r = 0; r < rows_; ++r) {
        double sum = 0.0;
        const double* row_ptr = data_.data() + r * cols_;
        for (int c = 0; c < cols_; ++c) sum += row_ptr[c] * v[c];
        result[r] = sum;
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix T(cols_, rows_);
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) T.at(c, r) = at(r, c);
    return T;
}

double Matrix::frobeniusNorm() const {
    double sum = 0.0;
    for (double v : data_) sum += v * v;
    return std::sqrt(sum);
}

double Matrix::offDiagonalNorm() const {
    double max_val = 0.0;

    for (int i = 1; i < rows_; ++i) {
        max_val = std::max(max_val, std::abs(at(i, i - 1)));
    }

    return max_val;
}

void Matrix::setColumn(int col, const std::vector<double>& v) {
    for (int r = 0; r < rows_; ++r) at(r, col) = v[r];
}

std::vector<double> Matrix::getColumn(int col) const {
    std::vector<double> v(rows_);
    for (int r = 0; r < rows_; ++r) v[r] = at(r, col);
    return v;
}

std::vector<double> Matrix::diagonal() const {
    int n = std::min(rows_, cols_);
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) d[i] = at(i, i);
    return d;
}

void Matrix::print(const std::string& label, int width, int prec) const {
    if (!label.empty()) std::cout << label << "\n";
    for (int r = 0; r < rows_; ++r) {
        std::cout << "  [ ";
        for (int c = 0; c < cols_; ++c)
            std::cout << std::setw(width) << std::fixed << std::setprecision(prec) << at(r, c) << " ";
        std::cout << "]\n";
    }
    std::cout << "\n";
}

double VecUtil::dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

double VecUtil::norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

std::vector<double> VecUtil::normalize(const std::vector<double>& v) {
    double n = norm(v);
    if (n < 1e-15) throw std::runtime_error("Zero vector.");
    return scale(v, 1.0 / n);
}

std::vector<double> VecUtil::subtract(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

std::vector<double> VecUtil::scale(const std::vector<double>& v, double s) {
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) r[i] = v[i] * s;
    return r;
}

void VecUtil::print(const std::vector<double>& v, const std::string& label, int prec) {
    if (!label.empty()) std::cout << label << ": ";
    std::cout << "[ ";
    for (double x : v) std::cout << std::fixed << std::setprecision(prec) << x << " ";
    std::cout << "]\n";
}