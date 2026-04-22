#pragma once
#include <vector>
#include <string>

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols, double fill = 0.0);
    Matrix(const Matrix&)            = default;
    Matrix(Matrix&&)                 = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&)      = default;
    ~Matrix()                        = default;

    static Matrix identity(int n);
    static Matrix fromVector(const std::vector<std::vector<double>>& data);
    static Matrix fromFile(const std::string& filename); // File IO

    inline double& at(int r, int c)       { return data_[r * cols_ + c]; }
    inline double  at(int r, int c) const { return data_[r * cols_ + c]; }
    inline int rows() const               { return rows_; }
    inline int cols() const               { return cols_; }

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const; 

    std::vector<double> multiply(const std::vector<double>& v) const;
    Matrix transpose() const;
    
    double frobeniusNorm() const;
    double offDiagonalNorm() const;

    void setColumn(int col, const std::vector<double>& v);
    std::vector<double> getColumn(int col) const;
    std::vector<double> diagonal() const;

    void print(const std::string& label = "", int width = 10, int prec = 4) const;

private:
    int rows_{0}, cols_{0};
    std::vector<double> data_; 
};

namespace VecUtil {
    double dot(const std::vector<double>& a, const std::vector<double>& b);
    double norm(const std::vector<double>& v);
    std::vector<double> normalize(const std::vector<double>& v);
    std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> scale(const std::vector<double>& v, double s);
    void print(const std::vector<double>& v, const std::string& label = "", int prec = 4);
}