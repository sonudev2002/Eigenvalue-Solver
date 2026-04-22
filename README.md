# Eigenvalue Solver & Benchmarking System (C++)

## Overview

This project implements and compares multiple eigenvalue algorithms in C++—**Power Iteration**, **Lanczos**, and **Shifted QR**—to study their performance, convergence behavior, and numerical stability on large matrices.

It includes a **multithreaded CLI-based benchmarking system** that evaluates tradeoffs between speed, accuracy, and scalability across different algorithms.

---

## Key Features

* 🚀 **Eigenvalue Algorithms Implemented**

  * Power Iteration (baseline method)
  * Lanczos Algorithm (efficient for large symmetric matrices)
  * Shifted QR Algorithm (accurate full-spectrum method)

* ⚡ **CLI-Based Benchmarking**

  * Measures:

    * Execution time
    * Iteration count
    * Residual error (||Av − λv||)
  * Outputs structured comparison directly in terminal

* 🧵 **Parallel Execution**

  * Power, Lanczos, and QR executed concurrently using `std::thread`

* 🧠 **Convergence & Stability Handling**

  * Residual-based stopping criteria
  * Wilkinson shift in QR
  * Krylov subspace construction in Lanczos

---

## Algorithms

### Power Iteration

* Computes dominant eigenvalue
* Time Complexity: **O(n² × k)**
* Limitation: Performs poorly when eigenvalues are close (small spectral gap)

---

### Lanczos Algorithm

* Designed for large symmetric matrices
* Uses Krylov subspace to approximate dominant eigenvalues
* Time Complexity: **O(n² × k)**
* Efficient and scalable in practice

---

### Shifted QR Algorithm

* Computes full eigenvalue spectrum
* Uses Hessenberg reduction + Wilkinson shift
* Time Complexity: **O(n³)**
* Most accurate but computationally expensive

---

## Example Benchmark Output

| Algorithm | Time (ms) | Iterations | Error |
| --------- | --------- | ---------- | ----- |
| Power     | ~5000     | 2000       | 4e-01 |
| Lanczos   | ~100      | 100        | 1e-07 |
| QR        | ~3700     | 2000+      | 1e-13 |

---

## Interpretation

* **Power Iteration**

  * Simple baseline but unreliable for difficult matrices

* **Lanczos**

  * Fast and effective for large matrices
  * Best practical choice in this implementation

* **QR**

  * Most accurate
  * Suitable when full eigenvalue spectrum is required

---

## System Design

* Modular implementation (`solver.cpp`, `matrix.cpp`)
* Centralized benchmarking module (`benchmark.cpp`)
* CLI output formatting for structured comparison
* Parallel execution using threads

---

## Project Structure

```
├── solver.cpp        # Power, Lanczos, QR implementations
├── solver.hpp
├── matrix.cpp        # Matrix operations
├── matrix.hpp
├── benchmark.cpp     # Multithreaded CLI comparison (timing, error, analysis)
├── benchmark.hpp
├── pca.cpp           # PCA using eigen decomposition
├── pca.hpp
├── cli.cpp           # CLI interaction / output formatting
├── cli.hpp
├── main.cpp          # Entry point
```

---

## How to Run

```bash
g++ *.cpp -O2 -march=native -o main
./main
```

---

## Key Learnings

* Power Iteration depends heavily on spectral gap
* Lanczos improves convergence using subspace methods
* QR provides high accuracy but has higher computational cost
* Algorithm selection is important for performance

---

## Future Improvements

* Sparse matrix optimization
* Better memory/cache optimization
* SIMD/vectorization
* Automatic algorithm selection

---

## Resume Description

> Implemented and benchmarked eigenvalue algorithms (Power Iteration, Lanczos, Shifted QR) in C++, analyzing performance, convergence, and numerical stability on large matrices using multithreading.

---

## Author

C++ developer focused on algorithms, performance, and systems-level problem solving.
