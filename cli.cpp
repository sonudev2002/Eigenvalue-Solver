#include "cli.hpp"
#include "matrix.hpp"
#include "solver.hpp"
#include "pca.hpp"
#include "benchmark.hpp"
#include <iostream>

AppArgs CLI::parse(int argc, char** argv) {
    AppArgs args;
    if (argc > 1) args.run_tests = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--algo" && i + 1 < argc) { args.algo = argv[++i]; }
        else if (arg == "--input" && i + 1 < argc) { args.input_file = argv[++i]; }
        else if (arg == "--pca") { args.run_pca = true; }
        else if (arg == "--compare") { args.run_compare = true; }
        else if (arg == "--k" && i + 1 < argc) { args.k = std::stoi(argv[++i]); }
    }
    return args;
}

void CLI::execute(const AppArgs& args) {
    if (args.run_pca && !args.input_file.empty()) {
        Matrix data = Matrix::fromFile(args.input_file);
        auto res = PCA::run(data, args.k);
        std::cout << "\n[PCA Pipeline] Projected Data (" << res.projected_data.rows() << "x" << res.projected_data.cols() << "):\n";
        res.projected_data.print();
    } 
    else if (args.run_compare && !args.input_file.empty()) {
        Benchmark::runComparisonDashboard(Matrix::fromFile(args.input_file));
    }
    else if (!args.algo.empty() && !args.input_file.empty()) {
        Matrix A = Matrix::fromFile(args.input_file);
        SolverConfig cfg; cfg.verbose = true;
        if (args.algo == "power") EigenSolver::powerIteration(A, std::vector<double>(A.cols(), 1.0), cfg);
        else if (args.algo == "qr") EigenSolver::shiftedQR(A, cfg);
        else std::cerr << "Unknown algo: " << args.algo << "\n";
    }
}