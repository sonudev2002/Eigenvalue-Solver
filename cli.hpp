#pragma once
#include <string>

struct AppArgs {
    bool run_tests = true;
    bool run_compare = false;
    bool run_pca = false;
    std::string algo = "";
    std::string input_file = "";
    int k = 1;
};

class CLI {
public:
    static AppArgs parse(int argc, char** argv);
    static void execute(const AppArgs& args);
};