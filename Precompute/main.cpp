#include "HelperFunctions.h"
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "FeatureOp.cpp"
#include <unistd.h>


XoshiroGenerator fRNG;

XoshiroGenerator init_rng(uint64_t seed) {
    XoshiroGenerator rng;
    rng.initialize(seed);
    return rng;
}

int main(int argc, char **argv) {
    param = parseArgs(argc, argv);
    fRNG = init_rng(param.seed);
    // Input graph
    Graph graph;
    graph.set_alpha(param.alpha);
    std::ifstream bin_file(param.data_folder + "/graph.bin");
    if (!bin_file.good()) {
        CleanGraph cleaner;
        cleaner.clean_graph(param.graph_file, param.data_folder);
    }
    graph.read_binary(param.data_folder + "/attribute.txt", param.data_folder + "/graph.bin");

    // Perfrom feature operations
    if (param.algorithm == "featpush"){
        Base base(graph, param);
        base.push();
        base.show_statistics();
    } else if (param.algorithm == "featreuse") {
        Base_reuse base(graph, param);
        base.push();
        base.show_statistics();
    }
    printf("%s\n", std::string(80, '-').c_str());
    return 0;
}
