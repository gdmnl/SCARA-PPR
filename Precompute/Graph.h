/*
  SpeedPPR implementation of graph processing
  Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank
*/
#ifndef SCARA_GRAPH_H
#define SCARA_GRAPH_H

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include "BasicDefinition.h"
#include "HelperFunctions.h"


template<class T>
inline void show_vector(const std::string &_header, const std::vector<T> &_vec) {
    if (_vec.empty()) {
        cout << "Empty Vector." << endl;
    } else {
        cout << endl << _header;
        bool identical = true;
        const T &elem = _vec.front();
        std::for_each(_vec.begin(), _vec.end(), [&](const T &e) { identical &= (e == elem); });
        if (identical) {
            cout << "\tSize of the Vector: " << _vec.size() << "\t Value of Each Element: " << elem;
        } else {
            cout << endl;
            std::copy(begin(_vec), end(_vec), std::ostream_iterator<T>(cout, "\t"));
        }
        cout << endl;
    }
}

class Graph {

private:
    NInt numOfVertices = 0;
    NInt numOfEdges = 0;
    NInt num_deadend_vertices = 0;
    NInt sid = 0;
    NInt dummy_id = 0;
    ScoreFlt alpha = 0.2;
    NInt max_size_edge_list = 0;
    IntVector out_degrees;
    IntVector in_degrees;
    IntVector start_pos_in_out_neighbor_lists;
    IntVector start_pos_in_appearance_pos_lists;
    IntVector out_neighbors_lists;
    IntVector appearance_pos_lists;
    IntVector deadend_vertices;

public:

    inline size_t get_num_dead_end() const {
        return deadend_vertices.size();
    }

    inline void set_dummy_out_degree_zero() {
        out_degrees[dummy_id] = 0;
        start_pos_in_out_neighbor_lists[dummy_id + 1] = start_pos_in_out_neighbor_lists[dummy_id];
    }

    inline void set_dummy_neighbor(const NInt &_id) {
        out_degrees[dummy_id] = 1;
        start_pos_in_out_neighbor_lists[dummy_id + 1] = start_pos_in_out_neighbor_lists[dummy_id] + 1;
        out_neighbors_lists[start_pos_in_out_neighbor_lists[dummy_id]] = _id;
    }

    inline void reset_set_dummy_neighbor() {
        out_degrees[dummy_id] = 0;
        out_neighbors_lists[start_pos_in_out_neighbor_lists[dummy_id]] = dummy_id;
        set_dummy_out_degree_zero();
    }

    inline const NInt &get_dummy_id() const {
        return dummy_id;
    }

    inline const NInt &get_sid() const {
        return sid;
    }

    inline const ScoreFlt &get_alpha() const {
        return alpha;
    }

    inline void set_alpha(const ScoreFlt _alpha = 0.2) {
        alpha = _alpha;
    }

    inline void fill_dead_end_neighbor_with_id(const NInt &_id) {
        for (NInt index = 0; index < num_deadend_vertices; ++index) {
            const NInt &id = deadend_vertices[index];
            const NInt &start = start_pos_in_out_neighbor_lists[id];
            out_neighbors_lists[start] = _id;
        }
    }

    inline void fill_dead_end_neighbor_with_id() {
        //cout<< "num_deadend_vertices: " << num_deadend_vertices <<endl;
        for (NInt index = 0; index < num_deadend_vertices; ++index) {
            const NInt &id = deadend_vertices[index];
            const NInt &start = start_pos_in_out_neighbor_lists[id];
            NInt _id = rand()%numOfVertices;
            out_neighbors_lists[start] = _id;
        }
    }


    inline void change_in_neighbors_adj(const NInt &_sid, const NInt &_target) {
        const NInt &idx_start = start_pos_in_appearance_pos_lists[_sid];
        const NInt &idx_end = start_pos_in_appearance_pos_lists[_sid + 1];
        for (NInt index = idx_start; index < idx_end; ++index) {
            out_neighbors_lists[appearance_pos_lists[index]] = _target;
        }
    }

    inline void restore_neighbors_adj(const NInt &_sid) {
        const NInt &idx_start = start_pos_in_appearance_pos_lists[_sid];
        const NInt &idx_end = start_pos_in_appearance_pos_lists[_sid + 1];
        for (NInt index = idx_start; index < idx_end; ++index) {
            out_neighbors_lists[appearance_pos_lists[index]] = _sid;
        }
    }

    inline void set_source_and_alpha(const NInt _sid, const ScoreFlt _alpha) {
        sid = _sid;
        alpha = _alpha;
//        fill_dead_end_neighbor_with_id(_sid);
    }

    Graph() = default;

    ~Graph() = default;

    inline const NInt &getNumOfVertices() const {
        return numOfVertices;
    }

    /**
     * @param _vid
     * @return return the original out degree
     */
    inline const NInt &original_out_degree(const NInt &_vid) const {
        assert(_vid < numOfVertices);
        return out_degrees[_vid];
    }

    inline const NInt &get_neighbor_list_start_pos(const NInt &_vid) const {
        assert(_vid < numOfVertices + 2);
        return start_pos_in_out_neighbor_lists[_vid];
    }


    inline const NInt &getOutNeighbor(const NInt &_index) const {
//        if (_index >= start_pos_in_out_neighbor_lists[dummy_id + 1]) {
//            MSG("Time to check " __FILE__)
//            MSG(__LINE__)
//        }
        assert(_index < start_pos_in_out_neighbor_lists[dummy_id + 1]);
        return out_neighbors_lists[_index];
    }

    inline const NInt &getNumOfEdges() const {
        return numOfEdges;
    }


    void read_binary(const std::string &_attribute_file,
                     const std::string &_graph_file) {
        {
            std::string line;
            std::ifstream attribute_file(_attribute_file);
            if (attribute_file.is_open()) {
                std::getline(attribute_file, line);
                size_t start1 = line.find_first_of('=');
                numOfVertices = std::stoul(line.substr(start1 + 1));
                std::getline(attribute_file, line);
                size_t start2 = line.find_first_of('=');
                numOfEdges = std::stoul(line.substr(start2 + 1));
                dummy_id = numOfVertices;
                // printf("The Number of Vertices: %" IDFMT "\n", numOfVertices);
                // printf("The Number of Edges: %" IDFMT "\n", numOfEdges);
                attribute_file.close();
            } else {
                printf(__FILE__ "; LINE %d; File Not Exists.\n", __LINE__);
                cout << _attribute_file << endl;
                exit(1);
            }
        }
        // const auto start = getCurrentTime();
        // create temporary graph
        std::vector<Edge> edges(numOfEdges);
        if (std::FILE *f = std::fopen(_graph_file.c_str(), "rb")) {
            size_t rtn = std::fread(edges.data(), sizeof edges[0], edges.size(), f);
            printf("Edge from fread: %zu\n", rtn);
            std::fclose(f);
        } else {
            printf("Graph::read_binary; File Not Exists.\n");
            cout << _graph_file << endl;
            exit(1);
        }
        // const auto end = getCurrentTime();
        // printf("Time Used For Loading BINARY : %.2f\n", end - start);

        // read the edges
        // the ids must be in the range from [0 .... the number of vertices - 1];
        numOfEdges = 0;
        out_degrees.clear();
        out_degrees.resize(numOfVertices + 2, 0);
        in_degrees.clear();
        in_degrees.resize(numOfVertices + 2, 0);
        for (auto &edge : edges) {
            const NInt &from_id = edge.from_id;
            const NInt &to_id = edge.to_id;
            // remove self loop
            if (from_id != to_id) {
                //the edge read is a directed one
                ++out_degrees[from_id];
                ++in_degrees[to_id];
                ++numOfEdges;
            }
        }
        /* final count */
//        printf("%d-th Directed Edge Processed.\n", numOfEdges);

        // sort the adj list
//        for (auto &neighbors : matrix) {
//            std::sort(neighbors.begin(), neighbors.end());
//        }

        // process the dead_end
        NInt degree_max = 0;
        deadend_vertices.clear();
        for (NInt i = 0; i < numOfVertices; ++i) {
            if (out_degrees[i] == 0) {
                deadend_vertices.emplace_back(i);
            }
            degree_max = std::max(degree_max, out_degrees[i]);
        }
        num_deadend_vertices = deadend_vertices.size();

        // process pos_list list
        start_pos_in_appearance_pos_lists.clear();
        start_pos_in_appearance_pos_lists.resize(numOfVertices + 2, 0);
        for (NInt i = 0, j = 1; j < numOfVertices; ++i, ++j) {
            start_pos_in_appearance_pos_lists[j] = start_pos_in_appearance_pos_lists[i] + in_degrees[i];
        }
        start_pos_in_appearance_pos_lists[numOfVertices] = numOfEdges;

        // process out list
        start_pos_in_out_neighbor_lists.clear();
        start_pos_in_out_neighbor_lists.resize(numOfVertices + 2, 0);
        for (NInt current_id = 0, next_id = 1; next_id < numOfVertices + 1; ++current_id, ++next_id) {
            start_pos_in_out_neighbor_lists[next_id] =
                    start_pos_in_out_neighbor_lists[current_id] + std::max(out_degrees[current_id], (NInt) 1u);
        }
        // process dummy vertex
        assert(start_pos_in_out_neighbor_lists[numOfVertices] == numOfEdges + deadend_vertices.size());
        out_degrees[dummy_id] = 0;
        start_pos_in_out_neighbor_lists[numOfVertices + 1] = start_pos_in_out_neighbor_lists[numOfVertices];

        // compute the positions
        IntVector out_positions_to_fill(start_pos_in_out_neighbor_lists.begin(),
                                                        start_pos_in_out_neighbor_lists.end());
        // fill the edge list
        out_neighbors_lists.clear();
        out_neighbors_lists.resize(numOfEdges + num_deadend_vertices + degree_max, 0);
        NInt edges_processed = 0;
        NInt msg_gap = std::max((NInt) 1u, numOfEdges / 10);
        std::vector<std::pair<NInt, NInt>> position_pair;
        position_pair.reserve(numOfEdges);
        for (auto &edge : edges) {
            const NInt &from_id = edge.from_id;
            const NInt &to_id = edge.to_id;
            // remove self loop
            if (from_id != to_id) {
                NInt &out_position = out_positions_to_fill[from_id];
                assert(out_position < out_positions_to_fill[from_id + 1]);
                out_neighbors_lists[out_position] = to_id;
                position_pair.emplace_back(to_id, out_position);
                ++out_position;
                ++edges_processed;
                // if (edges_processed % msg_gap == 0) {
                //     printf("%u edges processed.\n", edges_processed);
                // }
            }
        }
        edges.clear();
        printf("Edges processed: %" IDFMT "\n", edges_processed);

        // use reverse position
        IntVector in_positions_to_fill(start_pos_in_appearance_pos_lists.begin(),
                                                       start_pos_in_appearance_pos_lists.end());
        in_positions_to_fill[numOfVertices] = numOfEdges;
        const double time_sort_start = getCurrentTime();
        std::sort(position_pair.begin(), position_pair.end(), std::less<>());
        const double time_sort_end = getCurrentTime();
//        MSG(time_sort_end - time_sort_start);
        appearance_pos_lists.clear();
        appearance_pos_lists.resize(numOfEdges + num_deadend_vertices + degree_max, 0);
        NInt in_pos_pair = 0;
        for (const auto &pair : position_pair) {
            const NInt &to_id = pair.first;
            const NInt &pos = pair.second;
            NInt &in_position = in_positions_to_fill[to_id];
            assert(in_position < in_positions_to_fill[to_id + 1]);
            appearance_pos_lists[in_position] = pos;
            ++in_position;
            // if (++in_pos_pair % msg_gap == 0) {
            //     MSG(in_pos_pair);
            // }
        }

        printf("Vertices total:    %" IDFMT "\n", numOfVertices);
        printf("Vertices dead end: %" IDFMT "\n", num_deadend_vertices);
        // fill the dummy ids
        for (const NInt &id : deadend_vertices) {
            out_neighbors_lists[out_positions_to_fill[id]++] = dummy_id;
        }
        assert(get_neighbor_list_start_pos(get_dummy_id()) ==
               get_neighbor_list_start_pos(get_dummy_id() + 1));
        const double time_end = getCurrentTime();
        // printf("Graph Build Finished. TIME: %.4f\n", time_end - start);
        printf("%s\n", std::string(80, '-').c_str());
    }

    void show() const {
        // we need to show the dummy
        const NInt num_to_show = std::min(numOfVertices + 1, (NInt) 50u);
        // show the first elements
        show_vector("The Out Degrees of The Vertices:",
                    IntVector(out_degrees.data(), out_degrees.data() + num_to_show));
        show_vector("The Start Positions of The Vertices in Out Neighbor Lists:",
                    IntVector(start_pos_in_out_neighbor_lists.data(),
                                              start_pos_in_out_neighbor_lists.data() + num_to_show));
        show_vector("The In Degrees of The Vertices:",
                    IntVector(in_degrees.data(), in_degrees.data() + num_to_show));
        show_vector("The Start Positions of The Vertices in Appearance List:",
                    IntVector(start_pos_in_appearance_pos_lists.data(),
                                              start_pos_in_appearance_pos_lists.data() + num_to_show));
        // assume that the number of vertices >= the number of edges; otherwise, there is a potential bug here.
        show_vector("Out Neighbor Lists:",
                    IntVector(out_neighbors_lists.data(),
                                              out_neighbors_lists.data() +
                                              std::min(numOfEdges + num_deadend_vertices, (NInt) 50u)));
        show_vector("The Appearance Positions of Vertices in the Out Neighbor Lists:",
                    IntVector(appearance_pos_lists.data(),
                                              appearance_pos_lists.data() + std::min(numOfEdges, (NInt) 50u)));
//        show_vector("The adj list of the middel vertex", matrix[numOfVertices / 2]);
        printf("The position the id appears in outNeighbor List:\n");
        for (NInt id = 0; id < numOfVertices; ++id) {
            const NInt &idx_start = start_pos_in_appearance_pos_lists[id];
            const NInt &idx_end = start_pos_in_appearance_pos_lists[id + 1];
            printf("Id:%" IDFMT ";\tPositions: ", id);
            for (NInt index = idx_start; index < idx_end; ++index) {
                printf("%" IDFMT ", ", appearance_pos_lists[index]);
            }
            printf("\n");
        }
        show_vector("Dead End Vertices List:",
                    IntVector(deadend_vertices.data(),
                                              deadend_vertices.data() +
                                              std::min(num_deadend_vertices, (NInt) 50u)));
        printf("\n%s\n", std::string(80, '-').c_str());
    }
};


class CleanGraph {
    NInt numOfVertices = 0;
    NInt numOfEdges = 0;
public:

    void clean_graph(const std::string &_input_file,
                     const std::string &_data_folder) {
        std::ifstream inf(_input_file.c_str());
        if (!inf.is_open()) {
            printf("CleanGraph::clean_graph; File not exists.\n");
            printf("%s\n", _input_file.c_str());
            exit(1);
        }
        // status indicator
        // printf("\nReading Input Graph\n");

        std::string line;
        /**
         * skip the headers, we assume the headers are the comments that
         * begins with '#'
         */
        while (std::getline(inf, line) && line[0] == '#') {}
        if (line.empty() || !isdigit(line[0])) {
            printf("Error in CleanGraph::clean_graph. Raw File Format Error.\n");
            printf("%s\n", line.c_str());
            exit(1);
        }
        // create temporary graph
        std::vector<Edge> edges;
        numOfEdges = 0;
        /**
         * read the raw file
         */
        size_t num_lines = 0;
        // process the first line
        {
            NInt fromId, toID;
            ++num_lines;
            size_t end = 0;
            fromId = std::stoul(line, &end);
            toID = std::stoul(line.substr(end));
            // remove self-loops
            edges.emplace_back(fromId, toID);
        }
        // read the edges
        for (NInt fromId, toID; inf >> fromId >> toID;) {
            edges.emplace_back(fromId, toID);
            if (++num_lines % 5000000 == 0) { printf("%zu Valid Lines Read.\n", num_lines); }
        }

        // close the file
        inf.close();
        /* final count */
        printf("%zu Lines Read.\n", num_lines);
        numOfEdges = edges.size();
        printf("%" IDFMT "-th Non-Self Loop Edges.\n", numOfEdges);
        printf("Finish Reading.\n");
        printf("%s\n", std::string(80, '-').c_str());

        // find the maximum id
        size_t id_max = 0;
        size_t id_min = std::numeric_limits<uint64_t>::max();
        for (const auto &pair : edges) {
            id_max = std::max(id_max, (size_t) std::max(pair.from_id, pair.to_id));
            id_min = std::min(id_min, (size_t) std::min(pair.from_id, pair.to_id));
        }
        printf("Minimum ID: %zu, Maximum ID: %zu\n", id_min, id_max);
        if (id_max >= std::numeric_limits<NInt>::max()) {
            printf("Warning: Change NInt Type First.\n");
            exit(1);
        }
        const NInt one_plus_id_max = id_max + 1;
        IntVector out_degree(one_plus_id_max, 0);
        IntVector in_degree(one_plus_id_max, 0);
        // compute the degrees.
        for (const auto &edge : edges) {
            ++out_degree[edge.from_id];
            ++in_degree[edge.to_id];
        }
        // count the number of dead-end vertices
        NInt original_dead_end_num = 0;
        NInt num_isolated_points = 0;
        NInt max_degree = 0;
        for (NInt id = 0; id < one_plus_id_max; ++id) {
            if (out_degree[id] == 0) {
                ++original_dead_end_num;
                if (in_degree[id] == 0) {
                    ++num_isolated_points;
                }
            }
            // compute maximum out degree
            max_degree = std::max(out_degree[id], max_degree);
        }
        printf("The number of dead end vertices: %" IDFMT "\n", original_dead_end_num);
        printf("The number of isolated points: %" IDFMT "\n", num_isolated_points);
        printf("The maximum out degree is: %" IDFMT "\n", max_degree);

        // we assume the vertice ids are in the arrange of 0 ... numOfVertices - 1
        numOfVertices = one_plus_id_max;

        // sort the edges
        std::sort(edges.begin(), edges.end());

        // Write the attribute file
        numOfEdges = edges.size();
        std::string attribute_file = _data_folder + '/' + "attribute.txt";
        if (std::FILE *file = std::fopen(attribute_file.c_str(), "w")) {
            std::fprintf(file, "n=%" IDFMT "\nm=%" IDFMT "\n", numOfVertices, numOfEdges);
            std::fclose(file);
        } else {
            printf("Graph::clean_graph; File Not Exists.\n");
            printf("%s\n", attribute_file.c_str());
            exit(1);
        }

        // write the graph in binary
        std::string graph_bin_file = _data_folder + '/' + "graph.bin";
        if (std::FILE *file = std::fopen(graph_bin_file.c_str(), "wb")) {
            std::fwrite(edges.data(), sizeof edges[0], edges.size(), file);
            printf("Writing Binary Finished.\n");
            std::fclose(file);
        } else {
            printf("Graph::clean_graph; File Not Exists.\n");
            printf("%s\n", graph_bin_file.c_str());
            exit(1);
        }
        printf("%s\n", std::string(80, '-').c_str());
    }
};


#endif //SCARA_GRAPH_H
