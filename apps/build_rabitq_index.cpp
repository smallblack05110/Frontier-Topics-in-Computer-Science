
#include "common_includes.h"
#include <boost/program_options.hpp>
#include "utils.h"
#include "rabitq.h"
#include <random>
namespace po = boost::program_options;
using namespace diskann;

int build_rabitq(const std::string &base_path, const std::string &out_prefix)
{
    float *base_data = nullptr;
    size_t N, D, aligned_dim;
    diskann::load_aligned_bin<float>(base_path, base_data, N, D, aligned_dim);
    diskann::cout << "Loaded " << N << " vectors D=" << D << std::endl;
    const uint32_t n_words = (D + 63u) / 64u;

    // Per-dim centers
    std::vector<double> cd(D, 0.0);
    for (size_t i = 0; i < N; i++) {
        const float *x = base_data + i * aligned_dim;
        for (uint32_t d = 0; d < D; d++) cd[d] += x[d];
    }
    std::vector<float> centers(D);
    for (uint32_t d = 0; d < D; d++) centers[d] = (float)(cd[d] / (double)N);
    diskann::cout << "Centers computed." << std::endl;

    // Random signs for SRHT (fixed seed=42)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<int8_t> signs(D);
    for (uint32_t d = 0; d < D; d++) signs[d] = dist(rng) ? 1 : -1;

    // Binary codes + norms (apply SRHT before binarize)
    std::vector<uint64_t> codes(N * n_words, 0ULL);
    std::vector<float> norms_prime(N, 0.f);
    std::vector<float> buf(D);
    for (size_t i = 0; i < N; i++) {
        const float *x = base_data + i * aligned_dim;
        float sq = 0.f;
        for (uint32_t d = 0; d < D; d++) {
            float xp = x[d] - centers[d];
            sq += xp * xp;
            buf[d] = xp;
        }
        norms_prime[i] = std::sqrt(sq);
        // Normalize before SRHT so binary codes capture pure direction
        float norm_inv = (sq > 1e-12f) ? 1.f / norms_prime[i] : 0.f;
        for (uint32_t d = 0; d < D; d++) buf[d] *= norm_inv;
        // Apply SRHT to unit vector
        diskann::srht_inplace(buf.data(), signs.data(), (uint32_t)D);
        // Binarize rotated vector
        uint64_t *code = codes.data() + i * n_words;
        for (uint32_t d = 0; d < D; d++)
            if (buf[d] >= 0.f) code[d/64] |= (1ULL << (d%64));
    }
    diskann::cout << "Codes computed (with SRHT)." << std::endl;

    // Write output
    const std::string out_path = out_prefix + "_rabitq.bin";
    std::ofstream f(out_path, std::ios::binary);
    if (!f) { std::cerr << "Cannot write: " << out_path << std::endl; return -1; }
    uint64_t hdr[3] = {(uint64_t)N, (uint64_t)D, (uint64_t)n_words};
    f.write((char*)hdr, sizeof(hdr));
    f.write((char*)centers.data(), D * sizeof(float));
    f.write((char*)signs.data(), D * sizeof(int8_t));  // NEW: store signs
    f.write((char*)codes.data(), N * n_words * sizeof(uint64_t));
    f.write((char*)norms_prime.data(), N * sizeof(float));
    f.close();
    double mb = (3.0*8 + D*4 + D + (double)N*n_words*8 + (double)N*4) / 1048576.0;
    diskann::cout << "Written: " << out_path << "  (" << mb << " MB)" << std::endl;
    delete[] base_data;
    return 0;
}

int main(int argc, char **argv)
{
    std::string base_path, prefix;
    po::options_description desc("Build RaBitQ+SRHT quantization index");
    desc.add_options()
        ("help,h", "Print help")
        ("base_path,b", po::value<std::string>(&base_path)->required(), "Base vectors .fbin")
        ("index_path_prefix,i", po::value<std::string>(&prefix)->required(), "Output prefix");
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { diskann::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n" << desc << std::endl; return -1;
    }
    return build_rabitq(base_path, prefix);
}
