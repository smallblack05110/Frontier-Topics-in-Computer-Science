// Gorgeous Block Optimization: embeds each node's first k_copy neighbors'
// adjacency lists into its sector, allowing the next search hop to skip I/O.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include "defaults.h"
#include "utils.h"
#include "cached_io.h"

namespace po = boost::program_options;
using namespace diskann;

int build_gorgeous_index(const std::string &disk_index_path,
                         const std::string &output_prefix, uint32_t k_copy)
{
    const uint64_t SECTOR_LEN = defaults::SECTOR_LEN;

    std::ifstream reader(disk_index_path, std::ios::binary);
    if (!reader.is_open())
    {
        std::cerr << "Cannot open: " << disk_index_path << std::endl;
        return -1;
    }

    uint32_t nr, nc;
    reader.read(reinterpret_cast<char *>(&nr), 4);
    reader.read(reinterpret_cast<char *>(&nc), 4);

    std::vector<uint64_t> meta(nr);
    for (uint32_t i = 0; i < nr; i++)
        reader.read(reinterpret_cast<char *>(&meta[i]), 8);

    const uint64_t npts              = meta[0];
    const uint64_t ndims             = meta[1];
    const uint64_t medoid            = meta[2];
    const uint64_t orig_max_node_len = meta[3];
    const uint64_t orig_nps          = meta[4]; // orig nnodes_per_sector
    const uint64_t frozen_num        = meta[5];
    const uint64_t frozen_loc        = meta[6];

    if (orig_nps == 0) {
        std::cerr << "Multi-sector-per-node not supported for gorgeous." << std::endl;
        return -1;
    }

    const uint64_t dbpp = ndims * sizeof(float); // disk_bytes_per_point
    const uint32_t orig_max_deg =
        static_cast<uint32_t>((orig_max_node_len - dbpp) / sizeof(uint32_t)) - 1;

    diskann::cout << "Original: npts=" << npts << " ndims=" << ndims
                  << " max_node_len=" << orig_max_node_len
                  << " nnodes_per_sector=" << orig_nps
                  << " max_degree=" << orig_max_deg << std::endl;

    // Load all adjacency lists
    const uint64_t adj_stride = (uint64_t)(orig_max_deg + 1);
    std::vector<uint32_t> all_nhood(npts * adj_stride, 0u);

    reader.seekg((std::streamoff)SECTOR_LEN, std::ios::beg);
    {
        std::vector<char> sec(SECTOR_LEN);
        uint64_t nid = 0;
        while (nid < npts) {
            reader.read(sec.data(), (std::streamsize)SECTOR_LEN);
            for (uint64_t si = 0; si < orig_nps && nid < npts; si++, nid++) {
                const char *node = sec.data() + si * orig_max_node_len;
                const uint32_t *nh = reinterpret_cast<const uint32_t *>(node + dbpp);
                uint32_t nn = std::min(nh[0], orig_max_deg);
                uint32_t *dst = all_nhood.data() + nid * adj_stride;
                dst[0] = nn;
                std::memcpy(dst + 1, nh + 1, nn * sizeof(uint32_t));
            }
        }
    }
    diskann::cout << "Adj lists loaded (" << (all_nhood.size()*4/1024/1024) << " MB)." << std::endl;

    // Compute extended parameters
    const uint64_t emb_stride    = (uint64_t)(orig_max_deg + 1) * sizeof(uint32_t);
    const uint64_t ext_node_len  = orig_max_node_len + (uint64_t)k_copy * emb_stride;
    const uint64_t ext_nps       = SECTOR_LEN / ext_node_len;

    if (ext_nps == 0) {
        std::cerr << "ext_node_len=" << ext_node_len << " > SECTOR_LEN. Reduce k_copy." << std::endl;
        return -1;
    }

    const uint64_t n_data_sectors = (npts + ext_nps - 1) / ext_nps;
    const uint64_t file_size      = (1 + n_data_sectors) * SECTOR_LEN;

    diskann::cout << "Gorgeous: k_copy=" << k_copy
                  << " ext_node_len=" << ext_node_len
                  << " ext_nps=" << ext_nps
                  << " output=" << (file_size/1024/1024) << " MB" << std::endl;

    const std::string out_path = output_prefix + "_gorgeous_disk.index";

    {
        std::ofstream writer(out_path, std::ios::binary);
        if (!writer.is_open()) {
            std::cerr << "Cannot create: " << out_path << std::endl;
            return -1;
        }

        std::vector<char> blank(SECTOR_LEN, 0);
        writer.write(blank.data(), (std::streamsize)SECTOR_LEN);

        reader.seekg((std::streamoff)SECTOR_LEN, std::ios::beg);
        std::vector<char> orig_sec(SECTOR_LEN), out_sec(SECTOR_LEN);
        uint64_t out_nid = 0, orig_consumed = 0;

        while (out_nid < npts) {
            std::memset(out_sec.data(), 0, SECTOR_LEN);
            for (uint64_t si = 0; si < ext_nps && out_nid < npts; si++, out_nid++) {
                if (orig_consumed % orig_nps == 0)
                    reader.read(orig_sec.data(), (std::streamsize)SECTOR_LEN);
                const char *orig_node =
                    orig_sec.data() + (orig_consumed % orig_nps) * orig_max_node_len;
                orig_consumed++;
                char *out_node = out_sec.data() + si * ext_node_len;
                std::memcpy(out_node, orig_node, orig_max_node_len);

                const uint32_t *x_nh = reinterpret_cast<const uint32_t *>(orig_node + dbpp);
                const uint32_t  x_nn = x_nh[0];
                for (uint32_t ci = 0; ci < k_copy; ci++) {
                    char *slot = out_node + orig_max_node_len + ci * emb_stride;
                    if (ci < x_nn) {
                        uint32_t nbr = x_nh[1 + ci];
                        if (nbr < npts)
                            std::memcpy(slot, all_nhood.data() + nbr * adj_stride, emb_stride);
                    }
                }
            }
            writer.write(out_sec.data(), (std::streamsize)SECTOR_LEN);
        }
    }

    // Write header (overwrites sector 0, no truncation since file exists)
    std::vector<uint64_t> new_meta = {
        npts, ndims, medoid,
        ext_node_len, ext_nps,
        frozen_num, frozen_loc,
        0ULL,                          // no reorder
        (uint64_t)k_copy,              // gorgeous_k_copy  (field 8)
        orig_max_node_len,             // gorgeous_orig_max_node_len (field 9)
        file_size                      // disk_file_size (field 10)
    };
    diskann::save_bin<uint64_t>(out_path, new_meta.data(), new_meta.size(), 1, 0);

    diskann::cout << "Written: " << out_path << "  (" << npts << " nodes, k_copy=" << k_copy << ")" << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    std::string index_prefix;
    uint32_t k_copy;

    po::options_description desc(
        "Build gorgeous block-optimized disk index.\n"
        "Embeds k_copy neighbor adj lists into each node block.\n"
        "Output: <prefix>_gorgeous.index");
    desc.add_options()
        ("help,h", "Print this message")
        ("index_path_prefix,i",
         po::value<std::string>(&index_prefix)->required(),
         "DiskANN index prefix (reads <prefix>_disk.index)")
        ("k_copy,k",
         po::value<uint32_t>(&k_copy)->default_value(4),
         "Number of neighbor adj lists to embed per node (default 4)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { diskann::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n" << desc << std::endl;
        return -1;
    }
    return build_gorgeous_index(index_prefix + "_disk.index", index_prefix, k_copy);
}
