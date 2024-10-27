#pragma once
// clang-format off
#include "gpu/gpu_utils.h"
#include "utils.h"
#include "poppy.hpp"
namespace sbwt_lcs_gpu {


class SBWTContainer {
    public:
    u64 num_bits;//number of bits in bitvector
    u64 bitvector_size;//number of u64s in bitvector
    u64 kmer_size;
  
    SBWTContainer() : num_bits(0), bitvector_size(0), kmer_size(0) {}

    void initialize(u64 num_bits_, u64 bitvector_size_, u64 kmer_size_) {
        num_bits = num_bits_;
        bitvector_size = bitvector_size_;
        kmer_size = kmer_size_;
    }
    u64 get_bit_vector_size() const { return bitvector_size; }
    u64 get_num_bits() const { return num_bits; }
    u64 get_kmer_size() const { return kmer_size; }

//   public:
    // u64 SBWTContainer::get_bit_vector_size() const { return bit_vector_size; }
    // u64 SBWTContainer::get_num_bits() const { return num_bits; }
    // u64 SBWTContainer::get_kmer_size() const { return kmer_size; }
};

class SBWTContainerCPU : public SBWTContainer {
    // std::vector<u64> key_kmer_marks;

    // FCVector<std::vector<u64>>& acgt_,
    // FCVector<Poppy>& poppys_,
    // FCVector<u64>& c_map_,
  public:
    FCVector<std::vector<u64>> bitvectors;
    FCVector<Poppy> poppys;
    FCVector<u64> c_map;
    // std::vector<u64>& key_kmer_marks
    // SBWTContainerCPU(u64 num_bits, u64 bit_vector_size, u64 kmer_size, u64 key_kmer_alloc_size=0):
    //   SBWTContainer(num_bits, bit_vector_size, kmer_size),
    //   bitvectors(alphabet_size), poppys(alphabet_size), c_map(cmap_size), key_kmer_marks(key_kmer_alloc_size) {}
    SBWTContainerCPU() 
        : SBWTContainer(), bitvectors(alphabet_size), poppys(alphabet_size), c_map(cmap_size) {}

    // void initialize(u64 num_bits, u64 bitvector_size, u64 kmer_size, u64 key_kmer_alloc_size) {
    void initialize(u64 num_bits, u64 bitvector_size, u64 kmer_size) {
        SBWTContainer::initialize(num_bits, bitvector_size, kmer_size);
        // key_kmer_marks.resize(key_kmer_alloc_size);
        for (i32 i = 0; i < alphabet_size; i++) {
            bitvectors.emplace_back();
            poppys.emplace_back();
        }
    }
    void load_from_file(const std::string &filename) {
        ThrowingIfstream in_stream(filename, std::ios::in);
        const std::string variant = in_stream.read_string_with_size();
        if (variant != "v0.1") { // may not contain variant string
            if (variant != "plain-matrix") {
                throw std::runtime_error("Error input is not a plain-matrix SBWT");
            }
            const std::string version = in_stream.read_string_with_size();
            if (version != "v0.1") {
                throw std::runtime_error("Error: wrong SBWT version");
            }
        }
        u64 num_bits = in_stream.read<u64>();
        const u64 bitvector_bytes = ceil_div(num_bits, u64_bits) * sizeof(u64); // based on how sdsl stores bitvectors
        u64 kmer_size = -1;
        // seek back by u64
        in_stream.seekg(static_cast<std::ios::off_type>(-sizeof(u64)), std::ios::cur);
        const u64 vectors_start_position = in_stream.tellg();

        //$ segment is 1 size
        for (i32 i = 0; i < cmap_size; i++) c_map.emplace_back(1);
        kmer_size = read_k(in_stream);
        in_stream.close();

        initialize(num_bits, bitvector_bytes / sizeof(u64), kmer_size);

    #pragma omp parallel for
        for (u64 i = 0; i < alphabet_size; ++i) {
            std::ifstream st(filename);

            u64 pos = vectors_start_position + sizeof(u64) + i * (bitvector_bytes + sizeof(u64));
            st.seekg(static_cast<std::ios::off_type>(pos), std::ios::beg);

            bitvectors[i] = std::vector<u64>(bitvector_bytes / sizeof(u64) + bitvector_pad_u64s);
            st.read(reinterpret_cast<char *>(bitvectors[i].data()), static_cast<std::streamsize>(bitvector_bytes));

            poppys[i].build(bitvectors[i], num_bits + bitvector_pad_u64s*u64_bits);

            c_map[i + 1] = poppys[i].total_1s;
        }
        for (int i = 0; i < 4; ++i) {
            c_map[i + 1] += c_map[i];
        }
    }
    void print_info() {
        std::cout << "num_bits: " << num_bits << std::endl;
        std::cout << "bitvector_size: " << bitvector_size << std::endl;
        std::cout << "kmer_size: " << kmer_size << std::endl;
        std::cout << "cmap: "<<print_vec_new(c_map)<<std::endl;
        std::cout << "layer_0 len: " << poppys[0].layer_0.size() << std::endl;
        std::cout << "layer_1_2 len: " << poppys[0].layer_1_2.size() << std::endl;
    }

    void clear() {
        bitvectors.clear();
        poppys.clear();
        c_map.clear();
    }
  private:
    u64 read_k(std::istream &in_stream) {
        u64 kmer_size = -1;
        // in_stream.seekg(static_cast<std::ios::off_type>(bit_vector_bytes), std::ios::cur); // skip first vector
        // skip_unecessary_dbg_components(in_stream);

        //*skip_unecessary_dbg_components

        // skip acgt vectors and 4 rank structure vectors
        // all are stored as num_bits,vector..., but vector is ceil_div(num_bits, u64_bits) * sizeof(u64)
        // for (int i = 0; i < 3 + 4; ++i) { //was 3 because didn't seek back by u64
        for (int i = 0; i < alphabet_size * 2; ++i) {
            skip_bits_vector(in_stream);
        }
        skip_bits_vector(in_stream);                 // skip suffix group starts
        skip_bytes_vector(in_stream);                // skip C map
        skip_bytes_vector(in_stream);                // skip kmer_prefix_calc
        in_stream.seekg(sizeof(u64), std::ios::cur); // skip precalc_k
        in_stream.seekg(sizeof(u64), std::ios::cur); // skip n_nodes
        in_stream.seekg(sizeof(u64), std::ios::cur); // skip n_kmers
        in_stream.read(reinterpret_cast<char *>(&kmer_size), sizeof(u64));
        // Logger::log(Logger::LOG_LEVEL::DEBUG, prints_new("Using kmer size:", kmer_size));
        return kmer_size;
    }


    // std::vector<u64> get_key_kmer_marks() {
    //     if (colors_filename.empty()) {
    //         return {};
    //     }
    //     ThrowingIfstream in_stream(colors_filename, std::ios::in | std::ios::binary);
    //     std::string filetype = in_stream.read_string_with_size();
    //     if (filetype != "sdsl-hybrid-v4") {
    //         throw std::runtime_error("The colors file has an incorrect format. Expected 'sdsl-hybrid-v4'");
    //     }
    //     skip_unecessary_colors_components(in_stream);
    //     u64 num_bits = in_stream.read<u64>();
    //     // const u64 bit_vector_bytes = round_up<u64>(num_bits, u64_bits) / sizeof(u64);
    //     const u64 bit_vector_bytes = ceil_div(num_bits, u64_bytes);
    //     std::vector<u64> key_kmer_marks(bit_vector_bytes / sizeof(u64));
    //     in_stream.read(reinterpret_cast<char *>(key_kmer_marks.data()),
    //     static_cast<std::streamsize>(bit_vector_bytes)); return key_kmer_marks;
    // }

    // void skip_unecessary_colors_components(std::istream &in_stream) {
    //     sdsl::int_vector<> vector_discard;
    //     sdsl::rank_support_v5 rank_discard;
    //     sdsl::bit_vector bit_discard;

    //     bit_discard.load(in_stream);    // skip dense_arrays
    //     vector_discard.load(in_stream); // skip dense_arrays_intervals

    //     vector_discard.load(in_stream); // skip sparse_arrays
    //     vector_discard.load(in_stream); // skip sparse_arrays_intervals

    //     bit_discard.load(in_stream); // skip is_dense_marks
    //     // skip is_dense_marks rank structure
    //     rank_discard.load(in_stream, &bit_discard);
    // }
    // acgt(acgt_), poppys(poppys_), c_map(c_map_), key_kmer_marks(key_kmer_marks) {}
    //   [[nodiscard]] SBWTContainerGPU to_gpu() const; //should be done in SBWTContainerGPU instead
};
// clang-format on

class SBWTContainerGPU : public SBWTContainer {
  public:
    // FCVector<GpuPointer<u64>> acgt, layer_0, layer_1_2;
    FCVector<GpuPointer<u64>> bitvectors, layer_0, layer_1_2;
    GpuPointer<u64> c_map;
    // GpuPointer<u64> presearch_left, presearch_right;
    GpuPointer<u64 *> bitvector_pointers, layer_0_pointers, layer_1_2_pointers;
    // GpuPointer<u64> key_kmer_marks;
    // template <typename V_type>
    static GpuPointer<u64*> extract_bitvector_pointers(const FCVector<std::vector<u64>> &pointers) {
        std::vector<u64 *> result;
        result.reserve(pointers.size());
        for (i32 i = 0; i < pointers.size(); i++) {
            result.emplace_back(const_cast<u64*>(pointers[i].data()));
        }
        return GpuPointer<u64 *>(result);
    }
    static GpuPointer<u64 *> extract_layer0_pointers(const FCVector<Poppy> &poppys) {
        std::vector<u64 *> result;
        result.reserve(poppys.size());
        for (i32 i = 0; i < poppys.size(); i++) {
            // result.push_back(const_cast<u64 *>(poppys[i].layer_0.data()));
            result.emplace_back(const_cast<u64*>(poppys[i].layer_0.data()));
        }
        return GpuPointer<u64 *>(result);
    }
    static GpuPointer<u64 *> extract_layer1_2_pointers(const FCVector<Poppy> &poppys) {
        std::vector<u64 *> result;
        result.reserve(poppys.size());
        for (i32 i = 0; i < poppys.size(); i++) {
            // result.push_back(const_cast<u64 *>(poppys[i].layer_1_2.data()));
            result.emplace_back(const_cast<u64*>(poppys[i].layer_1_2.data()));
        }
        return GpuPointer<u64 *>(result);
    }
    // SBWTContainerGPU(const vector<vector<u64>> &cpu_acgt, const vector<Poppy> &cpu_poppy, const vector<u64> &cpu_c_map,
    //                  u64 bits_total, u64 bit_vector_size, u32 kmer_size, const vector<u64> &cpu_key_kmer_marks)
    SBWTContainerGPU(const SBWTContainerCPU &cpu_sbwt)
    // : SBWTContainer(), max_index(bits_total) {
    : SBWTContainer(),
    bitvectors(alphabet_size),
    layer_0(alphabet_size),
    layer_1_2(alphabet_size),
    c_map(cpu_sbwt.c_map.data(), cpu_sbwt.c_map.size()),
    bitvector_pointers(extract_bitvector_pointers(cpu_sbwt.bitvectors)),
    layer_0_pointers(extract_layer0_pointers(cpu_sbwt.poppys)),
    layer_1_2_pointers(extract_layer1_2_pointers(cpu_sbwt.poppys)) 
    {
        // SBWTContainer::initialize(bits_total, bit_vector_size, kmer_size);
        SBWTContainer::initialize(cpu_sbwt.num_bits, cpu_sbwt.bitvector_size, cpu_sbwt.kmer_size);
        // acgt.reserve(4);
        for (u64 i = 0; i < 4; ++i) {
            // acgt.push_back(GpuPointer<u64>(cpu_acgt[i]));
            bitvectors.emplace_back(cpu_sbwt.bitvectors[i]);
            // layer_0.push_back(GpuPointer<u64>(cpu_poppy[i].layer_0));
            layer_0.emplace_back(cpu_sbwt.poppys[i].layer_0);
            // layer_1_2.push_back(GpuPointer<u64>(cpu_poppy[i].layer_1_2));
            layer_1_2.emplace_back(cpu_sbwt.poppys[i].layer_1_2);
        }
    }
};

} // namespace sbwt_lcs_gpu
