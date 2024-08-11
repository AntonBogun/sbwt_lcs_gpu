#pragma once
#include "utils.h"
#include "structs.hpp"
#include <algorithm>
#include <filesystem>
// #include <queue>
// #include <stdexcept>
// #include <utility>

// class FilesizeLoadBalancer {
//   std::vector<std::pair<std::string,std::string>> files;
//   std::map<u64, std::vector<pair<std::string, std::string>>> size_to_files;

// public:
//   FilesizeLoadBalancer(
//     const std::vector<std::string> &in_files_, const std::vector<std::string> &out_files_
//   );
//   auto partition(u64 partitions)
//     -> pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>>;

// private:
//   auto populate_size_to_files() -> void;
//   auto get_smallest_partition_index(std::vector<u64> &partition_sizes) -> u64;
// };

// using std::runtime_error;
// using std::filesystem::file_size;

// FilesizeLoadBalancer::FilesizeLoadBalancer(
//   const std::vector<std::string> &in_files_, const std::vector<std::string> &out_files_
// ):
//     in_files(in_files_), out_files(out_files_) {
//   if (in_files.size() != out_files.size()) {
//     throw std::runtime_error("Input and output file sizes differ");
//   }
//   populate_size_to_files();
// }

// auto FilesizeLoadBalancer::partition(u64 partitions)
//   -> pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> {
//   std::vector<std::vector<std::string>> in_result(partitions);
//   std::vector<std::vector<std::string>> out_result(partitions);
//   std::vector<u64> partition_sizes(partitions, 0);
//   // iterate map in reverse order
//   // NOLINTNEXTLINE (modernize-loop-convert)
//   for (auto iter = size_to_files.rbegin(); iter != size_to_files.rend();
//        ++iter) {
//     for (pair<std::string, std::string> &in_out : iter->second) {
//       auto smallest_index = get_smallest_partition_index(partition_sizes);
//       in_result[smallest_index].push_back(in_out.first);
//       out_result[smallest_index].push_back(in_out.second);
//       partition_sizes[smallest_index] += iter->first;
//     }
//   }
//   return {in_result, out_result};
// }

// auto FilesizeLoadBalancer::populate_size_to_files() -> void {
//   for (u64 i = 0; i < in_files.size(); ++i) {
//     size_to_files[file_size(in_files[i])].emplace_back(
//       std::make_pair(in_files[i], out_files[i])
//     );
//   }
// }

// auto FilesizeLoadBalancer::get_smallest_partition_index(
//   std::vector<u64> &partition_sizes
// ) -> u64 {
//   return std::min_element(partition_sizes.begin(), partition_sizes.end())
//     - partition_sizes.begin();
// }
namespace sbwt_lcs_gpu {
enum class IndexSorterMethod {
    ByIndex,
    BySort,
    ByIndexInv,
    BySortInv
};
//cmp_type=ByIndex makes ([0..n), indx_arr) into (indx_arr, [0..n)) (in-place) //!REQUIRES V_type[V_cmp_type[i]] to be valid
//cmp_type=BySort makes ([0..n), indx_arr) into (dual(indx_arr), [0..n)) (in-place)
//where dual(indx_arr) = (tmp[indx_arr[i]] = i)
template <typename impl>
class IndexSorterValue {
public:
    i64 i;

    IndexSorterValue(i64 i_) : i(i_) {}
    IndexSorterValue(const IndexSorterValue& other) : i(other.i) {}

    bool operator<(const IndexSorterValue& other) const {
        return static_cast<const impl*>(this)->cmp_le_impl(i, other.i);
    }

    friend void swap(IndexSorterValue& first, IndexSorterValue& second) {
        static_cast<impl&>(first).swap_impl(static_cast<impl&>(second));
    }


    ~IndexSorterValue() = default;
};
template <IndexSorterMethod cmp_type, typename V_type, typename V_cmp_type>
class SingleVectorISV : public IndexSorterValue<SingleVectorISV<cmp_type, V_type, V_cmp_type>> {
public:
    V_type& v;
    V_cmp_type& index;

    SingleVectorISV(i64 i_, V_type& v_, V_cmp_type& index_) : IndexSorterValue<SingleVectorISV>(i_), v(v_), index(index_) {}
    SingleVectorISV(const SingleVectorISV& other) : IndexSorterValue<SingleVectorISV>(other), v(other.v), index(other.index) {}

    bool cmp_le_impl(const i64 i, const i64 j) const {
        // return index[i] < index[j];
        if constexpr (cmp_type==IndexSorterMethod::ByIndex || cmp_type==IndexSorterMethod::BySort) {
            return index[i] < index[j];
        } else {
            return index[i] > index[j];
        }
    }
    // void swap_impl(i64 i, i64 j) {
    //     if constexpr (cmp_type==IndexSorterMethod::ByIndex) {
    //         std::swap(first.v[first.index[first.i]], second.v[second.index[second.i]]);
    //     } else {
    //         std::swap(first.v[first.i], second.v[second.i]);
    //     }
    //     std::swap(first.index[first.i], second.index[second.i]);
    // }
    void swap_impl(SingleVectorISV& other) {
        // if constexpr (cmp_type==IndexSorterMethod::ByIndex) {
        if constexpr (cmp_type==IndexSorterMethod::ByIndex || cmp_type==IndexSorterMethod::ByIndexInv) {
            std::swap(v[index[this->i]], other.v[other.index[other.i]]);
        } else {
            std::swap(v[this->i], other.v[other.i]);
        }
        std::swap(index[this->i], other.index[other.i]);
    }
};
template <IndexSorterMethod cmp_type, typename V1_type, typename V2_type, typename V_cmp_type>
class DualVectorISV : public IndexSorterValue<DualVectorISV<cmp_type, V1_type, V2_type, V_cmp_type>> {
public:
    V1_type& v1;
    V2_type& v2;
    V_cmp_type& index;

    DualVectorISV(i64 i_,V1_type& v1_, V2_type& v2_, V_cmp_type& index_) : IndexSorterValue<DualVectorISV>(i_), v1(v1_), v2(v2_), index(index_) {}
    DualVectorISV(const DualVectorISV& other) : IndexSorterValue<DualVectorISV>(other), v1(other.v1), v2(other.v2), index(other.index) {}

    bool cmp_le_impl(const i64 i, const i64 j) const {
        // return index[i] < index[j];
        if constexpr (cmp_type==IndexSorterMethod::ByIndex || cmp_type==IndexSorterMethod::BySort) {
            return index[i] < index[j];
        } else {
            return index[i] > index[j];
        }
    }
    // void swap_impl(i64 i, i64 j) {
    //     if constexpr (cmp_type==IndexSorterMethod::ByIndex) {
    //         std::swap(first.v1[first.index[first.i]], second.v1[second.index[second.i]]);
    //     } else {
    //         std::swap(first.v1[first.i], second.v1[second.i]);
    //     }
    //     std::swap(first.index[first.i], second.index[second.i]);
    // }
    void swap_impl(DualVectorISV& other) {
        if constexpr (cmp_type==IndexSorterMethod::ByIndex || cmp_type==IndexSorterMethod::ByIndexInv) {
            std::swap(v1[index[this->i]], other.v1[other.index[other.i]]);
            std::swap(v2[index[this->i]], other.v2[other.index[other.i]]);
        } else {
            std::swap(v1[this->i], other.v1[other.i]);
            std::swap(v2[this->i], other.v2[other.i]);
        }
        std::swap(index[this->i], other.index[other.i]);
    }
};

// Iterator class for IndexSorterValue
template <typename value_impl>
class IndexSorterIterator {
public:
    using value_type = value_impl;
    using difference_type = i64;
    using reference = value_type&;
    using pointer = value_type*;
private:
    value_type value;
public:
    template <typename... Args>
    IndexSorterIterator(i64 i_, Args&&... args)
        : value(i_, std::forward<Args>(args)...) {}

    //copy constructor
    IndexSorterIterator(const IndexSorterIterator& other)
        : value(other.value) {}
    
    //copy assignment
    IndexSorterIterator& operator=(const IndexSorterIterator& other) {
        // value = other.value;
        value.i = other.value.i;
        return *this;
    }
    //destructors
    ~IndexSorterIterator() = default;
    //swap
    friend void swap(IndexSorterIterator& first, IndexSorterIterator& second) {
        std::swap(first.value.i, second.value.i);
    }
    //deref
    reference operator*() {
        if(value.i >= value.index.size()||value.i < 0) {
            throw std::out_of_range("IndexSorterIterator op *: Index out of range: " + std::to_string(value.i));
        }
        return value;
    }
    pointer operator->() {
        if(value.i >= value.index.size()||value.i < 0) {
            throw std::out_of_range("IndexSorterIterator op ->: Index out of range: " + std::to_string(value.i));
        }
        return &value;
    }
    //++, --
    IndexSorterIterator& operator++() {
        ++value.i;
        return *this;
    }
    IndexSorterIterator operator++(int) {
        IndexSorterIterator tmp(*this);
        operator++();
        return tmp;
    }
    IndexSorterIterator& operator--() {
        --value.i;
        return *this;
    }
    IndexSorterIterator operator--(int) {
        IndexSorterIterator tmp(*this);
        operator--();
        return tmp;
    }
    //compare
    bool operator==(const IndexSorterIterator& other) const {
        return value.i == other.value.i;
    }
    bool operator!=(const IndexSorterIterator& other) const {
        return value.i != other.value.i;
    }

    bool operator<(const IndexSorterIterator& other) const {
        return value.i < other.value.i;
    }
    bool operator>(const IndexSorterIterator& other) const {
        return value.i > other.value.i;
    }
    bool operator<=(const IndexSorterIterator& other) const {
        return value.i <= other.value.i;
    }
    bool operator>=(const IndexSorterIterator& other) const {
        return value.i >= other.value.i;
    }
    
    //difference
    difference_type operator-(const IndexSorterIterator& other) const {
        return value.i - other.value.i;
    }
    //addition
    IndexSorterIterator operator+(difference_type n) const {
        IndexSorterIterator tmp(*this);
        tmp.value.i += n;
        return tmp;
    }
    IndexSorterIterator& operator+=(difference_type n) {
        value.i += n;
        return *this;
    }
    //subtraction
    IndexSorterIterator operator-(difference_type n) const {
        IndexSorterIterator tmp(*this);
        tmp.value.i -= n;
        return tmp;
    }
    IndexSorterIterator& operator-=(difference_type n) {
        value.i -= n;
        return *this;
    }
    //index
    reference operator[](difference_type n) {
        if(value.i + n >= value.index.size()||value.i + n < 0) {
            throw std::out_of_range("IndexSorterIterator op []: Index out of range: " + std::to_string(value.i)+"+"+std::to_string(n));
        }
        return *(*this + n);
    }
};

// Helper function to create iterators
template <typename value_impl, typename... Args>
auto make_index_sorter(i64 n,Args&&... args) {
    return std::make_pair(
        IndexSorterIterator<value_impl>(0, std::forward<Args>(args)...),
        IndexSorterIterator<value_impl>(n, std::forward<Args>(args)...)
    );
}

// inline void balance_files(std::vector<std::string> &in_files_, std::vector<std::string> &out_files_, u64 partitions) {
// std::vector<std::vector<std::tuple<std::string,std::string,i64>>> balance_files(std::vector<std::string> &in_files_, std::vector<std::string> &out_files_) {
std::vector<StreamFilenamesContainer> balance_files(std::vector<std::string> &in_files_, std::vector<std::string> &out_files_) {
    if (in_files_.size() != out_files_.size()) {
        throw std::runtime_error("Input and output file sizes differ");
    }

    // std::vector<std::pair<u64, i32>> size_to_indx;
    // for (u64 i = 0; i < in_files_.size(); ++i) {
    //     size_to_indx.push_back({std::filesystem::file_size(in_files_[i]), i});
    // }
    // std::sort(size_to_indx.begin(), size_to_indx.end());
    // std::vector<u64> partition_sizes(partitions, 0);
    // std::vector<std::vector<i64>> in_result(partitions);
    // std::vector<u64> out_order;
    // // iterate map in reverse order
    // // for (auto iter = size_to_files.rbegin(); iter != size_to_files.rend(); ++iter) {
    // //     for (std::pair<std::string, std::string> &in_out : iter->second) {
    // //         auto smallest_index =
    // //             std::min_element(partition_sizes.begin(), partition_sizes.end()) - partition_sizes.begin();
    // //         in_result[smallest_index].push_back(in_out.first);
    // //         out_result[smallest_index].push_back(in_out.second);
    // //         partition_sizes[smallest_index] += iter->first;
    // //     }
    // // }
    // for (i64 i = size_to_indx.size() - 1; i >= 0; --i) {
    //     auto smallest_index = std::min_element(partition_sizes.begin(), partition_sizes.end()) - partition_sizes.begin();
    //     in_result[smallest_index].push_back(size_to_indx[i].second);
    //     partition_sizes[smallest_index] += size_to_indx[i].first;
    // }
    // //use sizes as time
    // for(u64 i=0; i<partition_sizes.size(); i++) partition_sizes[i] = 0;
    // for (u64 i = 0; i < in_files_.size(); ++i) {
    //     for (u64 j = 0; j < partitions; ++j) {
    //         if (in_result[j].size() > i) {
    //             out_order.push_back(in_result[j][i]);
    //         }
    //     }
    // }
    
    std::vector<i64> sizes;
    using sort_value_type = DualVectorISV<IndexSorterMethod::BySort, std::vector<std::string>, std::vector<std::string>, std::vector<i64>>;
    for (i64 i = 0; i < in_files_.size(); ++i) {
        sizes.push_back(((i64)std::filesystem::file_size(in_files_[i])));//sort from smallest to largest
    }
    auto sorter = make_index_sorter<sort_value_type>(in_files_.size(), in_files_, out_files_, sizes);
    quickSort(sorter.first, sorter.second);
    // std::vector<std::vector<std::tuple<std::string,std::string,i64>>> result;
    std::vector<StreamFilenamesContainer> result;
    // std::vector<i64> stream_sizes;
    if (in_files_.size()){
        result.push_back({}); //initialize first partition
        // stream_sizes.push_back(0);
    }
    for (i64 j = 0; j < in_files_.size(); ++j) {
        // if (stream_sizes.back()>desired_stream_size) {
        if (result.back().total_length>desired_stream_size) {
            result.push_back({});
            // stream_sizes.push_back(0);
        }
        // result.back().push_back({in_files_[j], out_files_[j], sizes[j]});
        result.back().filenames.push_back(in_files_[j]);
        result.back().output_filenames.push_back(out_files_[j]);
        result.back().lengths.push_back(sizes[j]);
        result.back().total_length += sizes[j];
        // stream_sizes.back() += sizes[j];
    }
    // std::reverse(result.begin(), result.end());
    // using sort_value_type2 = SingleVectorISV<IndexSorterMethod::BySortInv, decltype(result), decltype(stream_sizes)>;
    // auto sorter2 = make_index_sorter<sort_value_type2>(result.size(), result, stream_sizes);
    // quickSort(sorter2.first, sorter2.second);
    std::sort(result.begin(), result.end(), [](const StreamFilenamesContainer& a, const StreamFilenamesContainer& b) {
        return a.total_length > b.total_length;
    });
    // swap(result[0], result.back());
    return std::move(result);//return largest streams first
}
} // namespace sbwt_lcs_gpu