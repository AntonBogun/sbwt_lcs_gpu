
#pragma once
#include <map>
#include <chrono>
#include <iomanip>
#include "utils.h"
namespace sbwt_lcs_gpu {

class Timer {
  private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> start_times;

  public:
    void start(const std::string &name) {
        start_times[name] = std::chrono::high_resolution_clock::now();
        std::cout << "[" << get_current_time() << "] " << name << " started\n";
    }
    void stop(const std::string &name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = start_times[name];
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "[" << get_current_time() << "] " << name << " ended, took " << duration << " s\n";
    }

  private:
    std::string get_current_time() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }
};

} // namespace sbwt_lcs_gpu