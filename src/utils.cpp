#include "utils.h"
namespace sbwt_lcs_gpu {
// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g
#ifdef __linux__

#include <unistd.h>
u64 get_total_cpu_memory() {
  u64 pages = sysconf(_SC_PHYS_PAGES);
  u64 page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
u64 get_free_cpu_memory() {
  u64 pages = sysconf(_SC_AVPHYS_PAGES);
  u64 page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

#elif _WIN32

#include <windows.h>
u64 get_total_cpu_memory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}
u64 get_free_cpu_memory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullAvailPhys;
}

#endif

} // namespace sbwt_lcs_gpu