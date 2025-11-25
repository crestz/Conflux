#ifndef CONFLUX_COMMON_H
#define CONFLUX_COMMON_H

#include <atomic>
#include <cstddef>
#include <new>
#include <thread>

// --- Symbol Export Macros ---
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef CONFLUX_EXPORT_SYMBOLS
#    define CONFLUX_API __declspec(dllexport)
#  else
#    define CONFLUX_API __declspec(dllimport)
#  endif
#else
#  if __GNUC__ >= 4
#    define CONFLUX_API __attribute__((visibility("default")))
#  else
#    define CONFLUX_API
#  endif
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
inline void cpu_relax()
{
  _mm_pause();
}
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
inline void cpu_relax()
{
  asm volatile("yield");
}
#else
// Fallback for unknown architectures
inline void cpu_relax()
{
  std::this_thread::yield();
}
#endif

namespace conflux
{

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
// Fallback for compilers that don't support it yet (e.g., older Clang)
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

class SpinLock
{

public:
  void lock() noexcept
  {
    // TTAS (Test-Test-And-Set) Pattern
    while (true)
    {
      // 1. Test: Spin read-only until the lock looks free.
      //    This keeps the cache line in Shared state (read-only),
      //    preventing "cache bouncing" between cores.
      while (flag_.test(std::memory_order_relaxed))
      {
        cpu_relax();
      }

      // 2. Set: Attempt to acquire the lock.
      //    This invalidates the cache line for other cores.
      //    If it fails, we go back to the read-only loop.
      if (!flag_.test_and_set(std::memory_order_acquire))
      {
        return;
      }
    }
  }

  void unlock() noexcept
  {
    flag_.clear(std::memory_order_release);
  }

private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

} // namespace conflux

#endif
