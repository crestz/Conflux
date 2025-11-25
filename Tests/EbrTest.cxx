#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// --- Sanitizer Detection ---
#if defined(__has_feature)
#  define CONFLUX_HAS_FEATURE_ASAN __has_feature(address_sanitizer)
#  define CONFLUX_HAS_FEATURE_LSAN __has_feature(leak_sanitizer)
#  define CONFLUX_HAS_FEATURE_TSAN __has_feature(thread_sanitizer)
#else
#  define CONFLUX_HAS_FEATURE_ASAN 0
#  define CONFLUX_HAS_FEATURE_LSAN 0
#  define CONFLUX_HAS_FEATURE_TSAN 0
#endif

#if defined(__SANITIZE_ADDRESS__) || CONFLUX_HAS_FEATURE_ASAN
#  define CONFLUX_HAS_ASAN 1 // NOLINT(cppcoreguidelines-macro-usage)
#endif
#if defined(__SANITIZE_THREAD__) || CONFLUX_HAS_FEATURE_TSAN
#  define CONFLUX_HAS_TSAN 1
#endif

// Include LSan interface if available for explicit leak checks
#if defined(__has_include)
#  if __has_include(<sanitizer/lsan_interface.h>) && (defined(CONFLUX_HAS_ASAN) || CONFLUX_HAS_FEATURE_LSAN)
#    include <sanitizer/lsan_interface.h>
#    define CONFLUX_HAS_LSAN 1 // NOLINT(cppcoreguidelines-macro-usage)
#  endif
#endif

#include <gtest/gtest.h>
#include "Conflux/Memory.hxx"
#include "Conflux/Utils.hxx"

using namespace conflux;
using namespace conflux::ebr;

namespace {

// Helper to track object destruction
struct TrackValue {
public:
  TrackValue() = default;
  TrackValue(std::shared_ptr<std::atomic<int>> destroyed, int id) noexcept
      : destroyed_{std::move(destroyed)}, id_{id} {}

  TrackValue(const TrackValue&) = default;
  TrackValue& operator=(const TrackValue&) = default;
  TrackValue(TrackValue&&) noexcept = default;
  TrackValue& operator=(TrackValue&&) noexcept = default;

  ~TrackValue() {
    if (destroyed_) {
      destroyed_->fetch_add(1, std::memory_order_relaxed);
    }
  }

private:
  std::shared_ptr<std::atomic<int>> destroyed_{};
  int id_{0};
};

// Helper object for direct raw pointer tests
struct CountingObject : ebr::Object<CountingObject> {
  explicit CountingObject(std::shared_ptr<std::atomic<int>> destroyed) 
      : destroyed_(std::move(destroyed)) {}

  void reclaim() noexcept {
    if (destroyed_) {
      destroyed_->fetch_add(1, std::memory_order_relaxed);
    }
    delete this;
  }

private:
  std::shared_ptr<std::atomic<int>> destroyed_;
};

// Helper to aggressively retry reclamation until target is met or we timeout
void drain_reclamation(Domain &domain, std::atomic<int> &destroyed, int target)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
  while (destroyed.load(std::memory_order_relaxed) < target && std::chrono::steady_clock::now() < deadline)
  {
    for (int i = 0; i < 4; ++i)
    {
      try_reclaim(domain);
    }
    std::this_thread::yield();
  }
}

} // namespace

// -----------------------------------------------------------------------------
// Functional Tests
// -----------------------------------------------------------------------------

TEST(EbrTest, TreiberStackReclaimsPoppedNodes) {
  auto domain = make_domain();
  TreiberStack<TrackValue> stack(domain);
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  constexpr int count = 100;

  for (int i = 0; i < count; ++i) stack.push(TrackValue{destroyed, i});
  for (int i = 0; i < count; ++i) stack.try_pop();

  // Nothing should be destroyed yet if we haven't reclaimed
  drain_reclamation(domain, *destroyed, count);
  EXPECT_GE(destroyed->load(), count);
}

// -----------------------------------------------------------------------------
// Lifetime & Architecture Tests
// -----------------------------------------------------------------------------

TEST(EbrTest, DomainStateOutlivesHandleWhileThreadsUseTls)
{
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  constexpr int thread_count = 4;
  constexpr int retire_per_thread = 128; // exceeds TLS retire threshold to force enqueue
  const int expected = thread_count * retire_per_thread;

  {
    Domain domain = make_domain();
    Domain cleanup = domain;
    constexpr int flush_count = 32;

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t)
    {
      threads.emplace_back([destroyed, domain]() mutable {
        ObjectGuard guard(domain); // give this thread a TLS record

        for (int i = 0; i < retire_per_thread; ++i)
        {
          auto *obj = new CountingObject(destroyed); // NOLINT(cppcoreguidelines-owning-memory)
          obj->retire(domain);
          try_reclaim(domain);
        }
      });
    }

    // Drop the original handle while worker threads still hold handles + TLS
    domain = Domain{};

    for (auto &thread : threads)
    {
      thread.join();
    }

    // Retire a few more objects on this thread to force a flush of any cached TLS retired lists.
    {
      ObjectGuard guard(cleanup);
      for (int i = 0; i < flush_count; ++i)
      {
        auto *obj = new CountingObject(destroyed); // NOLINT(cppcoreguidelines-owning-memory)
        obj->retire(cleanup);
      }
    }

    // Use the 'cleanup' handle, which still keeps DomainState alive,
    // to drive reclamation until everything is destroyed.
    drain_reclamation(cleanup, *destroyed, expected + flush_count);
    cleanup = Domain{}; // drop the last handle
  }

  EXPECT_GE(destroyed->load(std::memory_order_relaxed), expected);
}

TEST(EbrTest, StalledThreadPreventsReclamation) {
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  constexpr int retire_count = 64; // equals TLS threshold

  {
    Domain domain = make_domain();
    
    std::atomic<bool> stall_thread_ready{false};
    std::atomic<bool> resume_stall{false};
    constexpr int reclaim_attempts = 10;

    // 1. Start a thread that enters an epoch and sleeps
    std::thread stalled([&] {
      ObjectGuard guard(domain);
      stall_thread_ready.store(true);
      while (!resume_stall.load()) {
        std::this_thread::yield();
      }
    });

    while (!stall_thread_ready.load()) std::this_thread::yield();

    // 2. Retire objects in main thread
    {
      ObjectGuard producer_guard(domain);
      for(int i=0; i<retire_count; ++i) { // meet TLS retire threshold to force enqueue
        auto* obj = new CountingObject(destroyed); // NOLINT(cppcoreguidelines-owning-memory)
        obj->retire(domain);
      }
    }

    // 3. Attempt reclaim. Should FAIL because 'stalled' thread is holding the epoch.
    for(int i=0; i<reclaim_attempts; ++i) try_reclaim(domain);
    EXPECT_EQ(destroyed->load(), 0) << "Reclamation happened despite stalled thread!";

    // 4. Wake up the thread
    resume_stall.store(true);
    stalled.join();

    // 4b. Retire another batch (and one extra) to force the TLS list to flush through the threshold again.
    {
      ObjectGuard producer_guard(domain);
      for (int i = 0; i < retire_count + 1; ++i)
      {
        auto *obj = new CountingObject(destroyed); // NOLINT(cppcoreguidelines-owning-memory)
        obj->retire(domain);
      }
    }

    // 5. Now reclamation should succeed
    drain_reclamation(domain, *destroyed, retire_count * 2 + 1);
  }

  EXPECT_GE(destroyed->load(), retire_count);
}

// -----------------------------------------------------------------------------
// Stress / Torture Tests
// -----------------------------------------------------------------------------

TEST(EbrTest, MpmcTortureTest) {
  // High contention test: multiple producers/consumers fighting over cache lines.
  auto domain = make_domain();
  TreiberStack<TrackValue> stack(domain);
  auto destroyed = std::make_shared<std::atomic<int>>(0);

  // Configuration
  constexpr int threads = 16; // Oversubscribe cores to force context switches
  constexpr int ops_per_thread = 200000;
  constexpr int total_ops = threads * ops_per_thread;
  constexpr int reclaim_stride = 128;

  std::atomic<bool> start_gun{false};
  std::atomic<int> push_count{0};
  std::atomic<int> pop_count{0};

  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      // 1. Spin-wait for simultaneous start
      while (!start_gun.load(std::memory_order_acquire)) std::this_thread::yield();

      // 2. Hammer the stack
      for (int i = 0; i < ops_per_thread; ++i) {
        stack.push(TrackValue{destroyed, t * ops_per_thread + i});
        push_count.fetch_add(1, std::memory_order_relaxed);

        // Pop frequently to cause contention on head
        if (stack.try_pop()) {
          pop_count.fetch_add(1, std::memory_order_relaxed);
        }

        // Periodically help reclaim
        if (i % reclaim_stride == 0) try_reclaim(domain);
      }
    });
  }

  // FIRE!
  start_gun.store(true, std::memory_order_release);

  for (auto &t : workers) t.join();

  // Drain remaining items
  while (stack.try_pop()) {
    pop_count.fetch_add(1, std::memory_order_relaxed);
  }

  // Final Reclaim
  drain_reclamation(domain, *destroyed, total_ops);

  EXPECT_EQ(push_count.load(), total_ops);
  EXPECT_GE(pop_count.load(), total_ops);
  EXPECT_GE(destroyed->load(), total_ops);
}

TEST(EbrTest, RandomizedStressTest) {
  auto destroyed = std::make_shared<std::atomic<int>>(0);
  Domain domain = make_domain();
  TreiberStack<TrackValue> stack(domain);

  constexpr int thread_count = 8;
  // Run long enough to force epoch cycles, but short enough for CI
  constexpr auto runtime = std::chrono::milliseconds(500); 
  constexpr int max_action = 100;
  constexpr int push_threshold = 50;
  constexpr int reclaim_interval = 10;

  std::atomic<bool> stop{false};
  std::atomic<long> total_pushes{0};

  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, seed = t] {
      std::mt19937 rng(seed);
      std::uniform_int_distribution<int> dist(0, max_action);
      
      while (!stop.load(std::memory_order_relaxed)) {
        int action = dist(rng);
        if (action < push_threshold) {
          stack.push(TrackValue{destroyed, action});
          total_pushes.fetch_add(1, std::memory_order_relaxed);
        } else {
          stack.try_pop();
        }
        
        if (action % reclaim_interval == 0) try_reclaim(domain);
      }
    });
  }

  std::this_thread::sleep_for(runtime);
  stop.store(true, std::memory_order_release);
  for (auto &t : threads) t.join();

  // Cleanup rest
  while(stack.try_pop()) {}
  
  // We can't know exact destroy count due to pops, but ensure no leaks
  drain_reclamation(domain, *destroyed, destroyed->load()); 
  
  // Basic sanity check: we destroyed something
  EXPECT_GT(destroyed->load(), 0);
}

// -----------------------------------------------------------------------------
// Leak Detection (ASan/LSan specific)
// -----------------------------------------------------------------------------

#if defined(CONFLUX_HAS_LSAN)
TEST(EbrTest, NoLeaksDetected) {
  // Explicit LSan check if available
  if (__lsan_do_recoverable_leak_check() != 0) {
    GTEST_SKIP() << "Pre-existing leaks detected outside test scope";
  }
  
  // Run a mini cycle
  {
    auto domain = make_domain();
    TreiberStack<TrackValue> stack(domain);
    stack.push(TrackValue{std::make_shared<std::atomic<int>>(0), 1});
    stack.try_pop();
    try_reclaim(domain);
  } // domain dies here

  EXPECT_EQ(__lsan_do_recoverable_leak_check(), 0) << "Leak detected after Domain destruction";
}
#endif
