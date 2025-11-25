/**
 * @file EBR.hxx
 * @brief Private header for the epoch-based reclamation domain.
 */

#ifndef CONFLUX_MEMORY_EBR_FWD_H
#define CONFLUX_MEMORY_EBR_FWD_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "Common.hxx"
#include "Conflux/Memory.hxx"
#include "detail/CRQ.hxx"

namespace conflux::ebr
{

/// Epoch value used to mark threads that are not currently participating.
constexpr std::uint64_t IDLE_EPOCH = std::numeric_limits<std::uint64_t>::max();
/// Threshold of retired objects per bin before attempting reclamation.
constexpr std::uint64_t TLS_RETIRED_THRESHOLD = 64;
/// Number of bins to stripe retire operations across.
constexpr std::uint64_t MAX_BINS = 3;

namespace detail
{

class ThreadLocalState;

/**
 * @brief Core epoch-based reclamation state shared across all handles.
 */
class DomainState
{
public:
  using QueueBase = CRQ<Retireable, 8>;

  /// Queue segment that tracks its owning domain for recycling callbacks.
  struct Segment : QueueBase
  {
    explicit Segment(DomainState *owner) noexcept : owner_{owner} {}

    DomainState *owner_{nullptr};
  };

  /// Pair of head and tail pointers for the per-bin retire queues.
  struct LimboBag
  {
    alignas(hardware_destructive_interference_size) std::atomic<Segment *> tail_seg_;
    alignas(hardware_destructive_interference_size) std::atomic<Segment *> head_seg_;
  };

  /// Per-thread record maintained while the thread is attached to the domain.
  struct ThreadRecord
  {

    std::array<std::array<Retireable *, TLS_RETIRED_THRESHOLD>, MAX_BINS> retired_lists_;
    std::array<std::size_t, MAX_BINS> retired_sizes_;

    std::atomic<std::uint64_t> epoch_;
    std::atomic<ThreadRecord *> next_available_;
    std::atomic<ThreadRecord *> next_;
    std::atomic<bool> active_;

    ThreadRecord();
  };

  DomainState();
  ~DomainState() noexcept;

  DomainState(const DomainState &) = delete;
  DomainState &operator=(const DomainState &) = delete;
  DomainState(DomainState &&) = delete;
  DomainState &operator=(DomainState &&) = delete;

  void retire(Retireable *object, void (*)(Retireable *));
  void leave();
  void try_reclaim();

  [[nodiscard]] std::uint64_t id() const noexcept
  {
    return id_.load(std::memory_order_relaxed);
  }

  void add_ref() noexcept
  {
    ref_count_.fetch_add(1, std::memory_order_relaxed);
  }
  void release_ref() noexcept;

private:
  friend class ::conflux::ebr::Domain;
  friend class ThreadLocalState;

  void enter(ThreadRecord *rec);

  /// Acquire or create a ThreadRecord for the calling thread.
  ThreadRecord *acquire_record();

  /// Release a previously acquired ThreadRecord back to the inactive list.
  void release_record(ThreadRecord *rec) noexcept;

  /// Acquire or create a queue segment from the recycled pool.
  Segment *acquire_segment();

  /// Return a queue segment to the recycled pool.
  void release_segment(Segment *) noexcept;

  /// Delete a segment via the owning domain's deleter.
  static void recycle_segment(Retireable *retired) noexcept;
  /// Retire a segment through the owning domain's Domain::retire path.
  static void retire_segment(Retireable *retired) noexcept;

  using OnSwing = void (*)(Segment *, void *);

  /// Enqueue a retireable object into the specified bin.
  void enqueue(std::size_t bin_idx, Retireable *item);

  /// Internal retire that skips triggering a reclaim pass (used when reclaiming segments).
  void retire_without_reclaim(Retireable *object, void (*deleter)(Retireable *));

  /// Attempt to dequeue a retireable object from the specified bin.
  std::optional<Retireable *> dequeue(std::size_t bin_idx, OnSwing on_swing, void *ctx);

  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> id_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> ref_count_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> epoch_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> retired_count_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> deleted_count_;
  alignas(hardware_destructive_interference_size) std::atomic<ThreadRecord *> active_thread_records_;
  alignas(hardware_destructive_interference_size) std::atomic<ThreadRecord *> inactive_thread_records_;
  alignas(hardware_destructive_interference_size) std::atomic<Segment *> recycled_segments_;
  std::array<LimboBag, MAX_BINS> retired_lists_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> active_threads_;
};

/**
 * @brief Thread-local cache for Domain attachments.
 */
class ThreadLocalState
{
public:
  ThreadLocalState() = default;
  ThreadLocalState(const ThreadLocalState &) = delete;
  ThreadLocalState &operator=(const ThreadLocalState &) = delete;
  ThreadLocalState(ThreadLocalState &&) = delete;
  ThreadLocalState &operator=(ThreadLocalState &&) = delete;

  /**
   * @brief Return a cached record for the given domain id if present.
   */
  DomainState::ThreadRecord *get_cached_rec(std::uint64_t id);

  /**
   * @brief Return cached record if present without allocating.
   */
  DomainState::ThreadRecord *get_cached_rec_no_allocate(std::uint64_t id) noexcept;

  /**
   * @brief Cache the mapping between a domain and its thread record.
   */
  void set_cached_state(std::uint64_t id, ::conflux::ebr::Domain state, DomainState::ThreadRecord *rec) noexcept;

  /**
   * @brief Clear cached mapping for a domain in the current thread.
   */
  void clear_cached_state(std::uint64_t id) noexcept;

  ~ThreadLocalState();

private:
  struct Entry
  {
    ::conflux::ebr::Domain state_;
    DomainState::ThreadRecord *rec_{nullptr};
  };

  std::vector<Entry> domains_;
};

} // namespace detail
} // namespace conflux::ebr

#endif
