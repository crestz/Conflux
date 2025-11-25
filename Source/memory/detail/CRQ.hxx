/**
 * @file CRQ.hxx
 * @brief Cache-friendly ring queue used internally by EBR segments.
 */

#ifndef CONFLUX_MEMORY_DETAIL_CRQ_H
#define CONFLUX_MEMORY_DETAIL_CRQ_H

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "Common.hxx"
#include "Conflux/Memory.hxx"

namespace conflux::ebr::detail
{

/**
 * @brief Cache-friendly ring queue used by EBR segments to batch retireable objects.
 *
 * @tparam T element type stored in each cell (must satisfy Retireable).
 * @tparam PowerTwo log2 of the number of cells in the queue.
 */
template <typename T, size_t PowerTwo> class CRQ : public Retireable
{

  static constexpr std::uint64_t ZERO_BITS = std::countr_zero(alignof(T));
  static constexpr std::uint64_t NUM_CELLS = (1ULL << PowerTwo);
  static constexpr std::uint64_t NIL = 0;

  static_assert(ZERO_BITS >= 1 && "alignment of type T must be >= 2.");

  /// Returns a per-thread marker used to reserve a cell.
  [[nodiscard]] inline std::uintptr_t token() noexcept
  {
    static thread_local T *x;
    // Cast to uintptr_t before bit manipulation
    return reinterpret_cast<std::uintptr_t>(&x) | 1ULL;
  }

  /// Checks whether the stored value is a token marker.
  [[nodiscard]] inline bool is_token(std::uintptr_t value) noexcept
  {
    return (value & 1ULL) == 1ULL;
  }

  /// Checks whether the stored value represents an empty cell.
  [[nodiscard]] inline bool is_nil(std::uintptr_t value) noexcept
  {
    return value == NIL;
  }

  /// Checks whether the stored value is a properly aligned payload pointer.
  [[nodiscard]] inline bool is_value(std::uintptr_t value) noexcept
  {
    // It is a value if it is NOT NIL and the lower alignment bits are clean
    return value != NIL && (value & ((1ULL << ZERO_BITS) - 1)) == 0;
  }

  [[nodiscard]] inline std::uint64_t extract_epoch(std::uint64_t packed) noexcept
  {
    return packed & ((1ULL << 63) - 1);
  }

  [[nodiscard]] inline bool extract_safe(std::uint64_t packed) noexcept
  {
    return (packed & (1ULL << 63)) != 0ULL;
  }

  [[nodiscard]] inline std::uint64_t pack(bool safe, std::uint64_t cycle) noexcept
  {
    return (safe ? 1ULL << 63 : 0ULL) | (cycle & ((1ULL << 63) - 1ULL));
  }

  struct alignas(hardware_destructive_interference_size) Cell
  {
    alignas(hardware_destructive_interference_size) std::atomic<std::uintptr_t> value_;
    alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> safe_and_epoch_;
  };

public:
  /**
   * @brief Construct an empty queue and initialize all cells as safe and empty.
   */
  CRQ() : head_{NUM_CELLS}, tail_{NUM_CELLS}, next_{nullptr}, closed_{false}
  {
    constexpr std::uint64_t sne = 1ULL << 63;
    for (auto &cell : cells_)
    {
      cell.safe_and_epoch_.store(sne, std::memory_order_relaxed);
      cell.value_.store(NIL, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Enqueue a pointer into the ring.
   *
   * @param item pointer to store.
   * @return true if the item was enqueued, false if the queue is closed/overflowed.
   */
  bool enqueue(T *item) noexcept
  {
    for (;;)
    {
      const std::uint64_t t = tail_.fetch_add(1, std::memory_order_relaxed);
      if (closed_.load(std::memory_order_acquire))
        return false;

      const std::uint64_t cycle = t >> PowerTwo;
      const std::uint64_t index = t & (NUM_CELLS - 1);

      auto &cell = cells_[index];
      const std::uint64_t sne = cell.safe_and_epoch_.load(std::memory_order_acquire);
      const std::uintptr_t value = cell.value_.load(std::memory_order_acquire);

      bool safe = extract_safe(sne);
      std::uint64_t epoch = extract_epoch(sne);
      if (is_nil(value) || is_token(value) && epoch < cycle && (safe || head_.load(std::memory_order_acquire) <= t))
      {
        std::uintptr_t old_value = value;
        std::uintptr_t tok = token();
        if (!cell.value_.compare_exchange_strong(old_value, tok, std::memory_order_acq_rel, std::memory_order_relaxed))
          goto checkOverflow;

        std::uint64_t old_safe_and_epoch = sne;
        if (!cell.safe_and_epoch_.compare_exchange_strong(old_safe_and_epoch, pack(true, cycle),
                                                          std::memory_order_acq_rel, std::memory_order_relaxed))
        {
          cell.value_.compare_exchange_strong(tok, NIL, std::memory_order_acq_rel, std::memory_order_relaxed);
          goto checkOverflow;
        }

        if (cell.value_.compare_exchange_strong(tok, reinterpret_cast<std::uintptr_t>(item), std::memory_order_acq_rel,
                                                std::memory_order_relaxed))
        {
          return true;
        }
      }

    checkOverflow:
      const std::uint64_t h = head_.load(std::memory_order_acquire);
      if (t >= h && t - h >= NUM_CELLS)
      {
        closed_.store(true, std::memory_order_release);
        return false;
      }
    }
  }

  /**
   * @brief Attempt to dequeue a pointer from the ring.
   *
   * @return std::optional containing the dequeued pointer, or std::nullopt if empty.
   */
  std::optional<T *> dequeue() noexcept
  {
    for (;;)
    {
      const std::uint64_t h = head_.fetch_add(1, std::memory_order_relaxed);

      const std::uint64_t cycle = h >> PowerTwo;
      const std::uint64_t index = h & (NUM_CELLS - 1);
      auto &cell = cells_[index];
      for (;;)
      {
        const std::uint64_t sne1 = cell.safe_and_epoch_.load(std::memory_order_acquire);
        const std::uintptr_t value = cell.value_.load(std::memory_order_acquire);
        const std::uint64_t sne2 = cell.safe_and_epoch_.load(std::memory_order_acquire);

        if (sne1 != sne2)
          continue;

        bool safe = extract_safe(sne1);
        std::uint64_t epoch = extract_epoch(sne1);

        if (epoch == cycle && is_value(value))
        {
          cell.value_.store(NIL, std::memory_order_release);
          return reinterpret_cast<T *>(value);
        }

        if (epoch <= cycle && (is_nil(value) || is_token(value)))
        {
          if (is_token(value))
          {
            std::uintptr_t old_value = value;
            if (!cell.value_.compare_exchange_strong(old_value, NIL, std::memory_order_acq_rel,
                                                     std::memory_order_relaxed))
            {
              continue;
            }
          }

          std::uint64_t old_safe_and_epoch = sne1;
          if (!cell.safe_and_epoch_.compare_exchange_strong(old_safe_and_epoch, pack(safe, cycle),
                                                            std::memory_order_acq_rel, std::memory_order_relaxed))
          {
            continue;
          }

          break;
        }

        if (epoch < cycle && is_value(value))
        {
          std::uint64_t old_safe_and_epoch = sne1;
          if (!cell.safe_and_epoch_.compare_exchange_strong(old_safe_and_epoch, pack(false, epoch),
                                                            std::memory_order_acq_rel, std::memory_order_relaxed))
          {
            continue;
          }

          break;
        }
        // deq is overtaken
        break;
      }
      const std::uint64_t t = tail_.load(std::memory_order_acquire);
      if (t <= h + 1)
        return std::nullopt;
    }
  }

  /**
   * @brief Reset the queue to an empty state for reuse.
   */
  void reset() noexcept
  {
    constexpr std::uint64_t sne = 1ULL << 63;
    for (auto &cell : cells_)
    {
      cell.safe_and_epoch_.store(sne, std::memory_order_relaxed);
      cell.value_.store(NIL, std::memory_order_relaxed);
    }

    next_.store(nullptr, std::memory_order_relaxed);
    next_available_.store(nullptr, std::memory_order_relaxed);
    head_.store(NUM_CELLS, std::memory_order_relaxed);
    tail_.store(NUM_CELLS, std::memory_order_relaxed);
    closed_.store(false, std::memory_order_relaxed);
  }

  /// Next segment in the linked list of segments owned by a domain.
  alignas(hardware_destructive_interference_size) std::atomic<CRQ *> next_;
  /// Lazily published pointer used during segment recycling.
  alignas(hardware_destructive_interference_size) std::atomic<CRQ *> next_available_;

protected:
  std::array<Cell, NUM_CELLS> cells_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> head_;
  alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> tail_;
  alignas(hardware_destructive_interference_size) std::atomic<bool> closed_;
};

} // namespace conflux::ebr::detail

#endif
