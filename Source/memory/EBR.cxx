#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gsl-lite/gsl-lite.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Common.hxx"
#include "Conflux/Memory.hxx"
#include "EBR.hxx"

namespace conflux::ebr
{

namespace gsl = ::gsl_lite;
namespace
{
detail::ThreadLocalState &get_thread_state() noexcept
{
  static thread_local detail::ThreadLocalState tls;
  return tls;
}

// Prevent recursive try_reclaim() invocation (e.g., when recycling segments during a reclaim pass).
bool &in_try_reclaim_flag() noexcept
{
  thread_local bool in_try_reclaim = false;
  return in_try_reclaim;
}

// Recycle domain ids to keep per-thread caches from growing unbounded.
class IdAllocator
{
public:
  std::uint64_t allocate() noexcept
  {
    lock_.lock();
    const auto unlock = gsl::finally([this] { lock_.unlock(); });

    if (!free_ids_.empty())
    {
      const auto id = free_ids_.back();
      free_ids_.pop_back();
      return id;
    }

    return next_id_++;
  }

  void deallocate(std::uint64_t id) noexcept
  {
    lock_.lock();
    const auto unlock = gsl::finally([this] { lock_.unlock(); });
    try
    {
      free_ids_.push_back(id);
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
      // OOM while recycling an id is non-fatal; leak the id.
    }
  }

private:
  ::conflux::SpinLock lock_;
  std::vector<std::uint64_t> free_ids_;
  std::uint64_t next_id_{0};
};

IdAllocator &domain_id_allocator() noexcept
{
  static IdAllocator allocator{};
  return allocator;
}

// Helper to pop an element from a lock-free singly-linked list stored in an atomic pointer.
template <typename T, typename NextFn>
[[nodiscard]] std::optional<T *> acquire(std::atomic<T *> &list_head, NextFn next) noexcept
{
  for (;;)
  {
    T *old_head = list_head.load(std::memory_order_acquire);
    if (!old_head)
      break;

    T *next_ptr = next(old_head);
    if (list_head.compare_exchange_weak(old_head, next_ptr, std::memory_order_acq_rel, std::memory_order_relaxed))
      return old_head;

    cpu_relax();
  }

  return std::nullopt;
}

template <typename T, typename SNextFn>
void release(std::atomic<T *> &list_head, T *new_head, SNextFn set_next) noexcept
{
  for (;;)
  {
    T *old_head = list_head.load(std::memory_order_acquire);
    set_next(new_head, old_head);
    if (list_head.compare_exchange_weak(old_head, new_head, std::memory_order_acq_rel, std::memory_order_relaxed))
      break;
    cpu_relax();
  }
}

} // namespace

namespace detail
{

void DomainState::release_ref() noexcept
{
  if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
  {
    const auto recycled_id = id_.load(std::memory_order_relaxed);
    delete this; // NOLINT(cppcoreguidelines-owning-memory)
    domain_id_allocator().deallocate(recycled_id);
  }
}

DomainState::ThreadRecord::ThreadRecord()
    : retired_lists_{}, retired_sizes_{}, epoch_{IDLE_EPOCH}, next_available_{nullptr}, next_{nullptr}, active_{false}
{
  retired_sizes_.fill(0);
  for (std::size_t bin_idx = 0; bin_idx < MAX_BINS; ++bin_idx)
  {
    retired_lists_[bin_idx].fill(nullptr);
  }
}

DomainState::DomainState()
    : id_{domain_id_allocator().allocate()}, ref_count_{0}, epoch_{0}, retired_count_{0}, deleted_count_{0},
      active_thread_records_{nullptr}, inactive_thread_records_{nullptr}, recycled_segments_{nullptr}, retired_lists_{},
      active_threads_{0}
{
  auto first_rec = std::make_unique<ThreadRecord>();
  ThreadRecord *first_rec_raw = first_rec.release();

  release(active_thread_records_, first_rec_raw,
          [](ThreadRecord *new_rec, ThreadRecord *old_rec) { new_rec->next_ = old_rec; });

  release(inactive_thread_records_, first_rec_raw,
          [](ThreadRecord *new_rec, ThreadRecord *old_rec) { new_rec->next_available_ = old_rec; });

  for (auto &bag : retired_lists_)
  {
    gsl::owner<Segment *> segment = acquire_segment();
    bag.head_seg_.store(segment, std::memory_order_relaxed);
    bag.tail_seg_.store(segment, std::memory_order_relaxed);
  }
}

void DomainState::enter(ThreadRecord *rec)
{
  auto current_epoch = epoch_.load(std::memory_order_acquire);

  rec->epoch_.store(current_epoch, std::memory_order_release);
}

DomainState::ThreadRecord *DomainState::acquire_record()
{
  auto opt = acquire(inactive_thread_records_,
                     [](ThreadRecord *old_rec) { return old_rec->next_available_.load(std::memory_order_relaxed); });
  if (opt.has_value())
  {
    ThreadRecord *reused = opt.value();
    reused->active_.store(true, std::memory_order_release);
    reused->epoch_.store(IDLE_EPOCH, std::memory_order_release);
    active_threads_.fetch_add(1, std::memory_order_relaxed);
    return reused;
  }

  auto rec_owner = std::make_unique<ThreadRecord>();
  ThreadRecord *rec = rec_owner.release();
  rec->active_.store(true, std::memory_order_release);
  active_threads_.fetch_add(1, std::memory_order_relaxed);
  release(active_thread_records_, rec, [](ThreadRecord *new_rec, ThreadRecord *old_rec) { new_rec->next_ = old_rec; });

  return rec;
}

void DomainState::retire(Retireable *object, void (*deleter)(Retireable *))
{
  retire_without_reclaim(object, deleter);

  const auto total_retired = retired_count_.load(std::memory_order_relaxed);
  const auto total_deleted = deleted_count_.load(std::memory_order_relaxed);
  if (total_retired - total_deleted >= TLS_RETIRED_THRESHOLD)
  {
    try_reclaim();
  }
}

void DomainState::retire_without_reclaim(Retireable *object, void (*deleter)(Retireable *))
{
  auto &tls = get_thread_state();
  auto *rec = tls.get_cached_rec(id());
  assert(rec && "enter() must be called before retire()");

  auto my_epoch = rec->epoch_.load(std::memory_order_relaxed);
  auto bin_idx = static_cast<std::size_t>(my_epoch % MAX_BINS);

  auto &retired_size = rec->retired_sizes_[bin_idx];
  auto &retired_list = rec->retired_lists_[bin_idx];

#if defined(CONFLUX_EBR_DEBUG)
  bool expected = false;
  if (!object->retired_flag.compare_exchange_strong(expected, true, std::memory_order_relaxed))
  {
    assert(!"Double retire detected");
  }
#endif

  if (retired_size >= TLS_RETIRED_THRESHOLD)
  {

    auto current_epoch = epoch_.load(std::memory_order_acquire);

    std::size_t reclaim_idx = 0;
    for (; reclaim_idx < retired_size; ++reclaim_idx)
    {
      if (!((retired_list[reclaim_idx]->epoch + 1) < current_epoch))
      {
        break;
      }
      retired_list[reclaim_idx]->deleter(retired_list[reclaim_idx]);
    }

    if (reclaim_idx > 0)
    {
      deleted_count_.fetch_add(reclaim_idx, std::memory_order_relaxed);
      retired_size -= reclaim_idx;
      std::memmove(static_cast<void *>(retired_list.data()),
                   static_cast<const void *>(retired_list.data() + reclaim_idx), retired_size * sizeof(Retireable *));
    }
    else
    {
      for (std::size_t defer_idx = 0; defer_idx < retired_size; ++defer_idx)
      {
        enqueue(bin_idx, retired_list[defer_idx]);
      }
      retired_size = 0;
    }
  }

  object->deleter = deleter;
  object->epoch = my_epoch;
  retired_list[retired_size] = object;
  ++retired_size;

  retired_count_.fetch_add(1, std::memory_order_relaxed);
}

void DomainState::leave()
{
  auto &tls = get_thread_state();
  auto *rec = tls.get_cached_rec_no_allocate(id());
  assert(rec && "enter() must be called before calling leave()");
  rec->epoch_.store(IDLE_EPOCH, std::memory_order_release);
}

void DomainState::try_reclaim()
{
  auto &in_try_reclaim = in_try_reclaim_flag();
  if (in_try_reclaim)
  {
    return;
  }
  const auto guard = gsl::finally([&in_try_reclaim] { in_try_reclaim = false; });
  in_try_reclaim = true;

  auto current_epoch = epoch_.load(std::memory_order_acquire);

  // Attempt to advance the epoch if no thread is lagging behind.
  const ThreadRecord *rec = active_thread_records_.load(std::memory_order_acquire);
  bool can_advance = true;
  while (rec != nullptr)
  {
    const auto rec_epoch = rec->epoch_.load(std::memory_order_acquire);
    if (rec_epoch != IDLE_EPOCH && rec_epoch < current_epoch)
    {
      can_advance = false;
      break;
    }
    rec = rec->next_.load(std::memory_order_relaxed);
  }

  if (can_advance)
  {
    std::uint64_t expected = current_epoch;
    (void)epoch_.compare_exchange_strong(expected, current_epoch + 1, std::memory_order_acq_rel,
                                         std::memory_order_relaxed);
    current_epoch = epoch_.load(std::memory_order_acquire);
  }

  const std::size_t reclaim_idx = (current_epoch + 1) % MAX_BINS;
  for (;;)
  {
    auto maybe_retired = dequeue(
        reclaim_idx,
        [](Segment *seg, void *ctx) {
          auto *domain = static_cast<DomainState *>(ctx);
          domain->retire_without_reclaim(seg, &DomainState::recycle_segment);
        },
        this);
    if (!maybe_retired.has_value())
    {
      break;
    }

    Retireable *retired = maybe_retired.value();
    if ((retired->epoch + 1) < current_epoch)
    {
      retired->deleter(retired);
      deleted_count_.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
      // FIFO queue: the first too-young element means the rest are also too young.
      retired->epoch = current_epoch;
      enqueue(current_epoch % MAX_BINS, retired);
      break;
    }
  }
}

void DomainState::release_record(ThreadRecord *rec) noexcept
{
  rec->epoch_.store(IDLE_EPOCH, std::memory_order_release);
  rec->active_.store(false, std::memory_order_release);
  active_threads_.fetch_sub(1, std::memory_order_relaxed);

  release(inactive_thread_records_, rec,
          [](ThreadRecord *new_rec, ThreadRecord *old_rec) { new_rec->next_available_ = old_rec; });
}

void DomainState::enqueue(std::size_t bin_idx, Retireable *item)
{
  LimboBag &bag = retired_lists_[bin_idx];
  for (;;)
  {
    auto *tail_seg = bag.tail_seg_.load(std::memory_order_acquire);
    if (tail_seg->enqueue(item))
      return;

    auto *new_seg = acquire_segment();

    const bool success = new_seg->enqueue(item);
    assert(success && "private enqueue() must always succeed");

    QueueBase *next = nullptr;
    if (tail_seg->next_.compare_exchange_strong(next, new_seg, std::memory_order_acq_rel, std::memory_order_acquire))
    {
      (void)bag.tail_seg_.compare_exchange_strong(tail_seg, new_seg, std::memory_order_acq_rel,
                                                  std::memory_order_relaxed);
      return;
    }

    // Safe because segments in the ring buffer are always DomainState::Segment instances.
    auto *const next_seg = static_cast<Segment *>(next); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    (void)bag.tail_seg_.compare_exchange_strong(tail_seg, next_seg, std::memory_order_acq_rel,
                                                std::memory_order_relaxed);
    release_segment(new_seg);
  }
}

DomainState::Segment *DomainState::acquire_segment()
{
  auto opt = acquire(recycled_segments_, [](Segment *seg) {
    return static_cast<Segment *>(seg->next_available_.load(std::memory_order_relaxed)); // NOLINT
  });
  if (opt.has_value())
  {
    Segment *segment = opt.value();
    segment->owner_ = this;
    segment->reset();
#if defined(CONFLUX_EBR_DEBUG)
    segment->retired_flag.store(false, std::memory_order_relaxed);
#endif
    return segment;
  }

  auto segment_owner = std::make_unique<Segment>(this);
  Segment *segment_raw = segment_owner.release();
  segment_raw->reset();
#if defined(CONFLUX_EBR_DEBUG)
  segment_raw->retired_flag.store(false, std::memory_order_relaxed);
#endif

  return segment_raw;
}

void DomainState::release_segment(DomainState::Segment *s) noexcept
{
  s->owner_ = this;
  s->reset();
  release(recycled_segments_, s, [](Segment *new_seg, Segment *old_seg) { new_seg->next_available_ = old_seg; });
}

void DomainState::recycle_segment(Retireable *retired) noexcept
{
  auto *segment = static_cast<Segment *>(retired); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
  if (segment->owner_ != nullptr)
  {
    segment->owner_->release_segment(segment);
    return;
  }

  // Fallback: owner unknown, delete permanently.
  delete segment; // NOLINT(cppcoreguidelines-owning-memory)
}

std::optional<Retireable *> DomainState::dequeue(std::size_t bin_idx, OnSwing on_swing, void *ctx)
{
  LimboBag &bag = retired_lists_[bin_idx];

  for (;;)
  {

    auto *head_seg = bag.head_seg_.load(std::memory_order_acquire);

    auto res = head_seg->dequeue();
    if (res.has_value())
      return res;

    QueueBase *next_base = head_seg->next_.load(std::memory_order_acquire);
    if (next_base == nullptr)
      return std::nullopt;

    res = head_seg->dequeue();
    if (res.has_value())
      return res;

    auto *next = static_cast<Segment *>(next_base); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    if (bag.head_seg_.compare_exchange_strong(head_seg, next, std::memory_order_acq_rel, std::memory_order_relaxed))
    {
      on_swing(head_seg, ctx);
    }
  }
}

DomainState::~DomainState() noexcept
{
  // Drain all limbo bags without calling back into retire/try_reclaim paths.
  for (auto &bag : retired_lists_)
  {
    auto delete_on_swing = [](Segment *seg, void *) { delete seg; }; // NOLINT(cppcoreguidelines-owning-memory)
    for (;;)
    {
      auto retired = dequeue(static_cast<std::size_t>(&bag - retired_lists_.data()), delete_on_swing, nullptr);
      if (!retired.has_value())
      {
        break;
      }
      retired.value()->deleter(retired.value());
    }

    gsl::owner<Segment *> seg = bag.head_seg_.load(std::memory_order_relaxed);
    while (seg != nullptr)
    {
      gsl::owner<Segment *> next = static_cast<Segment *>(seg->next_.load(std::memory_order_relaxed)); // NOLINT
      delete seg; // NOLINT(cppcoreguidelines-owning-memory)
      seg = next;
    }
  }

  gsl::owner<Segment *> recycled = recycled_segments_.load(std::memory_order_acquire);
  while (recycled != nullptr)
  {
    gsl::owner<Segment *> next =
        static_cast<Segment *>(recycled->next_available_.load(std::memory_order_relaxed)); // NOLINT
    delete recycled; // NOLINT(cppcoreguidelines-owning-memory)
    recycled = next;
  }

  gsl::owner<ThreadRecord *> head = active_thread_records_.load(std::memory_order_acquire);
  while (head != nullptr)
  {
    gsl::owner<ThreadRecord *> next = head->next_.load(std::memory_order_relaxed);
    delete head; // NOLINT(cppcoreguidelines-owning-memory)

    head = next;
  }
}

DomainState::ThreadRecord *ThreadLocalState::get_cached_rec(std::uint64_t id)
{
  if (domains_.size() <= id)
  {
    domains_.resize(id + 1);
  }

  return domains_[id].rec_;
}

DomainState::ThreadRecord *ThreadLocalState::get_cached_rec_no_allocate(std::uint64_t id) noexcept
{
  if (domains_.size() <= id)
  {
    return nullptr;
  }

  return domains_[id].rec_;
}

void ThreadLocalState::set_cached_state(std::uint64_t id, ::conflux::ebr::Domain state, DomainState::ThreadRecord *rec) noexcept
{
  assert(domains_.size() > id && "this thread has never been a part of domain.");

  domains_[id].state_ = std::move(state);
  domains_[id].rec_ = rec;
}

void ThreadLocalState::clear_cached_state(std::uint64_t id) noexcept
{
  if (domains_.size() <= id)
  {
    return;
  }

  auto &entry = domains_[id];
  if (entry.state_ && entry.rec_ != nullptr)
  {
    auto state = std::move(entry.state_);
    auto *rec = entry.rec_;
    entry.rec_ = nullptr;

    try
    {
      for (std::size_t bin_idx = 0; bin_idx < MAX_BINS; ++bin_idx)
      {
        auto &retired_size = rec->retired_sizes_[bin_idx];
        auto &retired_list = rec->retired_lists_[bin_idx];
        for (std::size_t j = 0; j < retired_size; ++j)
        {
          state.state_->enqueue(bin_idx, retired_list[j]);
        }
        retired_size = 0;
      }
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
      // Swallow to preserve noexcept destructor contract.
      (void)state;
    }

    state.state_->release_record(rec);
  }
}

ThreadLocalState::~ThreadLocalState()
{
  for (auto &entry : domains_)
  {
    if (!entry.state_ || entry.rec_ == nullptr)
    {
      continue;
    }

    auto state = std::move(entry.state_);
    auto *rec = entry.rec_;
    entry.rec_ = nullptr;

    try
    {
      for (std::size_t bin_idx = 0; bin_idx < MAX_BINS; ++bin_idx)
      {
        auto &retired_size = rec->retired_sizes_[bin_idx];
        auto &retired_list = rec->retired_lists_[bin_idx];
        for (std::size_t j = 0; j < retired_size; ++j)
        {
          state.state_->enqueue(bin_idx, retired_list[j]);
        }
        retired_size = 0;
      }
    }
    catch (...) // NOLINT(bugprone-empty-catch)
    {
      // Swallow to preserve noexcept destructor contract.
      (void)state;
    }
    state.state_->release_record(rec);
  }
}

} // namespace detail

Domain::Domain() = default;

Domain::Domain(const Domain &other) noexcept
{
  reset(other.state_);
}

Domain::Domain(Domain &&other) noexcept : state_{other.state_}
{
  other.state_ = nullptr;
}

Domain &Domain::operator=(const Domain &other) noexcept
{
  if (this != &other)
  {
    if (state_ != other.state_)
    {
      if (state_ != nullptr)
      {
        get_thread_state().clear_cached_state(state_->id());
      }
      reset(other.state_);
    }
  }
  return *this;
}

Domain &Domain::operator=(Domain &&other) noexcept
{
  if (this != &other)
  {
    if (state_ != nullptr)
    {
      get_thread_state().clear_cached_state(state_->id());
    }
    reset(other.state_);
    other.state_ = nullptr;
  }
  return *this;
}

Domain::~Domain() noexcept
{
  if (state_ != nullptr)
  {
    get_thread_state().clear_cached_state(state_->id());
    state_->release_ref();
  }
}

Domain::Domain(detail::DomainState *state) noexcept
{
  reset(state);
}

void Domain::reset(detail::DomainState *state) noexcept
{
  if (state_ != nullptr)
  {
    state_->release_ref();
  }

  state_ = state;

  if (state_ != nullptr)
  {
    state_->add_ref();
  }
}

void Domain::enter()
{
  assert(state_ && "Domain is empty");
  auto &tls = get_thread_state();
  auto *rec = tls.get_cached_rec(state_->id());
  if (!rec)
  {
    rec = state_->acquire_record();
    tls.set_cached_state(state_->id(), *this, rec);
  }

  state_->enter(rec);
}

void Domain::retire(Retireable *object, void (*deleter)(Retireable *))
{
  assert(state_ && "Domain is empty");
  state_->retire(object, deleter);
}

void Domain::leave()
{
  assert(state_ && "Domain is empty");
  state_->leave();
}

void Domain::try_reclaim()
{
  assert(state_ && "Domain is empty");
  state_->try_reclaim();
}

std::uint64_t Domain::id() const
{
  assert(state_ && "Domain is empty");
  return state_->id();
}

Stats Domain::stats() const
{
  assert(state_ && "Domain is empty");

  Stats snapshot{};
  snapshot.epoch = state_->epoch_.load(std::memory_order_relaxed);
  snapshot.retired = state_->retired_count_.load(std::memory_order_relaxed);
  snapshot.deleted = state_->deleted_count_.load(std::memory_order_relaxed);
  snapshot.active_threads = state_->active_threads_.load(std::memory_order_relaxed);
  return snapshot;
}

Domain make_domain()
{
  return Domain{new detail::DomainState()};
}

ObjectGuard::ObjectGuard(Domain &domain) : domain_{&domain}
{
  domain_->enter();
}

ObjectGuard::ObjectGuard(ObjectGuard &&other) noexcept : domain_{other.domain_}
{
  other.domain_ = nullptr;
}

ObjectGuard &ObjectGuard::operator=(ObjectGuard &&other) noexcept
{
  if (this != &other)
  {
    if (domain_ != nullptr)
    {
      domain_->leave();
    }
    domain_ = other.domain_;
    other.domain_ = nullptr;
  }
  return *this;
}

ObjectGuard::~ObjectGuard()
{
  if (domain_ != nullptr)
  {
    domain_->leave();
  }
}

} // namespace conflux::ebr
