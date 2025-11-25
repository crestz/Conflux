#ifndef CONFLUX_INTRUSIVE_LIST_HXX
#define CONFLUX_INTRUSIVE_LIST_HXX

#include "Common.hxx"
#include <atomic>

namespace conflux
{

template <typename T, T *T::*Next> class IntrusiveList
{
public:
  void push(T *node) noexcept
  {

    for (;;)
    {
      T *old_head = head_.load(std::memory_order_acquire);
      T *&next = node->*Next;
      next = old_head;

      if (head_.compare_exchange_weak(old_head, node, std::memory_order_acq_rel, std::memory_order_relaxed))
        break;
      cpu_relax();
    }
  }

  T *pop() noexcept
  {
    T *old_head = head_.load(std::memory_order_acquire);
    while (old_head != nullptr)
    {
      T *next = old_head->*Next;
      if (head_.compare_exchange_weak(old_head, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return old_head;
      cpu_relax();
    }

    return nullptr;
  }

  T *get_head() noexcept
  {
    return head_.load(std::memory_order_acquire);
  }

private:
  alignas(hardware_destructive_interference_size) std::atomic<T *> head_{nullptr};
};

} // namespace conflux

#endif // CONFLUX_INTRUVIDE_LIST_HXX