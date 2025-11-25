#ifndef CONFLUX_UTILS_H
#define CONFLUX_UTILS_H

#include <atomic>
#include <optional>
#include <type_traits>
#include <utility>

#include "Conflux/Memory.hxx"
#include "Common.hxx"

namespace conflux
{

template <typename T>
class TreiberStack
{
  struct Node : ebr::Object<Node>
  {
    template <typename U>
    explicit Node(U &&v) noexcept(std::is_nothrow_constructible_v<T, U &&>) : value(std::forward<U>(v)), next(nullptr)
    {
    }

    void reclaim() noexcept
    {
      delete this; // NOLINT(cppcoreguidelines-owning-memory)
    }

    T value;
    std::atomic<Node *> next;
  };

public:
  explicit TreiberStack(ebr::Domain &domain) noexcept : domain_{domain}, head_{nullptr} {}
  TreiberStack(const TreiberStack &) = delete;
  TreiberStack &operator=(const TreiberStack &) = delete;
  TreiberStack(TreiberStack &&) = delete;
  TreiberStack &operator=(TreiberStack &&) = delete;

  ~TreiberStack()
  {
    Node *node = head_.load(std::memory_order_acquire);
    while (node != nullptr)
    {
      Node *next = node->next.load(std::memory_order_relaxed);
      delete node; // NOLINT(cppcoreguidelines-owning-memory)
      node = next;
    }
  }

  template <typename U>
  void push(U &&value)
  {
    auto *node = new Node(std::forward<U>(value)); // NOLINT(cppcoreguidelines-owning-memory)

    ebr::ObjectGuard guard(domain_);
    Node *old_head = head_.load(std::memory_order_acquire);
    do
    {
      node->next.store(old_head, std::memory_order_relaxed);
    } while (!head_.compare_exchange_weak(old_head, node, std::memory_order_acq_rel, std::memory_order_acquire));
  }

  std::optional<T> try_pop()
  {
    ebr::ObjectGuard guard(domain_);
    Node *head = head_.load(std::memory_order_acquire);
    while (head != nullptr)
    {
      Node *next = head->next.load(std::memory_order_relaxed);
      if (head_.compare_exchange_weak(head, next, std::memory_order_acq_rel, std::memory_order_acquire))
      {
        std::optional<T> result{std::move(head->value)};
        head->retire(domain_);
        return result;
      }
    }

    return std::nullopt;
  }

private:
  ebr::Domain &domain_;
  std::atomic<Node *> head_;
};

} // namespace conflux

#endif // CONFLUX_UTILS_H
