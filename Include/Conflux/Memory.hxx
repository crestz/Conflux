#ifndef CONFLUX_MEMORY_H
#define CONFLUX_MEMORY_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

// Export macro (mirrors Source/Common.hxx for public headers)
#if !defined(CONFLUX_API)
#  if defined(_WIN32) || defined(__CYGWIN__)
#    ifdef CONFLUX_EXPORT_SYMBOLS
#      define CONFLUX_API __declspec(dllexport)
#    else
#      define CONFLUX_API __declspec(dllimport)
#    endif
#  else
#    if __GNUC__ >= 4
#      define CONFLUX_API __attribute__((visibility("default")))
#    else
#      define CONFLUX_API
#    endif
#  endif
#endif

namespace conflux::ebr
{

struct CONFLUX_API Retireable
{
  void (*deleter)(Retireable *);
  std::uint64_t epoch;
#if defined(CONFLUX_EBR_DEBUG)
  std::atomic<bool> retired_flag{false};
#endif
};

struct CONFLUX_API Stats
{
  std::uint64_t epoch{};
  std::uint64_t retired{};
  std::uint64_t deleted{};
  std::uint64_t active_threads{};
};

namespace detail
{
class DomainState;
class ThreadLocalState;
} // namespace detail

class CONFLUX_API Domain
{
public:
  Domain();
  Domain(const Domain &other) noexcept;
  Domain(Domain &&other) noexcept;
  Domain &operator=(const Domain &other) noexcept;
  Domain &operator=(Domain &&other) noexcept;
  ~Domain() noexcept;

  void enter();
  void retire(Retireable *object, void (*)(Retireable *));
  void leave();
  void try_reclaim();

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }
  [[nodiscard]] Stats stats() const;

private:
  explicit Domain(detail::DomainState *state) noexcept;
  void reset(detail::DomainState *state) noexcept;

  detail::DomainState *state_{nullptr};

  friend class ObjectGuard;
  friend detail::ThreadLocalState;
  friend Domain make_domain();
};

[[nodiscard]] CONFLUX_API Domain make_domain();

/**
 * @brief RAII helper that keeps the calling thread registered with a domain epoch.
 *
 * Constructing an ObjectGuard enters the domain for the current thread and leaving scope calls Domain::leave().
 * Marked [[nodiscard]] to avoid accidentally creating and discarding a temporary without protecting any access.
 */
class [[nodiscard]] CONFLUX_API ObjectGuard
{
public:
  explicit ObjectGuard(Domain &domain);
  ObjectGuard(const ObjectGuard &) = delete;
  ObjectGuard &operator=(const ObjectGuard &) = delete;
  ObjectGuard(ObjectGuard &&other) noexcept;
  ObjectGuard &operator=(ObjectGuard &&other) noexcept;
  ~ObjectGuard();

  [[nodiscard]] bool engaged() const noexcept { return domain_ != nullptr; }
  [[nodiscard]] Domain *domain() const noexcept { return domain_; }

private:
  Domain *domain_{nullptr};
};

inline void retire(Domain &domain, Retireable *object, void (*deleter)(Retireable *))
{
  domain.retire(object, deleter);
}

template <typename Derived>
class Object : public Retireable
{
public:
  Object() noexcept : Retireable{&Object::retire_trampoline, 0} {}
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  ~Object() = default;

  void retire(Domain &domain)
  {
    domain.retire(static_cast<Retireable *>(static_cast<Derived *>(this)), &Object::retire_trampoline);
  }

protected:
  // Derived must implement: void reclaim() noexcept;

private:
  static void retire_trampoline(Retireable *retired) noexcept
  {
    auto *self = static_cast<Derived *>(retired); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    self->reclaim();
  }
};

template <class T, class Deleter = std::default_delete<T>,
          typename = std::enable_if_t<!std::is_base_of_v<Retireable, T>>>
void retire(Domain &domain, T *ptr, Deleter d = {})
{
  if (!ptr)
    return;

  struct Wrapper : Retireable
  {
    T *obj{};
    Deleter del;

    static void do_delete(Retireable *r) noexcept
    {
      auto *self = static_cast<Wrapper *>(r);
      self->del(self->obj);
      delete self; // NOLINT(cppcoreguidelines-owning-memory)
    }
  };

  auto *wrapper = new Wrapper; // NOLINT(cppcoreguidelines-owning-memory)
  wrapper->obj = ptr;
  wrapper->del = std::move(d);
  wrapper->deleter = &Wrapper::do_delete;
  wrapper->epoch = 0;

  domain.retire(wrapper, wrapper->deleter);
}

inline void try_reclaim(Domain &domain)
{
  domain.try_reclaim();
}

inline Stats stats(Domain &domain)
{
  return domain.stats();
}

} // namespace conflux::ebr

#endif // CONFLUX_MEMORY_H
