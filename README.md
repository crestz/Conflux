Conflux Epoch-Based Reclamation (EBR)
=====================================

- Concept: threads join a `Domain` (via `ObjectGuard`) while they access shared objects. Objects are retired when removed from data structures, and are reclaimed only after all participating threads have advanced past the retire epoch. The implementation stripes retirements across bins to reduce contention and batches reclamation.

- Usage pattern:
  - Create a domain: `auto domain = conflux::ebr::make_domain();`
  - Enter while touching shared state: `conflux::ebr::ObjectGuard guard(domain);`
  - Retire objects you remove: `node->retire(domain);` (for `ebr::Object`), or `conflux::ebr::retire(domain, ptr, deleter);`
  - Occasionally help GC: `conflux::ebr::try_reclaim(domain);`

- Lifetime rules:
  - `Domain` is reference counted; copying handles keeps the domain alive across threads.
  - A thread must hold an `ObjectGuard` while dereferencing shared objects that could be reclaimed. The guard is `[[nodiscard]]` to discourage accidental temporaries.
  - Objects should only be retired once. Defining `CONFLUX_EBR_DEBUG` enables a double-retire assert.

- Introspection:
  - `ebr::stats(domain)` returns `{epoch, retired, deleted, active_threads}` for quick health checks and logging.
  - Example:

    ```c++
    auto s = conflux::ebr::stats(domain);
    std::cout << "epoch=" << s.epoch
              << " retired=" << s.retired
              << " deleted=" << s.deleted
              << " active_threads=" << s.active_threads
              << '\n';
    ```

- Tuning & behavior notes:
  - `TLS_RETIRED_THRESHOLD` governs when per-thread retired objects spill to the shared queues (currently 64).
  - `MAX_BINS` stripes retirements across per-epoch bins (currently 3) to reduce hot spots.
  - `try_reclaim(domain)` is safe to call from any thread; aggressive reclaimers reduce peak memory use.
  - Recycled internal segments are reused to limit allocations; the domain cleans up all resources on final destruction.
