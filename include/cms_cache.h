// Count-Min Sketch for dynamic cache frequency estimation.
// Lock-free add (atomic fetch_add on uint16_t cells), thread-safe estimate.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace diskann
{

class CountMinSketch
{
  public:
    static constexpr uint32_t ROWS = 4;
    static constexpr uint32_t COLS = 4096;

    CountMinSketch() { memset(_table, 0, sizeof(_table)); }

    void reset() { memset(_table, 0, sizeof(_table)); }

    inline void add(uint32_t id)
    {
        uint32_t h = id;
        for (uint32_t r = 0; r < ROWS; r++)
        {
            h = _mix(h + r * 0x9e3779b9u);
            uint32_t col = h & (COLS - 1);
            auto &cell = reinterpret_cast<std::atomic<uint16_t> &>(_table[r][col]);
            uint16_t v = cell.load(std::memory_order_relaxed);
            if (v < 0xffffu)
                cell.fetch_add(1, std::memory_order_relaxed);
        }
    }

    inline uint32_t estimate(uint32_t id) const
    {
        uint32_t h = id;
        uint32_t mn = 0xffffffffu;
        for (uint32_t r = 0; r < ROWS; r++)
        {
            h = _mix(h + r * 0x9e3779b9u);
            uint32_t col = h & (COLS - 1);
            uint32_t v = reinterpret_cast<const std::atomic<uint16_t> &>(_table[r][col])
                             .load(std::memory_order_relaxed);
            mn = std::min(mn, v);
        }
        return mn;
    }

  private:
    alignas(64) uint16_t _table[ROWS][COLS];

    static inline uint32_t _mix(uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x45d9f3bu;
        x ^= x >> 16;
        return x;
    }
};

} // namespace diskann
