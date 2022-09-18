#include "common.hpp"

#define N_OF_ITEMS 4194304

/***
 * THIS EXAMPLE IS FOR CPU THAT HAS 64 BYTES SIZED CACHE LINES
 * 
 * The following test is only relevant for parallelized processing, when multiple threads tries to get ownership of the same cache line, resulting in "false sharing".
 * To avoid this, keep data that are not used together apart from memory, specifically in this case, keep the data on separate cache lines. Add padding, if necessary.
 * This wastes space, but guarantees that the 2 data pieces are on separate cache lines.
 */

struct cache_aligned_data
{
    u32 x;        // 4 bytes
    char pad[60]; // pad remaining 60 bytes
};

struct cache_unaligned_data
{
    u32 x;
};

void AlignedWorker(cache_aligned_data *Data)
{
    for (u32 i = 0; i < N_OF_ITEMS; ++i)
    {
        Data->x += 1;
    }
}

void UnalignedWorker(cache_unaligned_data *Data)
{
    for (u32 i = 0; i < N_OF_ITEMS; ++i)
    {
        Data->x += 1;
    }
}

i32 main()
{
    LOG(sizeof(cache_aligned_data));
    LOG(sizeof(cache_unaligned_data));
    cache_aligned_data a;
    cache_aligned_data b;
    u64 CyclesStart = __rdtsc();
    thread t1(&AlignedWorker, &a);
    thread t2(&AlignedWorker, &b);
    t1.join();
    t2.join();
    u64 CyclesEnd = __rdtsc();
    LOG(setw(20 + strlen("Cycles taken")) << "Cycles taken");
    LOG(setw(20) << "Aligned worker: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");

    cache_unaligned_data c;
    cache_unaligned_data d;
    CyclesStart = __rdtsc();
    t1 = thread(&UnalignedWorker, &c);
    t2 = thread(&UnalignedWorker, &d);
    t1.join();
    t2.join();
    CyclesEnd = __rdtsc();
    LOG(setw(20) << "Unaligned worker: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");
}
