#include "common.hpp"

#define N_OF_ITEMS 1048576

/***
 * Since memory is the main bottleneck almost always, the branch predictor of the CPU does a lot of prefetching ahead of time.
 * Whenever there is a branch in the code, it introduces branching. The branch predictor guesses based on previous results in which way the instruction flow is going.
 * If the branch being predicted is incorrect, the work done has to be undone, which is expensive.
 * This example shows how having this concept in mind and to use most of the branch predictor:
 * - either avoid branching as much as we can, using other concepts such as existential processing (roughly meaning that process the things which needs to be processed with their transforms in a declarative style)
 * - ensure that branching happens in large chunks of true or false evaluations (in this example I just just had to sort the data for the condition)
 */

i32 main(void)
{
    vector<i32> Data(N_OF_ITEMS, 0);
    for (i32 i = 0; i < N_OF_ITEMS; ++i)
    {
        Data[i] = GetRand(0, 255);
    }

    i32 WriteOut = 0;
    u64 CyclesStart = __rdtsc();
    for (i32 i = 0; i < N_OF_ITEMS; ++i)
    {
        // as data is inconsistent, the branch predictor cannot generate
        // predictable results and needs to unroll for a lot of extra work
        if (Data[i] < 128)
        {
            WriteOut += Data[i];
        }
    }
    u64 CyclesEnd = __rdtsc();
    LOG(setw(20 + strlen("Clock cycles")) << "Clock cycles");
    LOG(setw(20) << "Without sort: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");

    WriteOut = 0;
    u64 SortStart = __rdtsc();
    sort(Data.begin(), Data.end());
    u64 SortEnd = __rdtsc();

    CyclesStart = __rdtsc();
    for (i32 i = 0; i < N_OF_ITEMS; ++i)
    {
        if (Data[i] < 128)
        {
            WriteOut += Data[i];
        }
    }
    CyclesEnd = __rdtsc();
    LOG(setw(20) << "With sort: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");

    LOG("However, sort takes time too so keep that in mind: " << (SortEnd - SortStart) / 1000000.0f << "M");
}