#ifndef COMMON_HPP
# define COMMON_HPP


# include <iostream>
using namespace std;
# define LOG(x) (cout << x << endl)
# define LINE() (LOG(__LINE__))

# include <cstdint>
typedef int8_t      i8;
typedef int16_t     i16;
typedef int32_t     i32;
typedef int64_t     i64;
typedef i32         b32;
typedef uint8_t     u8;
typedef uint16_t    u16;
typedef uint32_t    u32;
typedef uint64_t    u64;
typedef float       r32;
typedef double      r64;

# include <intrin.h> // for __rdtsc() intrinsic
# include <random>
# include <vector>
# include <algorithm>
# include <thread>
# include <iomanip>
# include <cstring>
# include <stdlib.h> // for aligned_alloc

random_device dev;
mt19937 rng(dev());
b32 IsRandomDeviceInitialized;

inline r32
GetRand(r32 low, r32 high)
{
    uniform_real_distribution<r32> dist(low, high);
    if (IsRandomDeviceInitialized == false)
    {
        IsRandomDeviceInitialized = true;
        rng.seed(42);
    }
    return (dist(rng));
}

inline i16
GetRand(i16 low, i16 high)
{
    uniform_int_distribution<i16> dist(low, high);
    if (IsRandomDeviceInitialized == false)
    {
        IsRandomDeviceInitialized = true;
        rng.seed(42);
    }
    return (dist(rng));
}

inline i32
GetRand(i32 low, i32 high)
{
    uniform_int_distribution<i32> dist(low, high);
    if (IsRandomDeviceInitialized == false)
    {
        IsRandomDeviceInitialized = true;
        rng.seed(42);
    }
    return (dist(rng));
}

inline i64
GetRand(i64 low, i64 high)
{
    uniform_int_distribution<i64> dist(low, high);
    if (IsRandomDeviceInitialized == false)
    {
        IsRandomDeviceInitialized = true;
        rng.seed(42);
    }
    return (dist(rng));
}

#endif
