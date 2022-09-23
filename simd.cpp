#include "common.hpp"
#include "immintrin.h"

#define N_OF_ITEMS 1048576

#if defined(__AVX512F__)
# define BYTE_BOUNDARY 64
#elif defined(__AVX__)
# define BYTE_BOUNDARY 32
#elif defined(__SSE__)
# define BYTE_BOUNDARY 16
#else
# error "instruction set not supported"
#endif

struct position
{
    r32 x;
    r32 y;
    r32 z;

    b32 operator==(position that) const
    {
        return (x == that.x && y == that.y && z == that.z);
    }
};

struct velocity
{
    r32 dx;
    r32 dy;
    r32 dz;

    inline b32 operator==(velocity that) const
    {
        return (dx == that.dx && dy == that.dy && dz == that.dz);
    }
};

struct acceleration
{
    r32 ddx;
    r32 ddy;
    r32 ddz;

    inline b32 operator==(acceleration that) const
    {
        return (ddx == that.ddx && ddy == that.ddy && ddz == that.ddz);
    }
};

struct pos_soa
{
    r32 *x;
    r32 *y;
    r32 *z;
};

struct vel_soa
{
    r32 *dx;
    r32 *dy;
    r32 *dz;
};

struct acc_soa
{
    r32 *ddx;
    r32 *ddy;
    r32 *ddz;
};

vector<position> Positions;
vector<velocity> Velocities;
vector<acceleration> Accelerations;
pos_soa Positions2;
vel_soa Velocities2;
acc_soa Accelerations2;

void
SingleInstructionSingleData(void)
{
    r32 dt = 1.0f / 60.0f;
    r32 dt_per_two = dt / 2.0f;
    for (u32 i = 0; i < N_OF_ITEMS; ++i)
    {
        velocity PrevVel = Velocities[i];
        r32 DeltaVelX = Accelerations[i].ddx * dt;
        r32 DeltaVelY = Accelerations[i].ddy * dt;
        r32 DeltaVelZ = Accelerations[i].ddz * dt;
        Velocities[i].dx += DeltaVelX;
        Velocities[i].dy += DeltaVelY;
        Velocities[i].dz += DeltaVelZ;
        r32 SummedVelX = Velocities[i].dx + PrevVel.dx;
        r32 SummedVelY = Velocities[i].dy + PrevVel.dy;
        r32 SummedVelZ = Velocities[i].dz + PrevVel.dz;
        r32 DeltaPosX = SummedVelX * dt_per_two;
        r32 DeltaPosY = SummedVelY * dt_per_two;
        r32 DeltaPosZ = SummedVelZ * dt_per_two;
        Positions[i].x += DeltaPosX;
        Positions[i].y += DeltaPosY;
        Positions[i].z += DeltaPosZ;
    }
}

// Intel Architecture Code Analyzer
// https://www.intel.com/content/www/us/en/developer/articles/tool/architecture-code-analyzer.html
#if 0
# include "iacaMarks.h"
#else
# define IACA_START
# define IACA_END
#endif

void
SingleInstructionMultipleData(void)
{

IACA_START;

#if defined (__FMA__)
# if defined(__AVX512F__)
    __m512 dt = _mm512_set1_ps(1.0f / 60.0f);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 dt_per_two = _mm512_div_ps(dt, two);
    u32 Stride = BYTE_BOUNDARY / sizeof(r32);
    u32 Iterations = N_OF_ITEMS - N_OF_ITEMS % Stride;
    for (u32 i = 0; i < Iterations; i += Stride)
    {
        __m512 PosX_16x = _mm512_load_ps(&Positions2.x[i]);
        __m512 PosY_16x = _mm512_load_ps(&Positions2.y[i]);
        __m512 PosZ_16x = _mm512_load_ps(&Positions2.z[i]);
        __m512 VelX_16x = _mm512_load_ps(&Velocities2.dx[i]);
        __m512 VelY_16x = _mm512_load_ps(&Velocities2.dy[i]);
        __m512 VelZ_16x = _mm512_load_ps(&Velocities2.dz[i]);
        __m512 PrevVelX_16x = _mm512_load_ps(&Velocities2.dx[i]);
        __m512 PrevVelY_16x = _mm512_load_ps(&Velocities2.dy[i]);
        __m512 PrevVelZ_16x = _mm512_load_ps(&Velocities2.dz[i]);
        __m512 AccX_16x = _mm512_load_ps(&Accelerations2.ddx[i]);
        __m512 AccY_16x = _mm512_load_ps(&Accelerations2.ddy[i]);
        __m512 AccZ_16x = _mm512_load_ps(&Accelerations2.ddz[i]);

        VelX_16x = _mm512_fmadd_ps(AccX_16x, dt, VelX_16x); // vel.x += acc.x * dt
        VelY_16x = _mm512_fmadd_ps(AccY_16x, dt, VelY_16x); // vel.y += acc.y * dt
        VelZ_16x = _mm512_fmadd_ps(AccZ_16x, dt, VelZ_16x); // vel.z += acc.z * dt
        PosX_16x = _mm512_fmadd_ps(_mm512_add_ps(VelX_16x, PrevVelX_16x), dt_per_two, PosX_16x); // pos.x += (dt / 2.0f * (vel.x + prevvel.x))
        PosY_16x = _mm512_fmadd_ps(_mm512_add_ps(VelY_16x, PrevVelY_16x), dt_per_two, PosY_16x); // pos.y += (dt / 2.0f * (vel.y + prevvel.y))
        PosZ_16x = _mm512_fmadd_ps(_mm512_add_ps(VelZ_16x, PrevVelZ_16x), dt_per_two, PosZ_16x); // pos.z += (dt / 2.0f * (vel.z + prevvel.z))

        _mm512_store_ps(&Velocities2.dx[i], VelX_16x);
        _mm512_store_ps(&Velocities2.dy[i], VelY_16x);
        _mm512_store_ps(&Velocities2.dz[i], VelZ_16x);
        _mm512_store_ps(&Positions2.x[i], PosX_16x);
        _mm512_store_ps(&Positions2.y[i], PosY_16x);
        _mm512_store_ps(&Positions2.z[i], PosZ_16x);
    }
# elif defined(__AVX2__)
/***
 * number of cycles, throughput
 * _mm256_load_ps  - 7, 0.5
 * _mm256_fmadd_ps - 4, 0.5
 * _mm256_add_ps   - 4, 0.5
 * _mm256_store_ps - 1, 1
 * 
 * n  instruction      total cycles
 * 12 _mm256_load_ps   84
 * 6  _mm256_fmadd_ps  24
 * 3  _mm256_add_ps    12
 * 6  _mm256_store_ps  6
 * 
 * sum cycle           126
 * 
 * iters(N_OF_ITEMS / 8)     total cycles(iters * sum cycles)
 * 131,072                   16,515,072
 */ 
    __m256 dt = _mm256_set1_ps(1.0f / 60.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 dt_per_two = _mm256_div_ps(dt, two);
    u32 Stride = BYTE_BOUNDARY / sizeof(r32);
    u32 Iterations = N_OF_ITEMS - N_OF_ITEMS % Stride;
    for (u32 i = 0; i < Iterations; i += Stride)
    {
        __m256 PosX_8x = _mm256_load_ps(Positions2.x + i);
        __m256 PosY_8x = _mm256_load_ps(Positions2.y + i);
        __m256 PosZ_8x = _mm256_load_ps(Positions2.z + i);
        __m256 VelX_8x = _mm256_load_ps(Velocities2.dx + i);
        __m256 VelY_8x = _mm256_load_ps(Velocities2.dy + i);
        __m256 VelZ_8x = _mm256_load_ps(Velocities2.dz + i);
        __m256 PrevVelX_8x = _mm256_load_ps(Velocities2.dx + i);
        __m256 PrevVelY_8x = _mm256_load_ps(Velocities2.dy + i);
        __m256 PrevVelZ_8x = _mm256_load_ps(Velocities2.dz + i);
        __m256 AccX_8x = _mm256_load_ps(Accelerations2.ddx + i);
        __m256 AccY_8x = _mm256_load_ps(Accelerations2.ddy + i);
        __m256 AccZ_8x = _mm256_load_ps(Accelerations2.ddz + i);

        VelX_8x = _mm256_fmadd_ps(AccX_8x, dt, VelX_8x); // vel.x += acc.x * dt
        VelY_8x = _mm256_fmadd_ps(AccY_8x, dt, VelY_8x); // vel.y += acc.y * dt
        VelZ_8x = _mm256_fmadd_ps(AccZ_8x, dt, VelZ_8x); // vel.z += acc.z * dt
        PosX_8x = _mm256_fmadd_ps(_mm256_add_ps(VelX_8x, PrevVelX_8x), dt_per_two, PosX_8x); // pos.x += (dt / 2.0f * (vel.x + prevvel.x))
        PosY_8x = _mm256_fmadd_ps(_mm256_add_ps(VelY_8x, PrevVelY_8x), dt_per_two, PosY_8x); // pos.y += (dt / 2.0f * (vel.y + prevvel.y))
        PosZ_8x = _mm256_fmadd_ps(_mm256_add_ps(VelZ_8x, PrevVelZ_8x), dt_per_two, PosZ_8x); // pos.z += (dt / 2.0f * (vel.z + prevvel.z))

        _mm256_store_ps(Velocities2.dx + i, VelX_8x);
        _mm256_store_ps(Velocities2.dy + i, VelY_8x);
        _mm256_store_ps(Velocities2.dz + i, VelZ_8x);
        _mm256_store_ps(Positions2.x + i, PosX_8x);
        _mm256_store_ps(Positions2.y + i, PosY_8x);
        _mm256_store_ps(Positions2.z + i, PosZ_8x);
    }
# elif defined(__SSE__)
    __m128 dt = _mm_set1_ps(1.0f / 60.0f);
    __m128 two = _mm_set1_ps(2.0f);
    __m128 dt_per_two = _mm_div_ps(dt, two);
    u32 Stride = BYTE_BOUNDARY / sizeof(r32);
    u32 Iterations = N_OF_ITEMS - N_OF_ITEMS % Stride;
    for (u32 i = 0; i < Iterations; i += Stride)
    {
        __m128 PosX_4x = _mm_load_ps(&Positions2.x[i]);
        __m128 PosY_4x = _mm_load_ps(&Positions2.y[i]);
        __m128 PosZ_4x = _mm_load_ps(&Positions2.z[i]);
        __m128 VelX_4x = _mm_load_ps(&Velocities2.dx[i]);
        __m128 VelY_4x = _mm_load_ps(&Velocities2.dy[i]);
        __m128 VelZ_4x = _mm_load_ps(&Velocities2.dz[i]);
        __m128 PrevVelX_4x = _mm_load_ps(&Velocities2.dx[i]);
        __m128 PrevVelY_4x = _mm_load_ps(&Velocities2.dy[i]);
        __m128 PrevVelZ_4x = _mm_load_ps(&Velocities2.dz[i]);
        __m128 AccX_4x = _mm_load_ps(&Accelerations2.ddx[i]);
        __m128 AccY_4x = _mm_load_ps(&Accelerations2.ddy[i]);
        __m128 AccZ_4x = _mm_load_ps(&Accelerations2.ddz[i]);

        VelX_4x = _mm_fmadd_ps(AccX_4x, dt, VelX_4x); // vel.x += acc.x * dt
        VelY_4x = _mm_fmadd_ps(AccY_4x, dt, VelY_4x); // vel.y += acc.y * dt
        VelZ_4x = _mm_fmadd_ps(AccZ_4x, dt, VelZ_4x); // vel.z += acc.z * dt
        PosX_4x = _mm_fmadd_ps(_mm_add_ps(VelX_4x, PrevVelX_4x), dt_per_two, PosX_4x); // pos.x += (dt / 2.0f * (vel.x + prevvel.x))
        PosY_4x = _mm_fmadd_ps(_mm_add_ps(VelY_4x, PrevVelY_4x), dt_per_two, PosY_4x); // pos.y += (dt / 2.0f * (vel.y + prevvel.y))
        PosZ_4x = _mm_fmadd_ps(_mm_add_ps(VelZ_4x, PrevVelZ_4x), dt_per_two, PosZ_4x); // pos.z += (dt / 2.0f * (vel.z + prevvel.z))

        _mm_store_ps(&Velocities2.dx[i], VelX_4x);
        _mm_store_ps(&Velocities2.dy[i], VelY_4x);
        _mm_store_ps(&Velocities2.dz[i], VelZ_4x);
        _mm_store_ps(&Positions2.x[i], PosX_4x);
        _mm_store_ps(&Positions2.y[i], PosY_4x);
        _mm_store_ps(&Positions2.z[i], PosZ_4x);
    }
# else
#  error "instruction set not supported"
# endif // avx512f, avx, sse
#endif // fma

IACA_END;

}

inline b32
Limit(r32 X, r32 Limit, r32 Epsilon)
{
    return (Limit - Epsilon <= X && X <= Limit + Epsilon);
}

int main()
{
    Positions.reserve(N_OF_ITEMS);
    Velocities.reserve(N_OF_ITEMS);
    Accelerations.reserve(N_OF_ITEMS);

    Positions2.x = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Positions2.y = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Positions2.z = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Velocities2.dx = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Velocities2.dy = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Velocities2.dz = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Accelerations2.ddx = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Accelerations2.ddy = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);
    Accelerations2.ddz = (r32 *)_aligned_malloc(N_OF_ITEMS * sizeof(r32), BYTE_BOUNDARY);

    for (u32 i = 0; i < N_OF_ITEMS; ++i)
    {
        Positions.push_back({ GetRand(-100000.0f, 100000.0f), GetRand(-100000.0f, 100000.0f), GetRand(-100000.0f, 100000.0f) });
        Velocities.push_back({ GetRand(-100.0f, 100.0f), GetRand(-100.0f, 100.0f), GetRand(-100.0f, 100.0f) });
        Accelerations.push_back({ GetRand(-10.0f, 10.0f), GetRand(-10.0f, 10.0f), GetRand(-10.0f, 10.0f) });
        Positions2.x[i] = Positions[i].x;
        Positions2.y[i] = Positions[i].y;
        Positions2.z[i] = Positions[i].z;
        Velocities2.dx[i] = Velocities[i].dx;
        Velocities2.dy[i] = Velocities[i].dy;
        Velocities2.dz[i] = Velocities[i].dz;
        Accelerations2.ddx[i] = Accelerations[i].ddx;
        Accelerations2.ddy[i] = Accelerations[i].ddy;
        Accelerations2.ddz[i] = Accelerations[i].ddz;
    }

    u64 CyclesStart = __rdtsc();
    SingleInstructionSingleData();
    u64 CyclesEnd = __rdtsc();
    LOG(setw(40 + strlen("Clock cycles")) << "Clock cycles");
    LOG(setw(40) << "Single Instruction Single Data: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");

    CyclesStart = __rdtsc();
    SingleInstructionMultipleData();
    CyclesEnd = __rdtsc();
    LOG(setw(40) << "Single Instruction Multiple Data: " << (CyclesEnd - CyclesStart) / 1000000.0f << "M");

    r32 Epsilon = 0.1f;
    for (u32 Index = 0;
         Index < N_OF_ITEMS;
         ++Index)
    {
        b32 Result =
            (Limit(Positions[Index].x, Positions2.x[Index], Epsilon) && Limit(Positions[Index].y, Positions2.y[Index], Epsilon) && Limit(Positions[Index].z, Positions2.z[Index], Epsilon) &&
             Limit(Velocities[Index].dx, Velocities2.dx[Index], Epsilon) && Limit(Velocities[Index].dy, Velocities2.dy[Index], Epsilon) && Limit(Velocities[Index].dz, Velocities2.dz[Index], Epsilon) &&
             Limit(Accelerations[Index].ddx, Accelerations2.ddx[Index], Epsilon) && Limit(Accelerations[Index].ddy, Accelerations2.ddy[Index], Epsilon) && Limit(Accelerations[Index].ddz, Accelerations2.ddz[Index], Epsilon));
        if (Result == false)
        {
            LOG(Positions[Index].x);
            LOG(Positions2.x[Index]);
            LOG(Positions[Index].y);
            LOG(Positions2.y[Index]);
            LOG(Positions[Index].z);
            LOG(Positions2.z[Index]);
            LOG(Velocities[Index].dx);
            LOG(Velocities2.dx[Index]);
            LOG(Velocities[Index].dy);
            LOG(Velocities2.dy[Index]);
            LOG(Velocities[Index].dz);
            LOG(Velocities2.dz[Index]);
            LOG(Accelerations[Index].ddx);
            LOG(Accelerations2.ddx[Index]);
            LOG(Accelerations[Index].ddy);
            LOG(Accelerations2.ddy[Index]);
            LOG(Accelerations[Index].ddz);
            LOG(Accelerations2.ddz[Index]);
            LOG("Failure, the two data sets are not equal..");
            exit(1);
        }
    }
    LOG("Success, the two data sets are equal!");
}
