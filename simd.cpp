#include "common.hpp"
#include "immintrin.h"

#define N_OF_ITEMS 1048576

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

vector<position> Positions;
vector<velocity> Velocities;
vector<acceleration> Accelerations;
vector<position> Positions2;
vector<velocity> Velocities2;
vector<acceleration> Accelerations2;

void
SingleInstructionSingleData(void)
{
    r32 dt = 1.0f / 60.0f;
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
        r32 MulledVelX = dt * SummedVelX;
        r32 MulledVelY = dt * SummedVelY;
        r32 MulledVelZ = dt * SummedVelZ;
        r32 DeltaPosX = MulledVelX / 2.0f;
        r32 DeltaPosY = MulledVelY / 2.0f;
        r32 DeltaPosZ = MulledVelZ / 2.0f;
        Positions[i].x += DeltaPosX;
        Positions[i].y += DeltaPosY;
        Positions[i].z += DeltaPosZ;
    }
}

void
SingleInstructionMultipleData(void)
{
    // initializing constants
    __m128 dt = _mm_set1_ps(1.0f / 60.0f);
    __m128 two = _mm_set1_ps(2.0f);
    for (u32 i = 0; i < N_OF_ITEMS; i += 4)
    {
        __m128 PosX_4x = _mm_set_ps(Positions2[i + 3].x, Positions2[i + 2].x, Positions2[i + 1].x, Positions2[i].x);
        __m128 PosY_4x = _mm_set_ps(Positions2[i + 3].y, Positions2[i + 2].y, Positions2[i + 1].y, Positions2[i].y);
        __m128 PosZ_4x = _mm_set_ps(Positions2[i + 3].z, Positions2[i + 2].z, Positions2[i + 1].z, Positions2[i].z);
        __m128 VelX_4x = _mm_set_ps(Velocities2[i + 3].dx, Velocities2[i + 2].dx, Velocities2[i + 1].dx, Velocities2[i].dx);
        __m128 VelY_4x = _mm_set_ps(Velocities2[i + 3].dy, Velocities2[i + 2].dy, Velocities2[i + 1].dy, Velocities2[i].dy);
        __m128 VelZ_4x = _mm_set_ps(Velocities2[i + 3].dz, Velocities2[i + 2].dz, Velocities2[i + 1].dz, Velocities2[i].dz);
        __m128 PrevVelX_4x = _mm_set_ps(Velocities2[i + 3].dx, Velocities2[i + 2].dx, Velocities2[i + 1].dx, Velocities2[i].dx);
        __m128 PrevVelY_4x = _mm_set_ps(Velocities2[i + 3].dy, Velocities2[i + 2].dy, Velocities2[i + 1].dy, Velocities2[i].dy);
        __m128 PrevVelZ_4x = _mm_set_ps(Velocities2[i + 3].dz, Velocities2[i + 2].dz, Velocities2[i + 1].dz, Velocities2[i].dz);
        __m128 AccX_4x = _mm_set_ps(Accelerations2[i + 3].ddx, Accelerations2[i + 2].ddx, Accelerations2[i + 1].ddx, Accelerations2[i].ddx);
        __m128 AccY_4x = _mm_set_ps(Accelerations2[i + 3].ddy, Accelerations2[i + 2].ddy, Accelerations2[i + 1].ddy, Accelerations2[i].ddy);
        __m128 AccZ_4x = _mm_set_ps(Accelerations2[i + 3].ddz, Accelerations2[i + 2].ddz, Accelerations2[i + 1].ddz, Accelerations2[i].ddz);
        __m128 DeltaVelX = _mm_mul_ps(AccX_4x, dt); // acc.x * dt
        __m128 DeltaVelY = _mm_mul_ps(AccY_4x, dt); // acc.y * dt
        __m128 DeltaVelZ = _mm_mul_ps(AccZ_4x, dt); // acc.z * dt
        VelX_4x = _mm_add_ps(VelX_4x, DeltaVelX); // vel.x += acc.x * dt
        VelY_4x = _mm_add_ps(VelY_4x, DeltaVelY); // vel.y += acc.y * dt
        VelZ_4x = _mm_add_ps(VelZ_4x, DeltaVelZ); // vel.z += acc.z * dt
        __m128 SummedVelX = _mm_add_ps(VelX_4x, PrevVelX_4x); // vel.x + prevvel.x
        __m128 SummedVelY = _mm_add_ps(VelY_4x, PrevVelY_4x); // vel.y + prevvel.y
        __m128 SummedVelZ = _mm_add_ps(VelZ_4x, PrevVelZ_4x); // vel.z + prevvel.z
        __m128 MulledVelX = _mm_mul_ps(dt, SummedVelX); // dt * (vel.x + prevvel.x)
        __m128 MulledVelY = _mm_mul_ps(dt, SummedVelY); // dt * (vel.y + prevvel.y)
        __m128 MulledVelZ = _mm_mul_ps(dt, SummedVelZ); // dt * (vel.z + prevvel.z)
        __m128 DeltaPosX = _mm_div_ps(MulledVelX, two); // (dt * (vel.x + prevvel.x)) / 2.0f
        __m128 DeltaPosY = _mm_div_ps(MulledVelY, two); // (dt * (vel.y + prevvel.y)) / 2.0f
        __m128 DeltaPosZ = _mm_div_ps(MulledVelZ, two); // (dt * (vel.z + prevvel.z)) / 2.0f
        PosX_4x = _mm_add_ps(PosX_4x, DeltaPosX); // pos.x += (dt * (vel.x + prevvel.x)) / 2.0f
        PosY_4x = _mm_add_ps(PosY_4x, DeltaPosY); // pos.y += (dt * (vel.y + prevvel.y)) / 2.0f
        PosZ_4x = _mm_add_ps(PosZ_4x, DeltaPosZ); // pos.z += (dt * (vel.z + prevvel.z)) / 2.0f
        for (u32 j = 0; j < 4; ++j)
        {
            Velocities2[i + j].dx = VelX_4x[j];
            Velocities2[i + j].dy = VelY_4x[j];
            Velocities2[i + j].dz = VelZ_4x[j];
            Positions2[i + j].x = PosX_4x[j];
            Positions2[i + j].y = PosY_4x[j];
            Positions2[i + j].z = PosZ_4x[j];
        }
        
    }
}

int main()
{
    Positions.reserve(N_OF_ITEMS);
    Velocities.reserve(N_OF_ITEMS);
    Accelerations.reserve(N_OF_ITEMS);

    for (u32 i = 0; i < N_OF_ITEMS; ++i)
    {
        Positions.push_back({ GetRand(-100000.0f, 100000.0f), GetRand(-100000.0f, 100000.0f), GetRand(-100000.0f, 100000.0f) });
        Velocities.push_back({ GetRand(-100.0f, 100.0f), GetRand(-100.0f, 100.0f), GetRand(-100.0f, 100.0f) });
        Accelerations.push_back({ GetRand(-10.0f, 10.0f), GetRand(-10.0f, 10.0f), GetRand(-10.0f, 10.0f) });
        Positions2.push_back(Positions[i]);
        Velocities2.push_back(Velocities[i]);
        Accelerations2.push_back(Accelerations[i]);
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

    if (Positions == Positions2 &&
        Velocities == Velocities2 &&
        Accelerations == Accelerations2)
    {
        LOG("Success, the two data sets are equal!");
    }
    else
    {
        LOG("Failure, the two data sets are not equal..");
    }
}
