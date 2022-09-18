#include "common.hpp"

/***
 * The point of this example is to show the concept of memory-alignment.
 * As the CPU reads at its word size, for a lot of reasons like speed, atomicity and cache-line aligment, the compiler will likely pad our structures in the following ways:
 * - Padding of each data type is aligned to the its size byte boundary.
 * - Padding at the end of the struct is padded to the biggest data type size boundary.
 */

struct not_padded_not_aligned
{
    char a; // 1 byte, aligned at 1-byte boundaries, starts at 0th byte
    i32 b;  // 4 byte, aligned at 4-byte boundaries, starts at 4th byte
    char c; // 1 byte, aligned at 1-byte boundaries, starts at 8th byte

            // biggest data type is 4 byte, so the whole structure need to be on a 4 byte boundary
            // currently we are on the 9th byte boundary, so we need 3 more bytes, making the struct a total of 12 bytes
};

// essentially same structure as above but now we can name the fields that are essentially free to use as they are likely loaded in (likely if )
struct padded_not_aligned
{
    char a;
    char pad0[3];
    i32 b;
    char c;
    char pad1[3];
};

// reordering our variables we save space that would otherwise be padded
struct padded_aligned
{
    i32 b;  // 4 byte, aligned at 4-byte boundaries, starts at 0th byte
    char a; // 1 byte, aligned at 1-byte boundaries, starts at 4th byte
    char c; // 1 byte, aligned at 1-byte boundaries, starts at 5th byte
    // biggest data 4 bytes, struct needs to be on 4-byte boundaries, currently we are on 6th byte -> 2 extra byte is padded
    char pad[2];
};

// https://gcc.gnu.org/onlinedocs/gcc-4.4.4/gcc/Structure_002dPacking-Pragmas.html
struct unpacked
{
    char c; // 1 byte, aligned at 1-byte boundaries, starts at 0th byte
    i32 a;  // 4 byte, aligned at 4-byte boundaries, starts at 4th byte
    i16 s;  // 2 byte, aligned at 2-byte boundaries, starts at 8th byte
    // biggest data 4 bytes, struct needs to be on 4-byte boundaries, currently we are on 10th byte -> 2 extra byte is padded
};

#pragma pack(push, 1) // enforce all data types to be on 1-byte boundaries
struct packed
{
    char c; // 1 byte, aligned at 1-byte boundaries, starts at 0th byte
    i32 a;  // 4 byte, aligned at 1-byte boundaries, starts at 1st byte
    i16 s;  // 2 byte, aligned at 1-byte boundaries, starts at 5th byte
    // no padding required as all data are on 1-byte boundaries
};
#pragma pack(pop)

i32 main(void)
{
    LOG(setw(40) << "sizeof(not_padded_not_aligned): " << sizeof(not_padded_not_aligned));
    LOG(setw(40) << "sizeof(padded_not_aligned): " << sizeof(padded_not_aligned));
    LOG(setw(40) << "sizeof(padded_aligned): " << sizeof(padded_aligned));
    LOG(setw(40) << "sizeof(unpacked): " << sizeof(unpacked));
    LOG(setw(40) << "sizeof(packed): " << sizeof(packed));
}
