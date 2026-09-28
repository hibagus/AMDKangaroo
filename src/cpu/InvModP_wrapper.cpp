// Wrapper for inverse256_skylake optimized modular inverse
//
// Based on the work of Daniel J. Bernstein and Bo-Yin Yang:
// "Fast constant-time gcd computation and modular inversion"
// IACR Transactions on Cryptographic Hardware and Embedded Systems, 2019(3):340–398
// https://tches.iacr.org/index.php/TCHES/article/view/8298
//
// Implementation by Daniel J. Bernstein
// https://gcd.cr.yp.to/index.html
//
// This is an AVX2-optimized constant-time modular inverse for 256-bit primes,
// specifically optimized for the secp256k1 prime used in Bitcoin.

#include <stdint.h>
#include <string.h>

// External assembly function
extern "C" void inverse256_skylake_asm(const unsigned char *in, unsigned char *out, const int64_t *table);

// Bitcoin secp256k1 prime P table
static const __attribute__((aligned(32)))
int64_t t_BTC_p[64]={
    0x3FFFFFFFLL, 0x3FFFFFFFLL, 0x3FFFFFFFLL, 0x3FFFFFFFLL,
    0x200000000LL, 0x200000000LL, 0x200000000LL, 0x200000000LL,
    (int64_t)0x8000000000000000LL, (int64_t)0x8000000000000000LL,
    (int64_t)0x8000000000000000LL, (int64_t)0x8000000000000000LL,
    0X7FFFFFFE00000000LL, 0X7FFFFFFE00000000LL,
    0X7FFFFFFE00000000LL, 0X7FFFFFFE00000000LL,
    0x20000000LL, 0x20000000LL, 0x20000000LL, 0x20000000LL,
    (int64_t)0xfffffffefffffc2fULL, (int64_t)0xffffffffffffffffULL,
    (int64_t)0xffffffffffffffffULL, (int64_t)0xffffffffffffffffULL,
    0x03ffffc2fLL, 0LL, 0LL, 0LL,
    0x03ffffffbLL, 0LL, 0LL, 1LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x03fffffffLL, 0LL, 0LL, 0LL,
    0x00000ffffLL, 0LL, 0LL, 0LL,
    (int64_t)0xd838091dd2253531ULL, 0LL, 0LL, 0LL};

// Wrapper function that matches our interface
// Input/output: 4x64-bit limbs (little-endian)
extern "C" void InvModP_asm(uint64_t* res, const uint64_t* a) {
    unsigned char in[32];
    unsigned char out[32];
    
    // Convert from u64[4] to byte array (little-endian)
    memcpy(in, a, 32);
    
    // Call optimized inverse
    inverse256_skylake_asm(in, out, t_BTC_p);
    
    // Convert back to u64[4]
    memcpy(res, out, 32);
}
