// EcAsm.h - Assembly Primitive Declarations
// Optimized x86-64 assembly implementations for secp256k1 operations

#ifndef ECASM_H
#define ECASM_H

#include "defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Assembly primitives for 256-bit modular arithmetic
// All operations are mod P where P = secp256k1 prime
// Input/output: 4×64-bit limbs (little-endian)

// Modular addition: res = (a + b) mod P
void AddModP_asm(u64* res, const u64* a, const u64* b);

// Modular subtraction: res = (a - b) mod P
void SubModP_asm(u64* res, const u64* a, const u64* b);

// Modular multiplication: res = (a * b) mod P
// Note: May return value in range [0, 2P)
void MulModP_asm(u64* res, const u64* a, const u64* b);

// Modular inverse: res = a^(-1) mod P
// Uses AVX2 optimized implementation
// Note: May return value in range [0, 2P)
void InvModP_asm(u64* res, const u64* a);

#ifdef __cplusplus
}
#endif

#endif // ECASM_H
