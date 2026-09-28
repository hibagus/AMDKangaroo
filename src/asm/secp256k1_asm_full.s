.intel_syntax noprefix
.text


.globl AddModP_asm
.type AddModP_asm, @function
AddModP_asm:
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r8,  QWORD PTR [rsi]
    mov r9,  QWORD PTR [rsi+8]
    mov r10, QWORD PTR [rsi+16]
    mov r11, QWORD PTR [rsi+24]
    
    add r8,  QWORD PTR [rdx]
    adc r9,  QWORD PTR [rdx+8]
    adc r10, QWORD PTR [rdx+16]
    adc r11, QWORD PTR [rdx+24]
    
    mov r12, r8
    mov r13, r9
    mov r14, r10
    mov r15, r11
    
    jnc .AddModP_no_overflow
    
    movabs rax, 0xFFFFFFFEFFFFFC2F
    sub r12, rax
    movabs rax, 0xFFFFFFFFFFFFFFFF
    sbb r13, rax
    sbb r14, rax
    sbb r15, rax
    jmp .AddModP_done
    
.AddModP_no_overflow:
    movabs rax, 0xFFFFFFFEFFFFFC2F
    cmp r12, rax
    jb .AddModP_done
    movabs rax, 0xFFFFFFFFFFFFFFFF
    cmp r13, rax
    jb .AddModP_done
    cmp r14, rax
    jb .AddModP_done
    cmp r15, rax
    jb .AddModP_done
    
    movabs rax, 0xFFFFFFFEFFFFFC2F
    sub r12, rax
    movabs rax, 0xFFFFFFFFFFFFFFFF
    sbb r13, rax
    sbb r14, rax
    sbb r15, rax
    
.AddModP_done:
    mov [rdi], r12
    mov [rdi+8], r13
    mov [rdi+16], r14
    mov [rdi+24], r15
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret


.globl SubModP_asm
.type SubModP_asm, @function
SubModP_asm:
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r8,  QWORD PTR [rsi]
    mov r9,  QWORD PTR [rsi+8]
    mov r10, QWORD PTR [rsi+16]
    mov r11, QWORD PTR [rsi+24]
    
    sub r8,  QWORD PTR [rdx]
    sbb r9,  QWORD PTR [rdx+8]
    sbb r10, QWORD PTR [rdx+16]
    sbb r11, QWORD PTR [rdx+24]
    
    jnc .SubModP_done
    
    movabs rax, 0xFFFFFFFEFFFFFC2F
    add r8, rax
    movabs rax, 0xFFFFFFFFFFFFFFFF
    adc r9, rax
    adc r10, rax
    adc r11, rax
    
.SubModP_done:
    mov [rdi], r8
    mov [rdi+8], r9
    mov [rdi+16], r10
    mov [rdi+24], r11
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret


.globl Mul256_by_64_asm
.type Mul256_by_64_asm, @function
Mul256_by_64_asm:
    push rbx
    push r12
    push r13
    
    mov rax, QWORD PTR [rdi]
    mul rsi
    mov r8, rax
    mov r9, rdx
    
    mov rax, QWORD PTR [rdi+8]
    mul rsi
    add r9, rax
    adc rdx, 0
    mov r10, rdx
    
    mov rax, QWORD PTR [rdi+16]
    mul rsi
    add r10, rax
    adc rdx, 0
    mov r11, rdx
    
    mov rax, QWORD PTR [rdi+24]
    mul rsi
    add r11, rax
    adc rdx, 0
    mov r12, rdx
    
    mov [rcx], r8
    mov [rcx+8], r9
    mov [rcx+16], r10
    mov [rcx+24], r11
    mov [rcx+32], r12
    
    pop r13
    pop r12
    pop rbx
    ret


.globl Add320_to_256_asm
.type Add320_to_256_asm, @function
Add320_to_256_asm:
    mov rax, QWORD PTR [rdi]
    add rax, QWORD PTR [rsi]
    mov [rdi], rax
    
    mov rax, QWORD PTR [rdi+8]
    adc rax, QWORD PTR [rsi+8]
    mov [rdi+8], rax
    
    mov rax, QWORD PTR [rdi+16]
    adc rax, QWORD PTR [rsi+16]
    mov [rdi+16], rax
    
    mov rax, QWORD PTR [rdi+24]
    adc rax, QWORD PTR [rsi+24]
    mov [rdi+24], rax
    
    mov rax, 0
    adc rax, QWORD PTR [rsi+32]
    mov [rdi+32], rax
    
    ret


.globl MulModP_asm
.type MulModP_asm, @function
MulModP_asm:
    push rbp
    mov rbp, rsp
    sub rsp, 144
    
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov QWORD PTR [rbp-8], rdi
    mov QWORD PTR [rbp-16], rsi
    mov QWORD PTR [rbp-24], rdx
    
    lea r15, [rbp-88]
    
    mov rdi, QWORD PTR [rbp-24]
    mov r14, QWORD PTR [rbp-16]
    mov rsi, QWORD PTR [r14]
    lea rcx, [rbp-88]
    call Mul256_by_64_asm
    
    mov rdi, QWORD PTR [rbp-24]
    mov r14, QWORD PTR [rbp-16]
    mov rsi, QWORD PTR [r14+8]
    lea rcx, [rbp-128]
    call Mul256_by_64_asm
    
    lea rdi, [rbp-80]
    lea rsi, [rbp-128]
    call Add320_to_256_asm
    
    mov rdi, QWORD PTR [rbp-24]
    mov r14, QWORD PTR [rbp-16]
    mov rsi, QWORD PTR [r14+16]
    lea rcx, [rbp-128]
    call Mul256_by_64_asm
    
    lea rdi, [rbp-72]
    lea rsi, [rbp-128]
    call Add320_to_256_asm
    
    mov rdi, QWORD PTR [rbp-24]
    mov r14, QWORD PTR [rbp-16]
    mov rsi, QWORD PTR [r14+24]
    lea rcx, [rbp-128]
    call Mul256_by_64_asm
    
    lea rdi, [rbp-64]
    lea rsi, [rbp-128]
    call Add320_to_256_asm
    
    lea rdi, [rbp-56]
    movabs rsi, 0x00000001000003D1
    lea rcx, [rbp-128]
    call Mul256_by_64_asm
    
    mov r8, QWORD PTR [rbp-88]
    mov r9, QWORD PTR [rbp-80]
    mov r10, QWORD PTR [rbp-72]
    mov r11, QWORD PTR [rbp-64]
    
    add r8, QWORD PTR [rbp-128]
    adc r9, QWORD PTR [rbp-120]
    adc r10, QWORD PTR [rbp-112]
    adc r11, QWORD PTR [rbp-104]
    
    mov QWORD PTR [rbp-88], r8
    mov QWORD PTR [rbp-80], r9
    mov QWORD PTR [rbp-72], r10
    mov QWORD PTR [rbp-64], r11
    
    mov r12, 0
    adc r12, QWORD PTR [rbp-96]
    
    mov rax, r12
    movabs rsi, 0x00000001000003D1
    mul rsi
    mov r13, rax
    mov r14, rdx
    
    mov r8, QWORD PTR [rbp-88]
    add r8, r13
    mov r9, QWORD PTR [rbp-80]
    adc r9, r14
    mov r10, QWORD PTR [rbp-72]
    adc r10, 0
    mov r11, QWORD PTR [rbp-64]
    mov r12, 0
    adc r11, 0
    adc r12, 0
    
    mov rdi, QWORD PTR [rbp-8]
    mov [rdi], r8
    mov [rdi+8], r9
    mov [rdi+16], r10
    mov [rdi+24], r11
    
    test r12, r12
    jz .MulModP_done
    
.MulModP_reduce_loop:
    movabs rax, 0xFFFFFFFEFFFFFC2F
    sub r8, rax
    movabs rax, 0xFFFFFFFFFFFFFFFF
    sbb r9, rax
    sbb r10, rax
    sbb r11, rax
    sbb r12, 0
    
    mov [rdi], r8
    mov [rdi+8], r9
    mov [rdi+16], r10
    mov [rdi+24], r11
    
    test r12, r12
    jnz .MulModP_reduce_loop
    
.MulModP_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    
    mov rsp, rbp
    pop rbp
    ret


.globl SqrModP_asm
.type SqrModP_asm, @function
SqrModP_asm:
    mov rdx, rsi
    jmp MulModP_asm

.section .note.GNU-stack,"",@progbits
