#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define SWZ(m, n) (((m) * 4 + (n)) % 64) // by half

#endif