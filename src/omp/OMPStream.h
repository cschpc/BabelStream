
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <omp.h>

#if defined(AVX2_INTRINSICS)
  #if defined(NON_TEMPORAL)
    #define IMPLEMENTATION_STRING "OpenMP, AVX2 intrinsics with non-temp"
  #else
    #define IMPLEMENTATION_STRING "OpenMP, AVX2 intrinsics "
  #endif
#elif defined(AVX512_INTRINSICS)
  #if defined(NON_TEMPORAL)
    #define IMPLEMENTATION_STRING "OpenMP, AVX512 intrinsics with non-temp"
  #else
    #define IMPLEMENTATION_STRING "OpenMP, AVX512 intrinsics"
  #endif
#else
   #define IMPLEMENTATION_STRING "OpenMP"
#endif

template <class T>
class OMPStream : public Stream<T>
{
  protected:
    // Size of arrays
    size_t array_size;

    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    OMPStream(const long, int);
    ~OMPStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};
