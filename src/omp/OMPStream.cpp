
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "OMPStream.h"
#if defined(AVX2_INTRINSICS) or defined(AVX512_INTRINSICS)
#include <immintrin.h>
#endif

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
OMPStream<T>::OMPStream(const long ARRAY_SIZE, int device)
{
  array_size = ARRAY_SIZE;

  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);

#ifdef OMP_TARGET_GPU
  omp_set_default_device(device);
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  // Set up data region on device
  #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif

}

template <class T>
OMPStream<T>::~OMPStream()
{
#ifdef OMP_TARGET_GPU
  // End data region on device
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif
  free(a);
  free(b);
  free(c);
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  size_t array_size = this->array_size;
#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{

#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif

  #pragma omp parallel for
  for (size_t i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }

}

template <class T>
void OMPStream<T>::copy()
{
#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::mul()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(c[0:0])
  #endif
}

template <class T>
void OMPStream<T>::add()
{
#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::triad()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::nstream()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
T OMPStream<T>::dot()
{
  T sum{};

#ifdef OMP_TARGET_GPU
  size_t array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  #pragma omp target teams distribute parallel for simd map(tofrom: sum) reduction(+:sum)
#else
  #pragma omp parallel for reduction(+:sum)
#endif
  for (size_t i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}

// Only double versions for intrinsics
#if defined(AVX2_INTRINSICS) or defined(AVX512_INTRINSICS)
template <>
void OMPStream<double>::copy()
{
  #pragma omp parallel for
#if defined(AVX2_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=4)
  {
    __m256d _a = _mm256_loadu_pd(&a[i]);
#ifdef NON_TEMPORAL
    _mm256_stream_pd(&c[i], _a);
#else
    _mm256_storeu_pd(&c[i], _a);
#endif
  }
#elif defined(AVX512_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=8)
  {
    __m512d _a = _mm512_loadu_pd(&a[i]);
#ifdef NON_TEMPORAL
    _mm512_stream_pd(&c[i], _a);
#else
    _mm512_storeu_pd(&c[i], _a);
#endif
  }
#endif
}

template <>
void OMPStream<double>::mul()
{
  const double scalar = startScalar;

#if defined(AVX2_INTRINSICS)
  __m256d _scalar = _mm256_broadcast_sd(&scalar);
#elif defined(AVX512_INTRINSICS)
  __m512d _scalar = _mm512_set1_pd(scalar);
#endif

  #pragma omp parallel for
#if defined(AVX2_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=4)
  {
     __m256d _c = _mm256_loadu_pd(&c[i]);
     __m256d _res = _mm256_mul_pd(_scalar, _c);
#ifdef NON_TEMPORAL
     _mm256_stream_pd(&b[i], _res);
#else
     _mm256_store_pd(&b[i], _res);
#endif
  }
#elif defined(AVX512_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=8)
  {
     __m512d _c = _mm512_loadu_pd(&c[i]);
     __m512d _res = _mm512_mul_pd(_scalar, _c);
#ifdef NON_TEMPORAL
     _mm512_stream_pd(&b[i], _res);
#else
     _mm512_store_pd(&b[i], _res);
#endif
  }
#endif
}

template <>
void OMPStream<double>::add()
{
  #pragma omp parallel for
#if defined(AVX2_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=4)
  {
     __m256d _a = _mm256_loadu_pd(&a[i]);
     __m256d _b = _mm256_loadu_pd(&b[i]);
     __m256d _res = _mm256_add_pd(_a, _b);
#ifdef NON_TEMPORAL
     _mm256_stream_pd(&c[i], _res);
#else
     _mm256_storeu_pd(&c[i], _res);
#endif
  }
#elif defined(AVX512_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=8)
  {
     __m512d _a = _mm512_loadu_pd(&a[i]);
     __m512d _b = _mm512_loadu_pd(&b[i]);
     __m512d _res = _mm512_add_pd(_a, _b);
#ifdef NON_TEMPORAL
     _mm512_stream_pd(&c[i], _res);
#else
     _mm512_storeu_pd(&c[i], _res);
#endif
  }
#endif
}

template <>
void OMPStream<double>::triad()
{
  const double scalar = startScalar;

#if defined(AVX2_INTRINSICS)
  __m256d _scalar = _mm256_broadcast_sd(&scalar);
#elif defined(AVX512_INTRINSICS)
  __m512d _scalar = _mm512_set1_pd(scalar);
#endif

  #pragma omp parallel for
#if defined(AVX2_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=4)
  {
         __m256d _b = _mm256_loadu_pd(&b[i]);
         __m256d _c = _mm256_loadu_pd(&c[i]);
         __m256d _res = _mm256_fmadd_pd(_scalar, _c, _b);
#ifdef NON_TEMPORAL
         _mm256_stream_pd(&a[i], _res);
#else
         _mm256_storeu_pd(&a[i], _res);
#endif    
  }
#elif defined(AVX512_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=8)
  {
         __m512d _b = _mm512_loadu_pd(&b[i]);
         __m512d _c = _mm512_loadu_pd(&c[i]);
         __m512d _res = _mm512_fmadd_pd(_scalar, _c, _b);
#ifdef NON_TEMPORAL
         _mm512_stream_pd(&a[i], _res);
#else
         _mm512_storeu_pd(&a[i], _res);
#endif    
  }
#endif
}

template <>
void OMPStream<double>::nstream()
{
  const double scalar = startScalar;

  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <>
double OMPStream<double>::dot()
{
  double sum{};

  #pragma omp parallel for reduction(+:sum)
#if defined(AVX512_INTRINSICS)
  for (size_t i = 0; i < array_size; i+=8)
  {
         __m512d _a = _mm512_loadu_pd(&a[i]);
         __m512d _b = _mm512_loadu_pd(&b[i]);
         __m512d _sum = _mm512_mul_pd(_a, _b);
         sum +=  _mm512_reduce_add_pd(_sum);
  }
#else
  for (size_t i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }
#endif

  return sum;
}
#endif // Intrinsics versions


void listDevices(void)
{
#ifdef OMP_TARGET_GPU
  // Get number of devices
  int count = omp_get_num_devices();

  // Print device list
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << "There are " << count << " devices." << std::endl;
  }
#else
  std::cout << "0: CPU" << std::endl;
#endif
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class OMPStream<float>;
template class OMPStream<double>;
