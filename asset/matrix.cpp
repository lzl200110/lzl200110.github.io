#include <chrono>
#include <cstring>
#include <emmintrin.h>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <new>
#include <omp.h>
#include <xmmintrin.h>

// 计时功能
#define Tick auto __begin = std::chrono::steady_clock::now();
#define ReTick __begin = std::chrono::steady_clock::now();
#define Tock                                                                       \
	{                                                                              \
		auto __end = std::chrono::steady_clock::now();                             \
		auto __time = __end - __begin;                                             \
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(__time) \
		                 .count()                                                  \
		          << "ms" << std::endl;                                            \
	}

#ifdef AVX512
const int Align_Val = 512;
#else
const int Align_Val = 256;
#endif
//------------------------generate matrix -----------------------------
float rand_float(float s) {
	return s * (1 - s) * 4;
}

void matrix_gen(float* a, float* b, int N, float seed) {
	float s = seed;
	for (int i = 0; i < N * N; i++) {
		s = rand_float(s);
		a[i] = s;
		s = rand_float(s);
		b[i] = s;
	}
}
//---------------------------矩阵转化---------------------------------------
/**
 * @brief    将矩阵划分为小矩阵,保证每个小矩阵的空间连续
 *
 * @param matrix    原矩阵
 * @param matirx_s  分块矩阵
 * @param N         原矩阵大小
 * @param M         块大小
 */
void transform_matrix_s(float* matrix, float* matirx_s, int N, int M) {
	// 分块矩阵每个小矩阵的大小
	int N_SMALL = N / M;
	int i, j, k;
	float* start = matirx_s;
	// 构造分块矩阵 的 matirx_s[i][j][k] --> matrix[i*M+k][j*M]
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < M; k++) {
				memcpy(start, matrix + (i * M + k) * N + j * M, M * sizeof(float));
				start += M;
			}
		}
	}
}
// 把小矩阵转化为大矩阵
void transform_matrix_b(float* matrix, float* matirx_b, int N, int M) {
	int N_SMALL = N / M;
	int i, j, k;
	float* start = matirx_b;
	// 每一个小矩阵
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < M; k++) {
				memcpy(matirx_b + i * N * M + k * N + j * M,
				       matrix + N * M * i + j * M * M + k * M, M * sizeof(float));
				start += M;
			}
		}
	}
}

//----------------------------baseline------------------------------------------
void baseline_matrix_multi(float* a, float* b, float* c, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float sum = 0.0f;
			for (int k = 0; k < N; k++) {
				sum += a[i * N + k] * b[k * N + j];
			}
			c[i * N + j] = sum;
		}
	}
}
//----------------------------  分块 -----------------------------------------
void matrix_multi_add_s(float* a, float* b, float* c, int M) {
	int i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			float sum = 0.0f;
			for (int k = 0; k < M; k++) {
				sum += a[i * M + k] * b[k * M + j];
			}
			c[i * M + j] += sum;
		}
	}
}
void partition_matrix_multi_omp(float* a, float* b, float* c, int M, int N_SMALL) {
	int i, j, k;
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (int k = 0; k < N_SMALL; k++) {
				matrix_multi_add_s(a + M * M * N_SMALL * i + M * M * k,
				                   b + M * M * N_SMALL * k + M * M * j,
				                   c + M * M * N_SMALL * i + M * M * j, M);
			}
		}
	}
}
//------------------------------分块+SIMD---------------------------------------
// 8*8*8 的SIMD计算 AVX2
#define MATRIX_8X8X8(m)                                                         \
	{                                                                           \
		vb = _mm256_load_ps(&(b[(m)*M]));                                       \
		vc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[0 * M + (m)])), vb, vc0); \
		vc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[1 * M + (m)])), vb, vc1); \
		vc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[2 * M + (m)])), vb, vc2); \
		vc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[3 * M + (m)])), vb, vc3); \
		vc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[4 * M + (m)])), vb, vc4); \
		vc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[5 * M + (m)])), vb, vc5); \
		vc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[6 * M + (m)])), vb, vc6); \
		vc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&(a[7 * M + (m)])), vb, vc7); \
	}
// 8*8*8 的SIMD计算 AVX2
void matrix_multi_add_simd8X8X8(float* a, float* b, float* c, int M) {
	__m256 vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7;
	vc0 = _mm256_load_ps(&(c[0]));
	vc1 = _mm256_load_ps(&(c[M]));
	vc2 = _mm256_load_ps(&(c[2 * M]));
	vc3 = _mm256_load_ps(&(c[3 * M]));
	vc4 = _mm256_load_ps(&(c[4 * M]));
	vc5 = _mm256_load_ps(&(c[5 * M]));
	vc6 = _mm256_load_ps(&(c[6 * M]));
	vc7 = _mm256_load_ps(&(c[7 * M]));
	__m256 vb;
	MATRIX_8X8X8(0);
	MATRIX_8X8X8(1);
	MATRIX_8X8X8(2);
	MATRIX_8X8X8(3);
	MATRIX_8X8X8(4);
	MATRIX_8X8X8(5);
	MATRIX_8X8X8(6);
	MATRIX_8X8X8(7);
	_mm256_store_ps(&(c[0]), vc0);
	_mm256_store_ps(&(c[M]), vc1);
	_mm256_store_ps(&(c[M * 2]), vc2);
	_mm256_store_ps(&(c[M * 3]), vc3);
	_mm256_store_ps(&(c[M * 4]), vc4);
	_mm256_store_ps(&(c[M * 5]), vc5);
	_mm256_store_ps(&(c[M * 6]), vc6);
	_mm256_store_ps(&(c[M * 7]), vc7);
}

// 小矩阵的 乘法 ，使用SIMD计算
void matrix_multi_add_avx_s(float* a, float* b, float* c, int M) {
	int i, j, k;
	int SM = M / 8;
	for (i = 0; i < SM; i++) {
		for (j = 0; j < SM; j++) {
			for (int k = 0; k < SM; k++) {
				matrix_multi_add_simd8X8X8(a + M * 8 * i + 8 * k,
				                           b + M * 8 * k + 8 * j,
				                           c + M * 8 * i + 8 * j, M);
			}
		}
	}
}

void partition_matrix_multi_avx(float* a, float* b, float* c, int M, int N_SMALL) {
	int i, j, k;
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < N_SMALL; k++) {
				matrix_multi_add_avx_s(a + M * M * N_SMALL * i + M * M * k,
				                       b + M * M * N_SMALL * k + M * M * j,
				                       c + M * M * N_SMALL * i + M * M * j, M);
			}
		}
	}
}
//------------------------------分块+SIMD+omp---------------------------------------
// N_SMALL 表示大矩阵划分为后分块矩阵的维度 ， M表示每个分块矩阵的大小，
// 再次划分并使用SIMD计算nf乘法
void partition_matrix_multi_avx_omp(float* a, float* b, float* c, int M,
                                    int N_SMALL) {
	int i, j, k;
#pragma omp parallel for private(i, j, k)
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < N_SMALL; k++) {
				matrix_multi_add_avx_s(a + M * M * N_SMALL * i + M * M * k,
				                       b + M * M * N_SMALL * k + M * M * j,
				                       c + M * M * N_SMALL * i + M * M * j, M);
			}
		}
	}
}
#ifdef AVX512
#define MATRIX_16X16X16(m)                                                 \
	{                                                                      \
		vb = _mm512_load_ps(&(b[(m)*M]));                                  \
		vc0 = _mm512_fmadd_ps(_mm512_set1_ps(a[0 * M + (m)]), vb, vc0);    \
		vc1 = _mm512_fmadd_ps(_mm512_set1_ps(a[1 * M + (m)]), vb, vc1);    \
		vc2 = _mm512_fmadd_ps(_mm512_set1_ps(a[2 * M + (m)]), vb, vc2);    \
		vc3 = _mm512_fmadd_ps(_mm512_set1_ps(a[3 * M + (m)]), vb, vc3);    \
		vc4 = _mm512_fmadd_ps(_mm512_set1_ps(a[4 * M + (m)]), vb, vc4);    \
		vc5 = _mm512_fmadd_ps(_mm512_set1_ps(a[5 * M + (m)]), vb, vc5);    \
		vc6 = _mm512_fmadd_ps(_mm512_set1_ps(a[6 * M + (m)]), vb, vc6);    \
		vc7 = _mm512_fmadd_ps(_mm512_set1_ps(a[7 * M + (m)]), vb, vc7);    \
		vc8 = _mm512_fmadd_ps(_mm512_set1_ps(a[8 * M + (m)]), vb, vc8);    \
		vc9 = _mm512_fmadd_ps(_mm512_set1_ps(a[9 * M + (m)]), vb, vc9);    \
		vc10 = _mm512_fmadd_ps(_mm512_set1_ps(a[10 * M + (m)]), vb, vc10); \
		vc11 = _mm512_fmadd_ps(_mm512_set1_ps(a[11 * M + (m)]), vb, vc11); \
		vc12 = _mm512_fmadd_ps(_mm512_set1_ps(a[12 * M + (m)]), vb, vc12); \
		vc13 = _mm512_fmadd_ps(_mm512_set1_ps(a[13 * M + (m)]), vb, vc13); \
		vc14 = _mm512_fmadd_ps(_mm512_set1_ps(a[14 * M + (m)]), vb, vc14); \
		vc15 = _mm512_fmadd_ps(_mm512_set1_ps(a[15 * M + (m)]), vb, vc15); \
	}
void matrix_multi_add_simd16X16X16(float* a, float* b, float* c, int M) {
	__m512 vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13,
	    vc14, vc15;
	vc0 = _mm512_load_ps(&(c[0]));
	vc1 = _mm512_load_ps(&(c[M]));
	vc2 = _mm512_load_ps(&(c[2 * M]));
	vc3 = _mm512_load_ps(&(c[3 * M]));
	vc4 = _mm512_load_ps(&(c[4 * M]));
	vc5 = _mm512_load_ps(&(c[5 * M]));
	vc6 = _mm512_load_ps(&(c[6 * M]));
	vc7 = _mm512_load_ps(&(c[7 * M]));
	vc8 = _mm512_load_ps(&(c[8 * M]));
	vc9 = _mm512_load_ps(&(c[9 * M]));
	vc10 = _mm512_load_ps(&(c[10 * M]));
	vc11 = _mm512_load_ps(&(c[11 * M]));
	vc12 = _mm512_load_ps(&(c[12 * M]));
	vc13 = _mm512_load_ps(&(c[13 * M]));
	vc14 = _mm512_load_ps(&(c[14 * M]));
	vc15 = _mm512_load_ps(&(c[15 * M]));
	__m512 vb;
	MATRIX_16X16X16(0);
	MATRIX_16X16X16(1);
	MATRIX_16X16X16(2);
	MATRIX_16X16X16(3);
	MATRIX_16X16X16(4);
	MATRIX_16X16X16(5);
	MATRIX_16X16X16(6);
	MATRIX_16X16X16(7);
	MATRIX_16X16X16(8);
	MATRIX_16X16X16(9);
	MATRIX_16X16X16(10);
	MATRIX_16X16X16(11);
	MATRIX_16X16X16(12);
	MATRIX_16X16X16(13);
	MATRIX_16X16X16(14);
	MATRIX_16X16X16(15);
	_mm512_store_ps(&(c[0]), vc0);
	_mm512_store_ps(&(c[M]), vc1);
	_mm512_store_ps(&(c[M * 2]), vc2);
	_mm512_store_ps(&(c[M * 3]), vc3);
	_mm512_store_ps(&(c[M * 4]), vc4);
	_mm512_store_ps(&(c[M * 5]), vc5);
	_mm512_store_ps(&(c[M * 6]), vc6);
	_mm512_store_ps(&(c[M * 7]), vc7);
	_mm512_store_ps(&(c[M * 8]), vc8);
	_mm512_store_ps(&(c[M * 9]), vc9);
	_mm512_store_ps(&(c[M * 10]), vc10);
	_mm512_store_ps(&(c[M * 11]), vc11);
	_mm512_store_ps(&(c[M * 12]), vc12);
	_mm512_store_ps(&(c[M * 13]), vc13);
	_mm512_store_ps(&(c[M * 14]), vc14);
	_mm512_store_ps(&(c[M * 15]), vc15);
}
// 小矩阵的 乘法 ，使用SIMD计算
void matrix_multi_add_avx512_s(float* a, float* b, float* c, int M) {
	int i, j, k;
	int SM = M / 16;
	for (i = 0; i < SM; i++) {
		for (j = 0; j < SM; j++) {
			for (int k = 0; k < SM; k++) {
				matrix_multi_add_simd16X16X16(a + M * 16 * i + 16 * k,
				                              b + M * 16 * k + 16 * j,
				                              c + M * 16 * i + 16 * j, M);
			}
		}
	}
}
void partition_matrix_multi_avx512(float* a, float* b, float* c, int M,
                                   int N_SMALL) {
	int i, j, k;
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < N_SMALL; k++) {
				matrix_multi_add_avx512_s(a + M * M * N_SMALL * i + M * M * k,
				                          b + M * M * N_SMALL * k + M * M * j,
				                          c + M * M * N_SMALL * i + M * M * j, M);
			}
		}
	}
}
void partition_matrix_multi_avx512_omp(float* a, float* b, float* c, int M,
                                       int N_SMALL) {
	int i, j, k;
#pragma omp parallel for private(i, j, k)
	for (i = 0; i < N_SMALL; i++) {
		for (j = 0; j < N_SMALL; j++) {
			for (k = 0; k < N_SMALL; k++) {
				matrix_multi_add_avx512_s(a + M * M * N_SMALL * i + M * M * k,
				                          b + M * M * N_SMALL * k + M * M * j,
				                          c + M * M * N_SMALL * i + M * M * j, M);
			}
		}
	}
}
#endif
//----------------------------打印矩阵------------------------------------------
// print matrix 展示基本矩阵
void print_matrix(float* matrix, int N) {
	std::cout << "matrix is" << std::endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << std::format("{:.3f} ", matrix[i * N + j]);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
// show partition matrix 展示分块矩阵
void print_matrix_s(float* matrix, int M, int N_SMALL) {
	std::cout << "matrix is" << std::endl;
	for (int i = 0; i < N_SMALL; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < N_SMALL; k++) {
				float* start = matrix + N_SMALL * M * M * i + j * M + k * M * M;
				for (int l = 0; l < M; l++) {
					std::cout << std::format("{:.3f} ", start[l]);
				}
			}
			std::cout << std::endl << std::endl;
		}
	}
}
// print matrix with is partition 展示分块后的矩阵，按原顺序
void print_pratition_matrix(float* matrix, int M, int N_SMALL) {
	int N = M * N_SMALL;
	std::cout << "matrix is" << std::endl;
	for (int i = 0; i < N * 6 + N_SMALL * 2 + 2; i++) {
		std::cout << '-';
	}
	std::cout << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "| ";
		for (int j = 0; j < N; j++) {
			std::cout << std::format("{:.3f} ", matrix[i * N + j]);
			if (j % M == M - 1) {
				std::cout << "| ";
			}
		}
		std::cout << std::endl;
		if (i % M == M - 1) {
			for (int i = 0; i < N * 6 + N_SMALL * 2 + 1; i++) {
				std::cout << '-';
			}
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

//-------------------------检验矩阵乘法的结果---------------------------------
// check 不同乘法之间的结果差距
float check_res(float* a, float* b, int N, int M, bool transfport = true) {
	float* a_s = a;
	if (transfport) {
		a_s = new float[static_cast<unsigned long>(N * N)];
		transform_matrix_s(a, a_s, N, M);
	}
	float res{};
	float res_max{};
	auto tempa_s = a_s;
	for (int i = 0; i < N * N; i++) {
		res = std::max(res, std::abs(*tempa_s - *b));
		res_max = std::max(res_max, std::max(*tempa_s, *b));
		tempa_s++;
		b++;
	}
	std::cout << "max element in matrix one : " << res_max << std::endl;
	std::cout << "max element in matrix two : " << res_max + res << std::endl;
	if (transfport) {
		delete[] a_s;
	}
	return res;
}
float trace(float* a, int N) {
	float res{};
	for (int i = 0; i < N; i++) {
		res += a[i + i * N];
	}
	return res;
}

int main(int argc, char** argv) {
	int N, M;
	N = argc == 3 ? std::atoi(argv[1]) : 4096;
	M = argc == 3 ? std::atoi(argv[2]) : 64;
	Tick;
	auto* matrix1 =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* matrix2 =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* res =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	matrix_gen(matrix1, matrix2, N, 0.4);
	auto* matrix2_s =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* matrix1_s =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* res_split =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* res_simd =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* res_omp =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	auto* res_b =
	    (float*)operator new(N* N * sizeof(float), std::align_val_t(Align_Val));
	// std::cout << "-------baseline矩阵乘法------" << std::endl;
	// ReTick;
	// baseline_matrix_multi(matrix1, matrix2, res, N);
	// Tock;
	std::cout << "-----------矩阵分块----------" << std::endl;
	ReTick;
	transform_matrix_s(matrix1, matrix1_s, N, M);
	transform_matrix_s(matrix2, matrix2_s, N, M);
	Tock;
	std::cout << "-----------分块乘法----------" << std::endl;
	ReTick;
	partition_matrix_multi_omp(matrix1_s, matrix2_s, res_split, M, N / M);
	Tock;

	std::cout << "-------分块乘法 + simd------" << std::endl;
	ReTick;
#ifdef AVX512
	partition_matrix_multi512_avx(matrix1_s, matrix2_s, res_simd, M, N / M);
#else
	partition_matrix_multi_avx(matrix1_s, matrix2_s, res_simd, M, N / M);
#endif
	Tock;
	std::cout << "----分块乘法 + simd + omp----" << std::endl;
	ReTick;
#ifdef AVX512
	partition_matrix_multi512_avx_omp(matrix1_s, matrix2_s, res_simd, M, N / M);
#else
	partition_matrix_multi_avx_omp(matrix1_s, matrix2_s, res_omp, M, N / M);
#endif
Tock;
ReTick;
    std::cout << "----------矩阵恢复---------" << std::endl;
	transform_matrix_b(res_omp, res_b, N, M);
	Tock;
	std::cout << "------------Trace------------" << std::endl;
	std::cout << trace(res_b, N) << std::endl;
	// std::cout << trace(res, N) - trace(res_b, N) << std::endl;

	delete matrix1;
	delete matrix2;
	delete matrix1_s;
	delete matrix2_s;
	delete res;
	delete res_split;
	delete res_simd;
	delete res_omp;
	delete res_b;
}
//    g++ matrix.cpp -o matrix  -O3  -mfma -mavx2  -fopenmp -mavx512f