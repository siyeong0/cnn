#pragma once
#include <functional>
#include <vector>
#include <thread>

#include <intrin.h>

#ifdef _DEBUG
#define Assert(E) if(!(E)){ __asm{ int 3 } }
#else
#define Assert(E) __assume(E)
#endif

inline float Max(float x, float y)
{
	return std::max(x, y);
}

inline float Min(float x, float y)
{
	return std::min(x, y);
}

#define AVX 1

#ifdef AVX
#define MM_ALIGNMENT 32
#define MM_BLOCK 8
// Types
#define MM_TYPE __m256
#define MM_TYPE_I __m256i
// Load/Store
#define MM_LOAD(X) _mm256_load_ps((X))
#define MM_STORE(X,Y) _mm256_store_ps((X),(Y))

#define MM_STORE_I(X,Y) _mm256_store_si256((X),(Y))
// Arithmetic operations
#define MM_ADD(X,Y) _mm256_add_ps((X),(Y))
#define MM_MUL(X,Y) _mm256_mul_ps((X),(Y))

// Bit operations
#define MM_AND(X,Y) _mm256_and_ps((X),(Y))
#define MM_XOR(X,Y) _mm256_xor_ps((X),(Y))

#define MM_AND_I(X,Y) _mm256_and_epi32((X),(Y))
#define MM_XOR_I(X,Y) _mm256_xor_epi32((X),(Y))
// Compare
#define MM_CMPGT(X,Y) _mm256_cmp_ps((X),(Y),_CMP_GT_OQ)
#define MM_CMPLT(X,Y) _mm256_cmp_ps((X),(Y),_CMP_LT_OQ)

// Set
#define MM_SETZERO() _mm256_setzero_ps()
#define MM_SET1(X) _mm256_set1_ps((X))

#define MM_SETZERO_I() _mm256_setzero_si256()
#define MM_SET1_I(X) _mm256_set1_epi32(X)
// Horizontal sum
inline float MMHorizSum(__m256 V)
{
	V = _mm256_hadd_ps(V, V);
	V = _mm256_hadd_ps(V, V);
	float x = _mm_cvtss_f32(_mm256_extractf128_ps(V, 1));
	float y = _mm256_cvtss_f32(V);
	return x + y;
}
#define MM_HORIZ_SUM(X) MMHorizSum(X)
// Casting
#define MM_CAST_F2I(X) _mm256_castps_si256(X)

#endif

template <typename T>
inline T* Alloc(size_t size)
{
#ifndef AVX
	return static_cast<T*>(malloc(sizeof(T) * size));
#else
	return static_cast<T*>(_aligned_malloc(sizeof(T) * size, MM_ALIGNMENT));
#endif
}

inline void Free(void* ptr)
{
#ifndef AVX
	free(ptr);
#else
	_aligned_free(ptr);
#endif
}


using data_t = float;

const unsigned int NUM_THREAD = std::thread::hardware_concurrency();

namespace cnn
{
	enum class EActFn
	{
		TANH,
		RELU,
		SOFTMAX,
		SIGMOID,
		IDEN,
	};

	enum class ENet { END = 0 };

	class Network;

	class ILayer
	{
	public:
		friend Network& operator>>(Network& net, ENet e);
	public:
		ILayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn);
		~ILayer();
		ILayer(const ILayer&) = delete;
		ILayer& operator=(const ILayer&) = delete;

		virtual void Forward(size_t threadIdx) = 0;
		virtual void BackProp(size_t threadIdx) = 0;

		void InitBatch();

		void Update(const size_t batchSize, const data_t learningRate);
	protected:
		inline size_t getInIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = (INPUT_PAD_LEN * y + x) * INPUT_DEPTH + d;
			Assert(idx < INPUT_SIZE);
			return idx;
		}
		inline size_t getOutIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = ((OUTPUT_LEN + 2 * mOutPad) * (y + mOutPad) + (mOutPad + x)) * OUTPUT_DEPTH + d;
			Assert(idx < (OUTPUT_LEN + 2 * mOutPad)* (OUTPUT_LEN + 2 * mOutPad)* OUTPUT_DEPTH);
			return idx;
		}
		inline size_t getWgtIdx(size_t x, size_t y, size_t inD, size_t outD) const
		{
			size_t idx = (KERNEL_LEN * KERNEL_LEN * inD + KERNEL_LEN * y + x) * OUTPUT_DEPTH + outD;
			Assert(idx < WGT_SIZE);
			return idx;
		}
		inline size_t getBiasIdx(size_t outD) const
		{
			Assert(outD < OUTPUT_DEPTH);
			return outD;
		}
		inline size_t getDeltaIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
			Assert(idx < DELTA_SIZE);
			return idx;
		}
		inline size_t getDInIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
			Assert(idx < DELTA_IN_SIZE);
			return idx;
		}
		inline size_t getDOutIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = (INPUT_LEN * y + x) * INPUT_DEPTH + d;
			Assert(idx < DELTA_OUT_SIZE);
			return idx;
		}
	protected:	// vector elements are buffers allocated to threads
		std::vector<data_t*> mIn;
		std::vector<data_t*> mOut;
		data_t* mWgt;
		data_t* mBias;
		// Buffers for back propagation
		std::vector<data_t*> mWgtDiff;
		std::vector<data_t*> mBiasDiff;
		std::vector<data_t*> mDelta;
		std::vector<data_t*> mDeltaIn;
		std::vector<data_t*> mDeltaOut;
		// Buffers for Adam
		data_t* mWgtGradSum;
		data_t* mBiasGradSum;
		data_t* mWgtVeloVec;
		data_t* mBiasVeloVec;
		// Activation func
		EActFn meActFn;
		std::function<data_t(data_t)> mActivate;
	protected:	// Constants for buffer sizes
		const size_t NUM_PAD;
		const size_t INPUT_LEN;
		const size_t INPUT_PAD_LEN;
		const size_t INPUT_DEPTH;
		const size_t INPUT_SIZE;
		const size_t OUTPUT_LEN;
		const size_t OUTPUT_DEPTH;
		const size_t OUTPUT_SIZE;
		const size_t KERNEL_LEN;
		const size_t KERNEL_SIZE;
		const size_t DELTA_SIZE;
		const size_t DELTA_IN_SIZE;
		const size_t DELTA_OUT_SIZE;
		const size_t WGT_SIZE;
		const size_t BIAS_SIZE;
		size_t mOutPad;
	private:	// Constatns for Adam
		data_t mB1T;
		data_t mB2T;
	};
}