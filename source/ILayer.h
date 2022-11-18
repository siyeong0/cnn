#pragma once
#include <functional>
#include <vector>
#include <thread>

#ifdef _DEBUG
#define Assert(E) if(!(E)){ __asm{ int 3 } }
#else
#define Assert(E) __assume(E)
#endif

template <typename T>
inline T* Alloc(size_t size)
{
	return static_cast<T*>(malloc(sizeof(T) * size));
}

inline void Free(void* ptr)
{
	free(ptr);
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
		size_t getOutIdx(size_t x, size_t y, size_t d) const;
		size_t getInIdx(size_t x, size_t y, size_t d) const;
		size_t getWgtIdx(size_t x, size_t y, size_t inD, size_t outD) const;
		size_t getBiasIdx(size_t outD) const;
		size_t getDeltaIdx(size_t x, size_t y, size_t d) const;
		size_t getDInIdx(size_t x, size_t y, size_t d) const;
		size_t getDOutIdx(size_t x, size_t y, size_t d) const;
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