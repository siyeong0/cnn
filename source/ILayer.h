#pragma once
#include <cstdlib>
#include <functional>

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

enum class EActFn
{
	TANH,
	RELU,
	SOFTMAX,
	SIGMOID,
};

enum class ENet { END = 0 };

class Network;

class ILayer
{
public:
	friend Network& operator>>(Network& net, ENet e);
public:
	ILayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen,size_t outDepth, EActFn eActFn);
	~ILayer();
	ILayer(const ILayer&) = delete;
	ILayer& operator=(const ILayer&) = delete;

	virtual void Forward() = 0;
	virtual void BackProp() = 0;

	void InitEpoch();
	void InitBatch();

	void Update(const size_t batchSize, const data_t learningRate);
public:
	data_t* mIn;
	data_t* mOut;
	data_t* mWgt;
	data_t* mBias;

	data_t* mWgtDiff;
	data_t* mBiasDiff;
	data_t* mDelta;
	data_t* mDeltaIn;
	data_t* mDeltaOut;

	EActFn meActFn;
	std::function<data_t(data_t)> mActivate;
public:
	const size_t NUM_PAD;
	const size_t INPUT_LEN;
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
};