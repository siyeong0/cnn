#include "ILayer.h"
#include <iostream>

static data_t gaussianRandom(data_t average, data_t stdev)
{
	data_t v1;
	data_t v2;
	data_t s;
	do
	{
		v1 = 2 * ((data_t)rand() / RAND_MAX) - 1;
		v2 = 2 * ((data_t)rand() / RAND_MAX) - 1;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1 || s == 0);

	s = sqrt((-2 * log(s)) / s);
	data_t temp = v1 * s;
	temp = (stdev * temp) + average;

	return temp;
}

ILayer::ILayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
	: mIn(nullptr)
	, mOut(nullptr)
	, mWgt(nullptr)
	, mBias(nullptr)
	, mWgtDiff(nullptr)
	, mBiasDiff(nullptr)
	, mDelta(nullptr)
	, mDeltaIn(nullptr)
	, mDeltaOut(nullptr)
	, meActFn(eActFn)
	, NUM_PAD(inLen == outLen ? (kernelLen - 1) / 2 : 0)
	, INPUT_LEN(inLen + 2 * NUM_PAD)
	, INPUT_DEPTH(inDepth)
	, INPUT_SIZE(INPUT_LEN* INPUT_LEN* INPUT_DEPTH)
	, OUTPUT_LEN(outLen)
	, OUTPUT_DEPTH(outDepth)
	, OUTPUT_SIZE(OUTPUT_LEN* OUTPUT_LEN* OUTPUT_DEPTH)
	, KERNEL_LEN(kernelLen)
	, KERNEL_SIZE(KERNEL_LEN* KERNEL_LEN)
	, DELTA_SIZE(OUTPUT_SIZE)
	, DELTA_IN_SIZE(OUTPUT_SIZE)
	, DELTA_OUT_SIZE(INPUT_SIZE)
	, WGT_SIZE(KERNEL_SIZE * INPUT_DEPTH* OUTPUT_DEPTH)
	, BIAS_SIZE(OUTPUT_DEPTH)
	, mActivate(nullptr)
{
	mIn = Alloc<data_t>(INPUT_SIZE);
	mWgt = Alloc<data_t>(WGT_SIZE);
	mWgtDiff = Alloc<data_t>(WGT_SIZE);
	mBias = Alloc<data_t>(BIAS_SIZE);
	mBiasDiff = Alloc<data_t>(BIAS_SIZE);
	mDelta = Alloc<data_t>(DELTA_SIZE);
	mDeltaOut = Alloc<data_t>(DELTA_OUT_SIZE);

	srand(time(NULL));
	for (size_t i = 0; i < WGT_SIZE; ++i)
	{
		mWgt[i] = gaussianRandom(0.f, 0.1f);
	}
	for (size_t i = 0; i < BIAS_SIZE; ++i)
	{
		mBias[i] = gaussianRandom(0.f, 0.00001f);
	}

	switch (eActFn)
	{
	case EActFn::TANH:
		mActivate = [](data_t val) { return tanh(val); };
		break;
	case EActFn::RELU:
		mActivate = [](data_t val) { return val > 0.f ? val : 0.f; };
		break;
	case EActFn::SOFTMAX:
		Assert(false);
		break;
	case EActFn::SIGMOID:
		mActivate = [](data_t val) { return 1.f / (1.f + pow(2.7f, -val)); };
		break;
	default:
		Assert(false);
		break;
	}
}

ILayer::~ILayer()
{
	Free(mIn);
	Free(mWgt);
	Free(mBias);
	Free(mWgtDiff);
	Free(mBiasDiff);
	Free(mDelta);
	Free(mDeltaOut);
}

void ILayer::InitEpoch()
{

}

void ILayer::InitBatch()
{
	memset(mWgtDiff, 0, sizeof(data_t) * WGT_SIZE);
	memset(mBiasDiff, 0, sizeof(data_t) * BIAS_SIZE);
}

void ILayer::Update(const size_t batchSize, const data_t learningRate)
{
	data_t alpha = 0.001f * sqrt((data_t)batchSize) * learningRate;
	for (size_t i = 0; i < WGT_SIZE; ++i)
	{
		data_t dw = mWgtDiff[i] / batchSize;
		data_t v1 = 0.1f * dw;
		data_t v2 = 0.001f * dw * dw;
		data_t sw = alpha * (v1 / (1 - 0.1f)) / sqrt(v2 / (1 - 0.001f) + 0.000001f);
		mWgt[i] -= sw;
	}

	for (size_t i = 0; i < BIAS_SIZE; ++i)
	{
		data_t dw = mBiasDiff[i] / batchSize;
		data_t v1 = 0.1f * dw;
		data_t v2 = 0.001f * dw * dw;
		data_t sw = alpha * (v1 / (1 - 0.1f)) / sqrt(v2 / (1 - 0.001f) + 0.000001f);
		mWgt[i] -= sw;
		mBias[i] -= sw;
	}
}