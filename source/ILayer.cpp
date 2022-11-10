#include "ILayer.h"
#include <iostream>

ILayer::ILayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
	: mIn()
	, mOut()
	, mWgt(nullptr)
	, mBias(nullptr)
	, mWgtGradSum(nullptr)
	, mBiasGradSum(nullptr)
	, mWgtVeloVec(nullptr)
	, mBiasVeloVec(nullptr)
	, mWgtDiff()
	, mBiasDiff()
	, mDelta()
	, mDeltaIn()
	, mDeltaOut()
	, meActFn(eActFn)
	, NUM_PAD(inLen == outLen ? (kernelLen - 1) / 2 : 0)
	, INPUT_LEN(inLen)
	, INPUT_PAD_LEN(INPUT_LEN + 2 * NUM_PAD)
	, INPUT_DEPTH(inDepth)
	, INPUT_SIZE(INPUT_PAD_LEN* INPUT_PAD_LEN* INPUT_DEPTH)
	, OUTPUT_LEN(outLen)
	, OUTPUT_DEPTH(outDepth)
	, OUTPUT_SIZE(OUTPUT_LEN* OUTPUT_LEN* OUTPUT_DEPTH)
	, KERNEL_LEN(kernelLen)
	, KERNEL_SIZE(KERNEL_LEN* KERNEL_LEN)
	, DELTA_SIZE(OUTPUT_SIZE)
	, DELTA_IN_SIZE(OUTPUT_SIZE)
	, DELTA_OUT_SIZE(INPUT_LEN* INPUT_LEN* INPUT_DEPTH)
	, WGT_SIZE(KERNEL_SIZE* INPUT_DEPTH* OUTPUT_DEPTH)
	, BIAS_SIZE(OUTPUT_DEPTH)
	, mActivate(nullptr)
	, mOutPad(0)
	, mB1T(0.9f)
	, mB2T(0.99f)
{
	// Alloc buffers
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		mIn.push_back(Alloc<data_t>(INPUT_SIZE));
		mWgtDiff.push_back(Alloc<data_t>(WGT_SIZE));
		mBiasDiff.push_back(Alloc<data_t>(BIAS_SIZE));
		mDelta.push_back(Alloc<data_t>(DELTA_SIZE));
		mDeltaOut.push_back(Alloc<data_t>(DELTA_OUT_SIZE));
		// Initialize to 0 , assert bit pattern 0x0000 means 0.0
		memset(mIn[i], 0, sizeof(data_t) * INPUT_SIZE);
		memset(mDelta[i], 0, sizeof(data_t) * DELTA_SIZE);
		memset(mDeltaOut[i], 0, sizeof(data_t) * DELTA_OUT_SIZE);
	}
	mWgt = Alloc<data_t>(WGT_SIZE);
	mBias = Alloc<data_t>(BIAS_SIZE);
	mWgtGradSum = Alloc<data_t>(WGT_SIZE);
	mBiasGradSum = Alloc<data_t>(BIAS_SIZE);
	mWgtVeloVec = Alloc<data_t>(WGT_SIZE);
	mBiasVeloVec = Alloc<data_t>(BIAS_SIZE);
	memset(mWgtGradSum, 0, sizeof(data_t) * WGT_SIZE);
	memset(mBiasGradSum, 0, sizeof(data_t) * BIAS_SIZE);
	memset(mWgtVeloVec, 0, sizeof(data_t) * WGT_SIZE);
	memset(mBiasVeloVec, 0, sizeof(data_t) * BIAS_SIZE);

	// Initialize Weigths
	// Glorot2010
	//		   sqrt(6)
	//	m = -------------		-m < Init val < m	
	//		sqrt(ni + no)
	const size_t FAN = KERNEL_SIZE * (INPUT_DEPTH + OUTPUT_DEPTH);
	const data_t INIT_MAX = sqrt(6.f / FAN);
	srand(time(NULL));
	const int RAND_HALF = (RAND_MAX + 1) / 2;
	for (size_t i = 0; i < WGT_SIZE; ++i)
	{
		int num = rand();
		data_t val = (data_t)(num - RAND_HALF) / RAND_HALF * INIT_MAX;
		mWgt[i] = val;
	}
	// Initialize Bias to 0
	memset(mBias, 0, sizeof(data_t) * BIAS_SIZE);

	// Initialize activation function
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
	case EActFn::IDEN:
		mActivate = [](data_t val) { return val; };
		break;
	default:
		Assert(false);
		break;
	}
}

ILayer::~ILayer()
{
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		Free(mIn[i]);
		Free(mWgtDiff[i]);
		Free(mBiasDiff[i]);
		Free(mDelta[i]);
		Free(mDeltaOut[i]);
	}
	Free(mWgt);
	Free(mBias);
	Free(mWgtGradSum);
	Free(mBiasGradSum);
	Free(mWgtVeloVec);
	Free(mBiasVeloVec);
}


void ILayer::InitBatch()
{
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		memset(mWgtDiff[i], 0, sizeof(data_t) * WGT_SIZE);
		memset(mBiasDiff[i], 0, sizeof(data_t) * BIAS_SIZE);
	}
}

void ILayer::Update(const size_t batchSize, const data_t learningRate)
{
	data_t* wgtDiffBuf = mWgtDiff[0];
	data_t* biasDiffBuf = mBiasDiff[0];
	// Sum diffs
	for (size_t i = 1; i < NUM_THREAD; ++i)
	{
		for (size_t j = 0; j < WGT_SIZE; ++j)
		{
			wgtDiffBuf[j] += mWgtDiff[i][j];
		}
	}
	for (size_t i = 1; i < NUM_THREAD; ++i)
	{
		for (size_t j = 0; j < BIAS_SIZE; ++j)
			biasDiffBuf[j] += mBiasDiff[i][j];
	}
	// Update parameters
	// Adaptive Moment
	const data_t EPS = 0.000001f;
	const data_t INV = 1.f / batchSize;
	const data_t B1 = 0.9f;
	const data_t B2 = 0.999f;
	mB1T = mB1T * B1;
	mB2T = mB2T * B2;

	data_t* mt = nullptr;
	data_t* vt = nullptr;
	data_t alpha = (0.001f * sqrt(batchSize) * learningRate);

	mt = mWgtGradSum;
	vt = mWgtVeloVec;
	for (size_t i = 0; i < WGT_SIZE; ++i)
	{
		data_t dw = wgtDiffBuf[i];
		dw = dw * INV;
		mt[i] = mt[i] * B1 + (1 - B1) * dw;
		vt[i] = vt[i] * B2 + (1 - B2) * dw * dw;
		data_t sw = (alpha * (mt[i] / (1 - mB1T)) / sqrt(vt[i] / (1 - mB2T) + EPS));
		mWgt[i] -= sw;
	}

	mt = mBiasGradSum;
	vt = mBiasVeloVec;
	for (size_t i = 0; i < BIAS_SIZE; ++i)
	{
		data_t dw = biasDiffBuf[i];
		dw = dw * INV;
		mt[i] = mt[i] * B1 + (1 - B1) * dw;
		vt[i] = vt[i] * B2 + (1 - B2) * dw * dw;
		data_t sw = (alpha * (mt[i] / (1 - mB1T)) / sqrt(vt[i] / (1 - mB2T) + EPS));
		mBias[i] -= sw;
	}
}

size_t ILayer::getInIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = (INPUT_PAD_LEN * y + x) * INPUT_DEPTH + d;
	size_t idx = INPUT_PAD_LEN * INPUT_PAD_LEN * d + INPUT_PAD_LEN * y + x;
	Assert(idx < INPUT_SIZE);
	return idx;
}
size_t ILayer::getOutIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = ((OUTPUT_LEN + 2 * mOutPad) * (y + mOutPad) + (mOutPad + x)) * OUTPUT_DEPTH + d;
	size_t idx = (OUTPUT_LEN + 2 * mOutPad) * (OUTPUT_LEN + 2 * mOutPad) * d + (OUTPUT_LEN + 2 * mOutPad) * (mOutPad + y) + (mOutPad + x);
	Assert(idx < (OUTPUT_LEN + 2 * mOutPad)* (OUTPUT_LEN + 2 * mOutPad)* OUTPUT_DEPTH);
	return idx;
}
size_t ILayer::getWgtIdx(size_t x, size_t y, size_t inD, size_t outD) const
{
	//size_t idx = ((KERNEL_LEN * y + x) * INPUT_DEPTH + inD) + KERNEL_LEN * KERNEL_LEN * INPUT_DEPTH * outD;
	size_t idx = KERNEL_LEN * KERNEL_LEN * INPUT_DEPTH * outD + KERNEL_LEN * KERNEL_LEN * inD + KERNEL_LEN * y + x;
	Assert(idx < WGT_SIZE);
	return idx;
}
size_t ILayer::getBiasIdx(size_t outD) const
{
	Assert(outD < OUTPUT_DEPTH);
	return outD;
}
size_t ILayer::getDeltaIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
	size_t idx = OUTPUT_LEN * OUTPUT_LEN * d + OUTPUT_LEN * y + x;
	Assert(idx < DELTA_SIZE);
	return idx;
}
size_t ILayer::getDInIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
	size_t idx = OUTPUT_LEN * OUTPUT_LEN * d + OUTPUT_LEN * y + x;
	Assert(idx < DELTA_IN_SIZE);
	return idx;
}
size_t ILayer::getDOutIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = (INPUT_LEN * y + x) * INPUT_DEPTH + d;
	size_t idx = INPUT_LEN * INPUT_LEN * d + INPUT_LEN * y + x;
	Assert(idx < DELTA_OUT_SIZE);
	return idx;
}
