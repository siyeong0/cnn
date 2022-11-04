#include "PoolLayer.h"
#include <iostream>
PoolLayer::PoolLayer(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn)
	: ILayer(kernelSize, inLen, depth, inLen / kernelSize, depth, eActFn)
	, mMaxIdxBuf()
{
	for (size_t i = 0; i < NUM_THREAD; i++)
	{
		mMaxIdxBuf.push_back(Alloc<size_t>(OUTPUT_SIZE));
	}
}

PoolLayer::~PoolLayer()
{
	for (size_t i = 0; i < NUM_THREAD; i++)
	{
		Free(mMaxIdxBuf[i]);
	}
}

void PoolLayer::Forward(size_t threadIdx)
{
	data_t* inBuf = mIn[threadIdx];
	data_t* outBuf = mOut[threadIdx];
	size_t* maxIdxBuf = mMaxIdxBuf[threadIdx];

	for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				// 2x2 kernel only
				Assert(KERNEL_LEN == 2);
				data_t v0 = inBuf[getInIdx(outX * 2 + 0, outY * 2 + 0, outD)];
				data_t v1 = inBuf[getInIdx(outX * 2 + 1, outY * 2 + 0, outD)];
				data_t v2 = inBuf[getInIdx(outX * 2 + 0, outY * 2 + 1, outD)];
				data_t v3 = inBuf[getInIdx(outX * 2 + 1, outY * 2 + 1, outD)];
				data_t maxVal = v0;
				size_t maxIdx = 0x0;
				if (v1 > maxVal) { maxVal = v1; maxIdx = 0x1; }
				if (v2 > maxVal) { maxVal = v2; maxIdx = 0x2; }
				if (v3 > maxVal) { maxVal = v3; maxIdx = 0x3; }
				Assert(maxIdx < 4);
				outBuf[getOutIdx(outX, outY, outD)] = mActivate(maxVal);
				maxIdxBuf[getMIBufIdx(outX, outY, outD)] = maxIdx;
			}
		}
	}
}

void PoolLayer::BackProp(size_t threadIdx)
{
	data_t* inBuf = mIn[threadIdx];
	data_t* outBuf = mOut[threadIdx];
	data_t* delInBuf = mDeltaIn[threadIdx];
	data_t* delOutBuf = mDeltaOut[threadIdx];
	size_t* maxIdxBuf = mMaxIdxBuf[threadIdx];

	memset(delOutBuf, 0, sizeof(data_t) * DELTA_OUT_SIZE);
	for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				data_t outVal = outBuf[getOutIdx(outX, outY, outD)];
				size_t maxIdx = maxIdxBuf[getMIBufIdx(outX, outY, outD)];
				size_t inX = outX * 2 + (maxIdx & 1);
				size_t inY = outY * 2 + (maxIdx >> 1);
				data_t deltaIn = delInBuf[getDInIdx(outX, outY, outD)];
				Assert(meActFn == EActFn::RELU);
				data_t deriv = outVal > 0.f ? 1 : 0;
				delOutBuf[getDOutIdx(inX, inY, outD)] = deriv * deltaIn;
			}
		}
	}
}

size_t PoolLayer::getMIBufIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
	size_t idx = OUTPUT_LEN * OUTPUT_LEN * d + OUTPUT_LEN * y + x;
	Assert(idx < OUTPUT_SIZE);
	return idx;
}