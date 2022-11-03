#include "PoolLayer.h"
#include <iostream>
PoolLayer::PoolLayer(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn)
	: ILayer(kernelSize, inLen, depth, inLen / kernelSize, depth, eActFn)
	, mMaxIdxBuf(nullptr)
{
	mMaxIdxBuf = Alloc<size_t>(OUTPUT_SIZE);
}

PoolLayer::~PoolLayer()
{
	Free(mMaxIdxBuf);
}

void PoolLayer::Forward()
{
	for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				// 2x2 kernel only
				Assert(KERNEL_LEN == 2);
				data_t v0 = mIn[getInIdx(outX * 2 + 0, outY * 2 + 0, outD)];
				data_t v1 = mIn[getInIdx(outX * 2 + 1, outY * 2 + 0, outD)];
				data_t v2 = mIn[getInIdx(outX * 2 + 0, outY * 2 + 1, outD)];
				data_t v3 = mIn[getInIdx(outX * 2 + 1, outY * 2 + 1, outD)];
				data_t maxVal = v0;
				size_t maxIdx = 0x0;
				if (v1 > maxVal) { maxVal = v1; maxIdx = 0x1; }
				if (v2 > maxVal) { maxVal = v2; maxIdx = 0x2; }
				if (v3 > maxVal) { maxVal = v3; maxIdx = 0x3; }
				Assert(maxIdx < 4);
				mOut[getOutIdx(outX, outY, outD)] = mActivate(maxVal);
				mMaxIdxBuf[getMIBufIdx(outX, outY, outD)] = maxIdx;
			}
		}
	}
}

void PoolLayer::BackProp()
{
	memset(mDeltaOut, 0, sizeof(data_t) * DELTA_OUT_SIZE);
	for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				data_t outVal = mOut[getOutIdx(outX, outY, outD)];
				size_t maxIdx = mMaxIdxBuf[getMIBufIdx(outX, outY, outD)];
				size_t inX = outX * 2 + (maxIdx & 1);
				size_t inY = outY * 2 + (maxIdx >> 1);
				data_t deltaIn = mDeltaIn[getDInIdx(outX, outY, outD)];
				Assert(meActFn == EActFn::RELU);
				data_t deriv = outVal > 0.f ? 1 : 0;
				mDeltaOut[getDOutIdx(inX, inY, outD)] = deriv * deltaIn;
			}
		}
	}
}

size_t PoolLayer::getMIBufIdx(size_t x, size_t y, size_t d) const
{
	size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
	Assert(idx < OUTPUT_SIZE);
	return idx;
}