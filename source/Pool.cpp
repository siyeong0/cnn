#include "Pool.h"
#include <iostream>

namespace cnn
{
	Pool::Pool(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn)
		: ILayer(kernelSize, inLen, depth, inLen / kernelSize, depth, eActFn)
		, mMaxIdxBuf()
	{
		for (size_t i = 0; i < NUM_THREAD; i++)
		{
			mMaxIdxBuf.push_back(Alloc<size_t>(OUTPUT_SIZE));
		}
	}

	Pool::~Pool()
	{
		for (size_t i = 0; i < NUM_THREAD; i++)
		{
			Free(mMaxIdxBuf[i]);
		}
	}

	void Pool::Forward(size_t threadIdx)
	{
		data_t* inBuf = mIn[threadIdx];
		data_t* outBuf = mOut[threadIdx];
		size_t* maxIdxBuf = mMaxIdxBuf[threadIdx];
		if (mbUseAvx == false)
		{
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
						// Get output
						outBuf[getOutIdx(outX, outY, outD)] = mActivate(maxVal);
						// Store max val's idx
						maxIdxBuf[getMIBufIdx(outX, outY, outD)] = maxIdx;
					}
				}
			}
		}
		else
		{
			for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
			{
				for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
				{
					for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
					{
						MM_TYPE mmMax = MM_LOAD(&inBuf[getInIdx(outX * KERNEL_LEN, outY * KERNEL_LEN, outD)]);
						MM_TYPE_I mmMIdx = MM_SETZERO_I();
						for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
						{
							for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
							{
								MM_TYPE mmIn = MM_LOAD(&inBuf[getInIdx(outX * KERNEL_LEN + kX, outY * KERNEL_LEN + kY, outD)]);
								MM_TYPE_I mmIdx = MM_SET1_I(kY * KERNEL_LEN + kX);

								MM_TYPE mmCmpMask = MM_CMPLT(mmMax, mmIn);
								// Get max value
								MM_TYPE mmXorMask = MM_XOR(mmMax, mmIn);
								mmXorMask = MM_AND(mmXorMask, mmCmpMask);
								mmMax = MM_XOR(mmMax, mmXorMask);
								// Get max index
								MM_TYPE_I mmXorMaskIdx = MM_XOR_I(mmMIdx, mmIdx);
								mmXorMaskIdx = MM_AND_I(mmXorMaskIdx, MM_CAST_F2I(mmCmpMask));
								mmMIdx = MM_XOR_I(mmMIdx, mmXorMaskIdx);
							}
						}
						// Get output
						float* dest = &outBuf[getOutIdx(outX, outY, outD)];
						MM_STORE(dest, mmMax);
						for (size_t i = 0; i < MM_BLOCK; ++i)
						{
							dest[i] = mActivate(dest[i]);
						}
						//Store max val's idx
						MM_STORE_I((__m256i*)(&maxIdxBuf[getMIBufIdx(outX, outY, outD)]), mmMIdx);
					}
				}
			}
		}
	}

	void Pool::BackProp(size_t threadIdx)
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
					data_t deriv = outVal > 0.f ? 1.f : 0.f;
					delOutBuf[getDOutIdx(inX, inY, outD)] = deriv * deltaIn;
				}
			}
		}
	}
}