#pragma once
#include <iostream>
#include <cassert>
#include "Conv.h"

namespace cnn
{
	Conv::Conv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
		: ILayer(kernelLen, inLen, inDepth, outLen, outDepth, eActFn)
	{

	}

	Conv::~Conv()
	{

	}

	void Conv::Forward(size_t threadIdx)
	{
		data_t* inBuf = mIn[threadIdx];
		data_t* outBuf = mOut[threadIdx];
		data_t* wgtBuf = mWgt;
		data_t* biasBuf = mBias;
		if (mbUseAvx == false)
		{
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
				{
					for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
					{
						data_t sum = 0.f;
						for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
						{
							for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
							{
								for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
								{
									data_t in = inBuf[getInIdx(outX + kX, outY + kY, inD)];
									data_t wgt = wgtBuf[getWgtIdx(kX, kY, inD, outD)];
									sum += in * wgt;
								}
							}
						}
						sum += biasBuf[getBiasIdx(outD)];
						outBuf[getOutIdx(outX, outY, outD)] = mActivate(sum);
					}
				}
			}
		}
		else
		{
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
					{
						MM_TYPE mmSum = MM_SETZERO();
						for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
						{
							for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
							{
								for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
								{
									data_t in = inBuf[getInIdx(outX + kX, outY + kY, inD)];
									MM_TYPE mmIn = MM_SET1(in);
									MM_TYPE mmWgt = MM_LOAD(&wgtBuf[getWgtIdx(kX, kY, inD, outD)]);

									MM_TYPE mmMul = MM_MUL(mmIn, mmWgt);
									mmSum = MM_ADD(mmSum, mmMul);
								}
							}
						}
						MM_TYPE mmBias = MM_LOAD(&biasBuf[getBiasIdx(outD)]);
						mmSum = MM_ADD(mmSum, mmBias);

						float* dest = &outBuf[getOutIdx(outX, outY, outD)];
						MM_STORE(dest, mmSum);
						for (size_t i = 0; i < MM_BLOCK; ++i)
						{
							dest[i] = mActivate(dest[i]);
						}
					}
				}
			}
		}
	}

	void Conv::BackProp(size_t threadIdx)
	{
		data_t* inBuf = mIn[threadIdx];
		data_t* outBuf = mOut[threadIdx];
		data_t* wgtBuf = mWgt;
		data_t* delInBuf = mDeltaIn[threadIdx];
		data_t* delBuf = mDelta[threadIdx];
		data_t* delOutBuf = mDeltaOut[threadIdx];
		data_t* wgtDiffBuf = mWgtDiff[threadIdx];
		data_t* biasDiffBuf = mBiasDiff[threadIdx];
		if (mbUseAvx == false)
		{
			// Get global delta
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
				{
					for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
					{
						data_t deltaIn = delInBuf[getDInIdx(outX, outY, outD)];
						data_t out = outBuf[getOutIdx(outX, outY, outD)];
						data_t deriv = 0.f;
						switch (meActFn)
						{
						case EActFn::TANH:
							deriv = 1 - out * out;
							break;
						case EActFn::RELU:
							deriv = out > 0.f ? 1.f : 0.f;
							break;
						default:
							Assert(false);
							break;
						}
						delBuf[getDeltaIdx(outX, outY, outD)] = deltaIn * deriv;
					}
				}
			}

			// Get Weights' gradient
			for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
			{
				for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
				{
					for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
					{
						for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
						{
							data_t sum = 0.f;
							for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
							{
								for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
								{
									data_t delta = delBuf[getDeltaIdx(outX, outY, outD)];
									data_t valIn = inBuf[getInIdx(outX + kX, outY + kY, inD)];
									sum += delta * valIn;
								}
							}
							wgtDiffBuf[getWgtIdx(kX, kY, inD, outD)] += sum;
						}
					}
				}
			}
			// Get Biases' gradient
			for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
			{
				data_t sum = 0.f;
				for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
				{
					for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
					{
						sum += delBuf[getDeltaIdx(outX, outY, outD)];
					}
				}
				biasDiffBuf[outD] += sum;
			}

			// Get out gradient : prev layer's input gradient
			const int IPAD = static_cast<int>(NUM_PAD);
			for (int inY = 0; inY < INPUT_LEN; ++inY)
			{
				for (int inX = 0; inX < INPUT_LEN; ++inX)
				{
					const int BX = Max(IPAD - inX, 0);
					const int BY = Max(IPAD - inY, 0);
					const int EX = Min(OUTPUT_LEN, Min(INPUT_LEN + IPAD - inX, KERNEL_LEN));
					const int EY = Min(OUTPUT_LEN, Min(INPUT_LEN + IPAD - inY, KERNEL_LEN));
					for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
					{
						data_t sum = 0.f;
						for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
						{
							for (size_t kY = BY; kY < EY; ++kY)
							{
								for (size_t kX = BX; kX < EX; ++kX)
								{
									size_t rkx = KERNEL_LEN - 1 - kX;
									size_t rky = KERNEL_LEN - 1 - kY;
									size_t outX = Min(OUTPUT_LEN - 1, inX - IPAD + kX);
									size_t outY = Min(OUTPUT_LEN - 1, inY - IPAD + kY);
									data_t delta = delBuf[getDeltaIdx(outX, outY, outD)];
									data_t wgt = wgtBuf[getWgtIdx(rkx, rky, inD, outD)];
									sum += delta * wgt;
								}
							}
						}
						delOutBuf[getDOutIdx(inX, inY, inD)] = sum;
					}
				}
			}
		}
		else
		{
			// Get global delta
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
					{
						// Get delta in
						MM_TYPE mmDIn = MM_LOAD(&delInBuf[getDInIdx(outX, outY, outD)]);
						// Get deriv
						MM_TYPE mmOut = MM_LOAD(&outBuf[getOutIdx(outX, outY, outD)]);
						MM_TYPE mmZero = MM_SETZERO();
						MM_TYPE mmOne = MM_SET1(1.f);
						Assert(meActFn == EActFn::RELU);
						MM_TYPE mmAnd = MM_CMPGT(mmOut, mmZero);
						MM_TYPE mmDeriv = MM_AND(mmOne, mmAnd);
						// Multyply
						MM_TYPE mmMul = MM_MUL(mmDIn, mmDeriv);
						// Store result
						float* dest = &delBuf[getDeltaIdx(outX, outY, outD)];
						MM_STORE(dest, mmMul);
					}
				}
			}

			// Get Weights' gradient
			for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
			{
				for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
				{
					for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
					{
						for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
						{
							// Get curr diff sum
							MM_TYPE mmSum = MM_SETZERO();
							for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
							{
								for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
								{
									MM_TYPE mmDelta = MM_LOAD(&delBuf[getDeltaIdx(outX, outY, outD)]);
									MM_TYPE mmXIn = MM_SET1(inBuf[getInIdx(outX + kX, outY + kY, inD)]);

									MM_TYPE mmMul = MM_MUL(mmDelta, mmXIn);
									mmSum = MM_ADD(mmSum, mmMul);
								}
							}
							// Add curr diff sum
							float* dest = &wgtDiffBuf[getWgtIdx(kX, kY, inD, outD)];
							MM_TYPE mmDest = MM_LOAD(dest);
							mmDest = MM_ADD(mmDest, mmSum);
							MM_STORE(dest, mmDest);
						}
					}
				}
			}
			// Get Biases' gradient
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
					{
						// Get curr diff sum
						MM_TYPE mmSum = MM_LOAD(&delBuf[getDeltaIdx(outX, outY, outD)]);
						// Add curr diff sum
						float* dest = &biasDiffBuf[outD];
						MM_TYPE mmDest = MM_LOAD(dest);
						mmDest = MM_ADD(mmDest, mmSum);
						MM_STORE(dest, mmDest);
					}
				}
			}

			// Get out gradient : prev layer's input gradient
			const int IPAD = static_cast<int>(NUM_PAD);
#pragma warning(push)
#pragma warning(disable : 4018)
			for (int inY = 0; inY < INPUT_LEN; ++inY)
			{
				for (int inX = 0; inX < INPUT_LEN; ++inX)
				{
					const int BX = Max(IPAD - inX, 0);
					const int BY = Max(IPAD - inY, 0);
					const int EX = Min(OUTPUT_LEN, Min(INPUT_LEN + IPAD - inX, KERNEL_LEN));
					const int EY = Min(OUTPUT_LEN, Min(INPUT_LEN + IPAD - inY, KERNEL_LEN));
					for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
					{
						MM_TYPE mmSum = MM_SETZERO();
						for (size_t outD = 0; outD < OUTPUT_DEPTH; outD += MM_BLOCK)
						{
							for (size_t kY = BY; kY < EY; ++kY)
							{
								for (size_t kX = BX; kX < EX; ++kX)
								{
									size_t rkx = KERNEL_LEN - 1 - kX;
									size_t rky = KERNEL_LEN - 1 - kY;
									size_t outX = Min(OUTPUT_LEN - 1,inX - IPAD + kX);
									size_t outY = Min(OUTPUT_LEN - 1, inY - IPAD + kY);
									MM_TYPE mmDelta = MM_LOAD(&delBuf[getDeltaIdx(outX, outY, outD)]);
									MM_TYPE mmWgt = MM_LOAD(&wgtBuf[getWgtIdx(rkx, rky, inD, outD)]);
									MM_TYPE mmMul = MM_MUL(mmDelta, mmWgt);

									mmSum = MM_ADD(mmSum, mmMul);
								}
							}
						}
						data_t sum = MM_HORIZ_SUM(mmSum);
						delOutBuf[getDOutIdx(inX, inY, inD)] = sum;
					}
				}
			}
#pragma warning(pop)
		}
	}
}