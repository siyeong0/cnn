#pragma once
#include <iostream>
#include <cassert>
#include "DWConv.h"

namespace cnn
{
	DwConv::DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn)
		: ILayer(kernelLen, inLen, inDepth, outLen, inDepth, eActFn)
	{

	}

	DwConv::~DwConv()
	{

	}

	void DwConv::Forward(size_t threadIdx)
	{
		data_t* inBuf = mIn[threadIdx];
		data_t* outBuf = mOut[threadIdx];
		data_t* wgtBuf = mWgt;
		data_t* biasBuf = mBias;

		for (size_t depth = 0; depth < OUTPUT_DEPTH; ++depth)
		{
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					data_t sum = 0.f;
					for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
					{
						for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
						{
							data_t in = inBuf[getInIdx(outX + kX, outY + kY, depth)];
							data_t wgt = wgtBuf[getWgtIdx(kX, kY, depth, depth)];
							sum += in * wgt;
						}
					}
					sum += biasBuf[getBiasIdx(depth)];
					outBuf[getOutIdx(outX, outY, depth)] = mActivate(sum);
				}
			}
		}
	}

	void DwConv::BackProp(size_t threadIdx)
	{
		data_t* inBuf = mIn[threadIdx];
		data_t* outBuf = mOut[threadIdx];
		data_t* wgtBuf = mWgt;
		data_t* delInBuf = mDeltaIn[threadIdx];
		data_t* delBuf = mDelta[threadIdx];
		data_t* delOutBuf = mDeltaOut[threadIdx];
		data_t* wgtDiffBuf = mWgtDiff[threadIdx];
		data_t* biasDiffBuf = mBiasDiff[threadIdx];

		// Get global delta
		for (size_t depth = 0; depth < OUTPUT_DEPTH; ++depth)
		{
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					data_t deltaIn = delInBuf[getDInIdx(outX, outY, depth)];
					data_t out = outBuf[getOutIdx(outX, outY, depth)];
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
					delBuf[getDeltaIdx(outX, outY, depth)] = deltaIn * deriv;
				}
			}
		}
		// Get Weights' gradient
		for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
		{
			for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
			{
				for (size_t depth = 0; depth < INPUT_DEPTH; ++depth)
				{
					data_t sum = 0.f;
					for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
					{
						for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
						{
							data_t delta = delBuf[getDeltaIdx(outX, outY, depth)];
							data_t valIn = inBuf[getInIdx(outX + kX, outY + kY, depth)];
							sum += delta * valIn;
						}
					}
					wgtDiffBuf[getWgtIdx(kX, kY, depth, depth)] += sum;
				}
			}
		}
		// Get Biases' gradient
		for (size_t depth = 0; depth < OUTPUT_DEPTH; ++depth)
		{
			data_t sum = 0.f;
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					sum += delBuf[getDeltaIdx(outX, outY, depth)];
				}
			}
			biasDiffBuf[depth] += sum;
		}

		// Get out gradient : prev layer's input gradient
		const int IPAD = static_cast<int>(NUM_PAD);
		for (size_t depth = 0; depth < OUTPUT_DEPTH; ++depth)
		{
			for (int inY = 0; inY < INPUT_LEN; ++inY)
			{
				for (int inX = 0; inX < INPUT_LEN; ++inX)
				{
					const int BX = Max(IPAD - inX, 0);
					const int BY = Max(IPAD - inY, 0);
					const int EX = Min(INPUT_LEN + IPAD - inX, KERNEL_LEN);
					const int EY = Min(INPUT_LEN + IPAD - inY, KERNEL_LEN);
					data_t sum = 0.f;
					for (size_t kY = BY; kY < EY; ++kY)
					{
						for (size_t kX = BX; kX < EX; ++kX)
						{
							size_t rkx = KERNEL_LEN - 1 - kX;
							size_t rky = KERNEL_LEN - 1 - kY;
							size_t outX = inX - IPAD + kX;
							size_t outY = inY - IPAD + kY;
							data_t delta = delBuf[getDeltaIdx(outX, outY, depth)];
							data_t wgt = wgtBuf[getWgtIdx(rkx, rky, depth, depth)];
							sum += delta * wgt;
						}
					}
					delOutBuf[getDOutIdx(inX, inY, depth)] = sum;
				}
			}
		}
	}
}
