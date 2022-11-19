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
					sum += mBias[getBiasIdx(depth)];
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
		memset(delOutBuf, 0, sizeof(data_t) * DELTA_OUT_SIZE);
		for (size_t depth = 0; depth < OUTPUT_DEPTH; ++depth)
		{
			for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
			{
				for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
				{
					data_t delta = delBuf[getDeltaIdx(outX, outY, depth)];
					for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
					{
						for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
						{
							size_t inY = outY + kY - NUM_PAD;
							size_t inX = outX + kX - NUM_PAD;
							if (inX >= INPUT_LEN || inY >= INPUT_LEN)
							{
								continue;
							}
							data_t wgt = wgtBuf[getWgtIdx(kX, kY, depth, depth)];
							delOutBuf[getDOutIdx(inX, inY, depth)] += delta * wgt;
						}
					}
				}
			}
		}
	}
}
