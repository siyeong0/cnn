#pragma once
#include <iostream>
#include <cassert>
#include "ConvLayer.h"

ConvLayer::ConvLayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
	: ILayer(kernelLen, inLen, inDepth, outLen, outDepth, eActFn)
{

}

ConvLayer::~ConvLayer()
{

}

void ConvLayer::Forward(size_t threadIdx)
{
	data_t* inBuf = mIn[threadIdx];
	data_t* outBuf = mOut[threadIdx];
	data_t* wgtBuf = mWgt;

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
				sum += mBias[getBiasIdx(outD)];
				outBuf[getOutIdx(outX, outY, outD)] = mActivate(sum);
			}
		}
	}
}

void ConvLayer::BackProp(size_t threadIdx)
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
	memset(delOutBuf, 0, sizeof(data_t) * DELTA_OUT_SIZE);
	for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
			{
				data_t delta = delBuf[getDeltaIdx(outX, outY, outD)];
				for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
				{
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
							data_t wgt = wgtBuf[getWgtIdx(kX, kY, inD, outD)];
							delOutBuf[getDOutIdx(inX, inY, inD)] += delta * wgt;
						}
					}
				}
			}
		}
	}
}