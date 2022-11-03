#pragma once
#include <iostream>
#include <cassert>
#include "ConvLayer.h"

ConvLayer::ConvLayer(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
	: ILayer(kernelLen, inLen, inDepth,outLen,outDepth,eActFn)
{

}

ConvLayer::~ConvLayer()
{

}

void ConvLayer::Forward()
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
							data_t in = mIn[getInIdx(outX + kX, outY + kY, inD)];
							data_t wgt = mWgt[getWgtIdx(kX, kY, inD, outD)];
							sum += in * wgt;
						}
					}
				}
				sum += mBias[getBiasIdx(outD)];
				mOut[getOutIdx(outX, outY, outD)] = mActivate(sum);
			}
		}
	}
}

void ConvLayer::BackProp()
{
	for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
			{
				data_t deltaIn = mDeltaIn[getDInIdx(outX, outY, outD)];
				data_t out = mOut[getOutIdx(outX, outY, outD)];
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
				mDelta[getDeltaIdx(outX, outY, outD)] = deltaIn * deriv;
			}
		}
	}

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
							data_t delta = mDelta[getDeltaIdx(outX, outY, outD)];
							data_t valIn = mIn[getInIdx(outX + kX, outY + kY, inD)];
							sum += delta * valIn;
						}
					}
					mWgtDiff[getWgtIdx(kX, kY, inD, outD)] += sum;
				}
			}
		}
	}

	memset(mDeltaOut, 0, sizeof(data_t) * DELTA_OUT_SIZE);
	// TODO : ?
	/*if (NUM_PAD == 0)
	{
		for (size_t inY = 0; inY < INPUT_PAD_LEN; ++inY)
		{
			for (size_t inX = 0; inX < INPUT_PAD_LEN; ++inX)
			{
				for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
				{
					data_t sum = 0.f;
					for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
					{
						for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
						{
							size_t outX = inX - kX;
							size_t outY = inY - kY;
							if (outX >= OUTPUT_LEN || outY >= OUTPUT_LEN)
							{
								continue;
							}
							for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
							{
								data_t delta = mDelta[getDeltaIdx(outX, outY, outD)];
								data_t wgt = mWgt[getWgtIdx(kX,kY,inD,outD)];
								sum += delta * wgt;
							}
						}
					}
					mDeltaOut[getDOutIdx(inX, inY, inD)] = sum;
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
				for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
				{
					data_t sum = 0.f;
					for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
					{
						for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
						{
							for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
							{
								data_t delta = mDelta[getDeltaIdx(outX, outY, outD)];
								data_t wgt = mWgt[getWgtIdx(kX, kY, inD, outD)];
								sum += delta * wgt;
							}
						}
					}
					mDeltaOut[getDOutIdx(outX, outY, inD)] += sum;
				}
			}
		}
	}*/


	for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
	{
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
			{
				data_t delta = mDelta[getDeltaIdx(outX, outY, outD)];
				for (size_t inD = 0; inD < INPUT_DEPTH; ++inD)
				{
					for (size_t kY = 0; kY < KERNEL_LEN; ++kY)
					{
						for (size_t kX = 0; kX < KERNEL_LEN; ++kX)
						{
							data_t wgt = mWgt[getWgtIdx(kX, kY, inD, outD)];
							mDeltaOut[getDOutIdx(outX, outY, inD)] += wgt * delta;
						}
					}
				}
			}
		}
	}

	for (size_t outD = 0; outD < OUTPUT_DEPTH; ++outD)
	{
		data_t sum = 0.f;
		for (size_t outY = 0; outY < OUTPUT_LEN; ++outY)
		{
			for (size_t outX = 0; outX < OUTPUT_LEN; ++outX)
			{
				sum += mDelta[getDeltaIdx(outX, outY, outD)];
			}
		}
		mBiasDiff[outD] += sum;
	}
}