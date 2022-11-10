#include "FullConnectLayer.h"

FullConnectLayer::FullConnectLayer(size_t inSize, size_t outSize, EActFn eActFn)
	: ILayer(1, 1, inSize, 1, outSize, eActFn)
{

}

FullConnectLayer::~FullConnectLayer()
{

}

void FullConnectLayer::Forward(size_t threadIdx)
{
	data_t* inBuf = mIn[threadIdx];
	data_t* outBuf = mOut[threadIdx];
	data_t* wgtBuf = mWgt;

	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t sum = 0.f;
		for (size_t x = 0; x < INPUT_SIZE; ++x)
		{
			sum += wgtBuf[getWgtIdx(0,0,x, y)] * inBuf[x];
		}
		sum += mBias[y];
		outBuf[y] = mActivate(sum);
	}
}
void FullConnectLayer::BackProp(size_t threadIdx)
{
	data_t* inBuf = mIn[threadIdx];
	data_t* outBuf = mOut[threadIdx];
	data_t* wgtBuf = mWgt;
	data_t* delInBuf = mDeltaIn[threadIdx];
	data_t* delBuf = mDelta[threadIdx];
	data_t* delOutBuf = mDeltaOut[threadIdx];
	data_t* wgtDiffBuf = mWgtDiff[threadIdx];
	data_t* biasDiffBuf = mBiasDiff[threadIdx];

	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t deltaIn = delInBuf[y];
		data_t valOut = outBuf[y];
		data_t deriv = 0.f;
		switch (meActFn)
		{
		case EActFn::TANH:
			deriv = 1 - valOut * valOut;
			break;
		case EActFn::RELU:
			deriv = valOut > 0.f ? 1.f : 0.f;
			break;
		case EActFn::SOFTMAX:
			deriv = 1.f;
			break;
		case EActFn::SIGMOID:
			deriv = valOut * (1 - valOut);
			break;
		case EActFn::IDEN:
			deriv = 1.f;
			break;
		default:
			Assert(false);
			break;
		}
		delBuf[y] = deltaIn * deriv;
	}

	memset(delOutBuf, 0, sizeof(data_t) * INPUT_SIZE);
	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t delta = delBuf[y];
		for (size_t x = 0; x < INPUT_SIZE; ++x)
		{
			data_t in = inBuf[x];
			data_t wgt = mWgt[getWgtIdx(0,0,x,y)];
			delOutBuf[x] += wgt * delta;
			wgtDiffBuf[getWgtIdx(0, 0, x, y)] += in * delta;
		}
	}

	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		biasDiffBuf[y] += delBuf[y];
	}
}