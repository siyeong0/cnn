#include "FullConnectLayer.h"

FullConnectLayer::FullConnectLayer(size_t inSize, size_t outSize, EActFn eActFn)
	: ILayer(1, 1, inSize, 1, outSize, eActFn)
{

}

FullConnectLayer::~FullConnectLayer()
{

}

void FullConnectLayer::Forward()
{
	Assert(mOutPad == 0);
	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t sum = 0.f;
		for (size_t x = 0; x < INPUT_SIZE; ++x)
		{
			sum += mWgt[getIdx(x, y)] * mIn[x];
		}
		sum += mBias[y];
		mOut[y] = mActivate(sum);
	}
}
void FullConnectLayer::BackProp()
{
	Assert(mOutPad == 0);
	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t deltaIn = mDeltaIn[y];
		data_t valOut = mOut[y];
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
		default:
			break;
		}
		mDelta[y] = deltaIn * deriv;
	}

	memset(mDeltaOut, 0, sizeof(data_t) * INPUT_SIZE);
	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		data_t delta = mDelta[y];
		for (size_t x = 0; x < INPUT_SIZE; ++x)
		{
			data_t in = mIn[x];
			data_t wgt = mWgt[getIdx(x, y)];
			mDeltaOut[x] += wgt * delta;
			mWgtDiff[getIdx(x, y)] += in * delta;
		}
	}

	for (size_t y = 0; y < OUTPUT_SIZE; ++y)
	{
		mBiasDiff[y] += mDelta[y];
	}
}

size_t FullConnectLayer::getIdx(size_t x, size_t y) const
{
	Assert(((y * INPUT_SIZE) + x) < (INPUT_SIZE * OUTPUT_SIZE));
	return (y * INPUT_SIZE) + x;
}