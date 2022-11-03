#pragma once
#include "ILayer.h"

class ConvLayer : public ILayer
{
public:
	ConvLayer(size_t kernelLen, size_t inLen,size_t inDepth, size_t outLen,size_t outDepth, EActFn eActFn);
	~ConvLayer();

	void Forward() override;
	void BackProp() override;
private:
};