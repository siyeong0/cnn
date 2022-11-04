#pragma once
#include "ILayer.h"

class ConvLayer : public ILayer
{
public:
	ConvLayer(size_t kernelLen, size_t inLen,size_t inDepth, size_t outLen,size_t outDepth, EActFn eActFn);
	~ConvLayer();

	void Forward(size_t threadIdx) override;
	void BackProp(size_t threadIdx) override;
};