#pragma once
#include "ConvLayer.h"

class DwConv : public ConvLayer
{
public:
	DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn);
	~DwConv();
};