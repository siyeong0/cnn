#pragma once
#include "ConvLayer.h"

class PWConv : public ConvLayer
{
public:
	PWConv(size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn);
	~PWConv();
};