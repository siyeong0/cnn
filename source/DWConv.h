#pragma once
#include "Conv.h"

namespace cnn
{
	class DwConv : public Conv
	{
	public:
		DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn);
		~DwConv();
	};
}