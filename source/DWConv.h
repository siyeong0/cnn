#pragma once
#include "ILayer.h"

namespace cnn
{
	class DwConv : public ILayer
	{
	public:
		DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn);
		~DwConv();

		void Forward(size_t threadIdx) override;
		void BackProp(size_t threadIdx) override;
	};
}