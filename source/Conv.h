#pragma once
#include "ILayer.h"

namespace cnn
{
	class Conv : public ILayer
	{
	public:
		Conv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn);
		~Conv();

		void Forward(size_t threadIdx) override;
		void BackProp(size_t threadIdx) override;
	};
}