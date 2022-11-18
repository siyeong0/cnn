#pragma once
#include "ILayer.h"

namespace cnn
{
	class Linear : public ILayer
	{
	public:
		Linear(size_t inSize, size_t outSize, EActFn eActFn);
		~Linear();

		void Forward(size_t threadIdx) override;
		void BackProp(size_t threadIdx) override;
	};
}