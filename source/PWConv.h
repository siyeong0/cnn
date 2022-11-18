#pragma once
#include "Conv.h"

namespace cnn
{
	class PWConv : public Conv
	{
	public:
		PWConv(size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn);
		~PWConv();
	};
}