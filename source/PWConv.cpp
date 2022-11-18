#pragma once
#include <iostream>
#include <cassert>
#include "PWConv.h"

namespace cnn
{
	PWConv::PWConv(size_t inLen, size_t inDepth, size_t outLen, size_t outDepth, EActFn eActFn)
		: Conv(1, inLen, inDepth, outLen, outDepth, eActFn)
	{

	}

	PWConv::~PWConv()
	{

	}
}