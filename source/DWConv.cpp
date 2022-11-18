#pragma once
#include <iostream>
#include <cassert>
#include "DWConv.h"

namespace cnn
{
	DwConv::DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn)
		: Conv(kernelLen, inLen, inDepth, outLen, inDepth, eActFn)
	{

	}

	DwConv::~DwConv()
	{

	}
}