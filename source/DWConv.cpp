#pragma once
#include <iostream>
#include <cassert>
#include "DWConv.h"

DwConv::DwConv(size_t kernelLen, size_t inLen, size_t inDepth, size_t outLen, EActFn eActFn)
	: ConvLayer(kernelLen, inLen, inDepth, outLen, inDepth, eActFn)
{

}

DwConv::~DwConv()
{

}