#pragma once
#include "ILayer.h"

class PoolLayer : public ILayer
{
public:
	PoolLayer(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn);
	~PoolLayer();

	void Forward(size_t threadIdx) override;
	void BackProp(size_t threadIdx) override;
private:
	size_t getMIBufIdx(size_t x, size_t y, size_t d) const;
private:
	std::vector<size_t*> mMaxIdxBuf;
};