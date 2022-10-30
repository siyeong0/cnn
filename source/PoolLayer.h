#pragma once
#include "ILayer.h"

class PoolLayer : public ILayer
{
public:
	PoolLayer(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn);
	~PoolLayer();

	void Forward() override;
	void BackProp() override;
private:
	inline size_t getInIdx(size_t x, size_t y, size_t d) const;
	inline size_t getOutIdx(size_t x, size_t y, size_t d) const;
	inline size_t getWgtIdx(size_t x, size_t y, size_t inD, size_t outD) const;
	inline size_t getBiasIdx(size_t outD) const;
	inline size_t getDeltaIdx(size_t x, size_t y, size_t d) const;
	inline size_t getDInIdx(size_t x, size_t y, size_t d) const;
	inline size_t getDOutIdx(size_t x, size_t y, size_t d) const;
private:
	size_t* mMaxIdxBuf;
};