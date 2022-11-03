#pragma once
#include "ILayer.h"

class FullConnectLayer : public ILayer
{
public:
	FullConnectLayer(size_t inSize, size_t outSize, EActFn eActFn);
	~FullConnectLayer();

	void Forward() override;
	void BackProp() override;
private:
	size_t getIdx(size_t x, size_t y) const;
};