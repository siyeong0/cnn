#pragma once
#include "ILayer.h"

class FullConnectLayer : public ILayer
{
public:
	FullConnectLayer(size_t inSize, size_t outSize, EActFn eActFn);
	~FullConnectLayer();

	void Forward(size_t threadIdx) override;
	void BackProp(size_t threadIdx) override;
};