#pragma once
#include <vector>
#include "FullConnectLayer.h"
#include "ConvLayer.h"

class Network
{
public:
	friend Network& operator>>(Network& net, ILayer& layer);
	friend Network& operator>>(Network& net, ENet e);
public:
	Network();
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	void Fit();
	data_t GetAccuracy(data_t* data, char* labels, size_t n);

	void SetData(data_t* td, size_t len, char* ld, size_t n);
	void SetBatchSize(size_t b);
	void SetEpochSize(size_t e);
	void SetLearningRate(data_t l);
private:
	int getPredict(size_t threadIdx);
	size_t getIdx(size_t x, size_t y, size_t d) const;
private:
	std::vector<ILayer*> mLayers;
	std::vector<data_t*> mInput;
	std::vector<data_t*> mOutput;
	std::vector<data_t*> mDeltaIn;

	data_t* mData;
	char* mLabels;
	size_t mNumBlocks;

	size_t mBatch;
	size_t mEpoch;
	data_t mLearningRate;

	size_t mInputLen;
	size_t mInputSize;
	size_t mInputDepth;
	size_t mOutputSize;
	size_t mNumPad;
};