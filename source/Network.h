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
	inline int getPredict();
private:
	std::vector<ILayer*> mLayers;
	data_t* mInput;
	data_t* mOutput;
	data_t* mDeltaIn;

	data_t* mData;
	char* mLabels;
	size_t mNumBlocks;

	size_t mBatch;
	size_t mEpoch;
	data_t mLearningRate;

	size_t mInputLen;
	size_t mInputSize;
	size_t mOutputSize;

	std::vector<size_t> mShuffleIdxs;
	size_t mNumPad;
};