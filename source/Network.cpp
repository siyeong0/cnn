#include "Network.h"
#include "ILayer.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

Network& operator>>(Network& net, ILayer& layer)
{
	net.mLayers.push_back(&layer);
	return net;
}

Network& operator>>(Network& net, ENet e)
{
	Assert(e == ENet::END);
	size_t size = net.mLayers.size();
	Assert(size > 0);

	ILayer& head = *(net.mLayers[0]);
	ILayer& tail = *(net.mLayers[size - 1]);
	// Set constants
	net.mInputLen = head.INPUT_PAD_LEN;
	net.mInputDepth = head.INPUT_DEPTH;
	net.mInputSize = net.mInputLen * net.mInputLen;
	net.mOutputSize = tail.OUTPUT_SIZE;
	net.mNumPad = head.NUM_PAD;
	// Connect network's buffers and head,tail layers' buffers
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		net.mInput.push_back(head.mIn[i]);
		net.mOutput.push_back(Alloc<data_t>(tail.OUTPUT_SIZE));
		tail.mOut.push_back(net.mOutput[i]);
		net.mDeltaIn.push_back(Alloc<data_t>(tail.OUTPUT_SIZE));
		tail.mDeltaIn.push_back(net.mDeltaIn[i]);
	}
	// Connect layers' buffers
	for (size_t i = 0; i < size - 1; ++i)
	{
		ILayer& curr = *(net.mLayers[i]);
		ILayer& next = *(net.mLayers[i + 1]);
		curr.mOutPad = next.NUM_PAD;
		for (size_t j = 0; j < NUM_THREAD; ++j)
		{
			curr.mOut.push_back(next.mIn[j]);
			curr.mDeltaIn.push_back(next.mDeltaOut[j]);
		}
	}

	return net;
}

Network::Network()
	: mInput()
	, mOutput()
	, mDeltaIn()
	, mData(nullptr)
	, mLabels(nullptr)
	, mNumBlocks(0)
	, mBatch(0)
	, mEpoch(0)
	, mLearningRate(0.f)
	, mInputLen(0)
	, mInputSize(0)
	, mInputDepth(0)
	, mOutputSize(0)
	, mNumPad(0)
{
}

Network::~Network()
{
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		if (mOutput[i] != nullptr) { Free(mOutput[i]); }
		if (mDeltaIn[i] != nullptr) { Free(mDeltaIn[i]); }
	}
}

void Network::Fit()
{
	std::vector<std::thread> threads;
	size_t numLayers = mLayers.size();
	size_t imgIdx = 0;

	for (size_t e = 0; e < mEpoch; ++e)
	{
		// Print progress
		if (e % 10 == 0)
		{
			system("cls");
			std::cout << ((double)(e) / mEpoch) * 100 << " %\n"<< GetAccuracy(mData, mLabels, 16);
		}
		// Initialize batch : set weight diff/bias diff to 0
		for (size_t i = 0; i < numLayers; ++i)
		{
			mLayers[i]->InitBatch();
		}
		// Get parameters' gradients
		size_t batch = mBatch - mBatch % NUM_THREAD;
		for (size_t b = 0; b < batch / NUM_THREAD; ++b)
		{
			for (size_t idx = 0; idx < NUM_THREAD; ++idx)
			{
				auto func = [this, &numLayers](size_t threadIdx, data_t* input, int label)
				{
					data_t* inputBuf = mInput[threadIdx];
					data_t* outputBuf = mOutput[threadIdx];
					data_t* delInBuf = mDeltaIn[threadIdx];
					// Copy input
					data_t* data = input;
					memset(inputBuf, 0, sizeof(data_t) * (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * mInputDepth);
					for (size_t y = 0; y < mInputLen; ++y)
					{
						for (size_t x = 0; x < mInputLen; ++x)
						{
							for (size_t d = 0; d < mInputDepth; ++d)
							{
								inputBuf[getIdx(x, y, d)] = data[mInputLen * mInputLen * d + y * mInputLen + x];
							}
						}
					}
					// Forward propagation
					for (size_t i = 0; i < numLayers; ++i)
					{
						mLayers[i]->Forward(threadIdx);
					}
					// Set output delta
					for (size_t i = 0; i < mOutputSize; i++)
					{
						data_t y = outputBuf[i];
						data_t yi = (label == i) ? 1.f : 0.f;
						delInBuf[i] = 0.2f * (y - yi);
					}
					// Back Propagation
					for (size_t i = 0; i < numLayers; i++)
					{
						size_t idx = numLayers - i - 1;
						mLayers[idx]->BackProp(threadIdx);
					}
				};
				data_t* input = mData + mInputSize * imgIdx;
				int label = static_cast<int>(mLabels[imgIdx]);
				threads.push_back(std::thread(func, idx, input, label));
				imgIdx++;
				imgIdx %= mNumBlocks;
			}
			for (size_t i = 0; i < NUM_THREAD; ++i)
			{
				threads[i].join();
			}
			threads.clear();
		}
		// Fit parameters
		for (size_t i = 0; i < numLayers; i++)
		{
			mLayers[i]->Update(mBatch, mLearningRate);
		}
	}
}

data_t Network::GetAccuracy(data_t* data, char* labels, size_t n)
{
	std::vector<std::thread> threads;
	std::vector<size_t> correctCount(NUM_THREAD);
	size_t imgIdx = 0;
	size_t numLayers = mLayers.size();
	auto func = [this, &numLayers, &correctCount](size_t threadIdx, data_t* input, int label)
	{
		data_t* inputBuf = mInput[threadIdx];
		data_t* outputBuf = mOutput[threadIdx];
		data_t* delInBuf = mDeltaIn[threadIdx];
		// Copy input
		data_t* data = input;
		memset(inputBuf, 0, sizeof(data_t) * (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * mInputDepth);
		for (size_t y = 0; y < mInputLen; ++y)
		{
			for (size_t x = 0; x < mInputLen; ++x)
			{
				for (size_t d = 0; d < mInputDepth; ++d)
				{
					inputBuf[getIdx(x, y, d)] = data[mInputLen * mInputLen * d + y * mInputLen + x];
				}
			}
		}
		// Forward propagation
		for (size_t i = 0; i < numLayers; ++i)
		{
			mLayers[i]->Forward(threadIdx);
		}
		//std::cout << label << "," << getPredict(threadIdx) << std::endl;
		if (label == getPredict(threadIdx))
		{
			correctCount[threadIdx]++;
		}
	};
	n -= n % NUM_THREAD;
	while (imgIdx < n)
	{
		for (size_t idx = 0; idx < NUM_THREAD; ++idx)
		{
			data_t* input = data + mInputSize * imgIdx;
			int label = static_cast<int>(mLabels[imgIdx]);
			threads.push_back(std::thread(func, idx, input, label));
			imgIdx++;
		}
		for (size_t i = 0; i < NUM_THREAD; ++i)
		{
			threads[i].join();
		}
		threads.clear();
	}
	size_t sum = 0;
	for (size_t i = 0; i < NUM_THREAD; ++i)
	{
		sum += correctCount[i];
	}

	return static_cast<data_t>(sum) / n;
}

void Network::SetData(data_t* td, size_t len, char* ld, size_t n)
{
	mData = td;
	mLabels = ld;
	mNumBlocks = n;
	mNumPad = (mInputLen - len) / 2;
	mInputLen -= 2 * mNumPad;
	mInputSize = mInputLen * mInputLen;
}

void Network::SetBatchSize(size_t b)
{
	mBatch = b;
}

void Network::SetEpochSize(size_t e)
{
	mEpoch = e;
}

void Network::SetLearningRate(data_t l)
{
	mLearningRate = l;
}

int Network::getPredict(size_t threadIdx)
{
	data_t* outBuf = mOutput[threadIdx];

	int idx = 0;
	data_t max = 0.f;
	for (size_t i = 0; i < mOutputSize; i++)
	{
		data_t out = outBuf[i];
		if (max < out)
		{
			idx = i;
			max = out;
		}
	}
	return idx;
}

size_t Network::getIdx(size_t x, size_t y, size_t d) const
{
	//size_t idx = ((mInputLen + 2 * mNumPad) * (mNumPad + y) + (mNumPad + x)) * mInputDepth + d;
	size_t idx = (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * d + (mInputLen + 2 * mNumPad) * (mNumPad + y) + (mNumPad + x);
	return idx;
};