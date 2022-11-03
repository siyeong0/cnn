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
	net.mInput = head.mIn;
	net.mOutput = Alloc<data_t>(tail.OUTPUT_SIZE);
	tail.mOut = net.mOutput;
	net.mDeltaIn = Alloc<data_t>(tail.OUTPUT_SIZE);
	tail.mDeltaIn = net.mDeltaIn;

	for (size_t i = 0; i < size - 1; ++i)
	{
		ILayer& curr = *(net.mLayers[i]);
		ILayer& next = *(net.mLayers[i + 1]);
		//Assert(curr.OUTPUT_DEPTH == next.INPUT_DEPTH);
		curr.mOut = next.mIn;
		curr.mDeltaIn = next.mDeltaOut;
		curr.mOutPad = next.NUM_PAD;
		int a = 3;
	}
	net.mInputLen = head.INPUT_PAD_LEN;
	net.mInputDepth = head.INPUT_DEPTH;
	net.mInputSize = net.mInputLen * net.mInputLen;
	net.mOutputSize = tail.OUTPUT_SIZE;
	net.mNumPad = head.NUM_PAD;

	return net;
}

Network::Network()
	: mInput(nullptr)
	, mOutput(nullptr)
	, mDeltaIn(nullptr)
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
	if (mOutput != nullptr) { Free(mOutput); }
	if (mDeltaIn != nullptr) { Free(mDeltaIn); }
}

void Network::Fit()
{
	if (mEpoch % mBatch != 0)
	{
		abort();
	}
	std::random_device rd;
	std::mt19937 g(rd());
	size_t numLayers = mLayers.size();
	int c = 0;
	std::shuffle(mShuffleIdxs.begin(), mShuffleIdxs.end(), g);
	size_t shIdx = 0;
	for (size_t e = 0; e < mEpoch; ++e)
	{
		if (shIdx >= mShuffleIdxs.size())
		{
			shIdx = 0;
			std::shuffle(mShuffleIdxs.begin(), mShuffleIdxs.end(), g);
		}
		for (size_t i = 0; i < numLayers; ++i)
		{
			mLayers[i]->InitBatch();
		}
		for (size_t b = 0; b < mBatch; ++b)
		{
			size_t currIdx = mShuffleIdxs[shIdx++];
			if (currIdx >= mNumBlocks)
			{
				currIdx = 0;
			}
			// Copy input
			data_t* data = mData + currIdx * mInputSize;
			memset(mInput, 0, sizeof(data_t) * (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * mInputDepth);
			for (size_t y = 0; y < mInputLen; ++y)
			{
				for (size_t x = 0; x < mInputLen; ++x)
				{
					for (size_t d = 0; d < mInputDepth; ++d)
					{
						int ii = getIdx(x, y, d);
						mInput[getIdx(x, y, d)] = data[mInputLen * mInputLen * d + y * mInputLen + x];
					}
				}
			}
			int label = static_cast<int>(mLabels[currIdx]);

			//Print code for debug
			//for (size_t d = 0; d < mInputDepth; d++)
			//{
			//	for (size_t y = 0; y < mInputLen + 2 * mNumPad; y++)
			//	{
			//		for (size_t x = 0; x < mInputLen + 2 * mNumPad; x++)
			//		{
			//			std::cout << (int)(mInput[(mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * d + (mInputLen + 2 * mNumPad) * y + x] + 0.5) << " ";
			//		}
			//		std::cout << std::endl;
			//	}
			//	std::cout << std::endl;
			//}

			for (size_t i = 0; i < numLayers; ++i)
			{
				mLayers[i]->Forward();
			}

			// Get predict result;
			int pr = getPredict();
			if (pr == label) c++;
			// Set delta buffer
			for (size_t i = 0; i < mOutputSize; i++)
			{
				data_t y = mOutput[i];
				data_t yi = (label == i) ? 1.f : 0.f;
				mDeltaIn[i] = 0.2f * (y - yi);
			}
			for (size_t i = 0; i < numLayers; i++)
			{
				size_t idx = numLayers - i - 1;
				mLayers[idx]->BackProp();
			}

			//std::cout << label << "," << pr << std::endl;
		}
		for (size_t i = 0; i < numLayers; i++)
		{
			mLayers[i]->Update(mBatch, mLearningRate);
		}

		// Print progress
		static int pp = 0;
		static int es = 0;
		pp++;
		if (pp % 100 == 0)
		{
			pp = 0;
			double prog = (double)(e + 1) / mEpoch;
			prog *= 100;
			system("cls");
			std::cout << (int)prog << "%  " << ((double)c / (mBatch * (e - es + 1)) * 100) << std::endl;
			es = e;
			c = 0;
		}
	}
}

data_t Network::GetAccuracy(data_t* data, char* labels, size_t n)
{
	size_t c = 0;
	size_t numLayers = mLayers.size();
	for (size_t i = 0; i < n; ++i)
	{
		data_t* data = mData + i * mInputSize;
		memset(mInput, 0, sizeof(data_t) * (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * mInputDepth);
		for (size_t y = 0; y < mInputLen; ++y)
		{
			for (size_t x = 0; x < mInputLen; ++x)
			{
				for (size_t d = 0; d < mInputDepth; ++d)
				{
					int ii = getIdx(x, y, d);
					mInput[getIdx(x, y, d)] = data[mInputLen * mInputLen * d + y * mInputLen + x];
				}
			}
		}
		int label = static_cast<int>(mLabels[i]);;

		for (size_t i = 0; i < numLayers; ++i)
		{
			mLayers[i]->Forward();
		}
		int pr = getPredict();

		//std::cout << pr << " " << label << std::endl;

		c += static_cast<size_t>(pr == label);
		data += mInputSize;
	}
	mInput = nullptr;
	return static_cast<data_t>(c) / n;
}

void Network::SetData(data_t* td, size_t len, char* ld, size_t n)
{
	mData = td;
	mLabels = ld;
	mNumBlocks = n;
	mNumPad = (mInputLen - len) / 2;
	mInputLen -= 2 * mNumPad;
	mInputSize = mInputLen * mInputLen;
	mShuffleIdxs.clear();
	mShuffleIdxs.reserve(mNumBlocks);
	for (size_t i = 0; i < mNumBlocks; i++)
	{
		mShuffleIdxs.push_back(i);
	}
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

int Network::getPredict()
{
	int idx = 0;
	data_t max = 0.f;
	for (size_t i = 0; i < mOutputSize; i++)
	{
		data_t out = mOutput[i];
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