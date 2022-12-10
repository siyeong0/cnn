#include "Network.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <ppl.h>

namespace cnn
{
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
		net.mInputLen = head.INPUT_LEN;
		net.mInputDepth = head.INPUT_DEPTH;
		net.mOutputSize = tail.OUTPUT_SIZE;
		net.mNumPad = head.NUM_PAD;
		net.mInputSize = net.mInputLen * net.mInputLen * net.mInputDepth;
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
		, mNumImages(0)
		, mBatchSize(0)
		, mEpochSize(0)
		, mLearningRate(0.01f)
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

	void Network::Fit(EAvx eAvx)
	{
		std::vector<std::thread> threads;
		const size_t NUM_LAYERS = mLayers.size();
		// Clear input buffers
		for (size_t i = 0; i < mInput.size(); ++i)
		{
			memset(mInput[i], 0, sizeof(data_t) * (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * mInputDepth);
		}
		//
		for (size_t i = 0; i < NUM_LAYERS; ++i)
		{
			mLayers[i]->UseAvx(eAvx == EAvx::TRUE);
		}
		//
		const data_t LR = mLearningRate;
		for (size_t e = 0; e < mEpochSize; ++e)
		{
			// Shuffle datas
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(mImages.begin(), mImages.end(), g);
			// Print progress
			std::cout << "EPOCH : " << e + 1 << "\n";
			std::cout << "|";
			// Initialize constants
			const size_t BATCH = mBatchSize - mBatchSize % NUM_THREAD;
			const size_t BATCH_PER_EPOCH = mNumImages / BATCH;
			const size_t BATCH_DIV_THREAD = BATCH / NUM_THREAD;
			// Train
			for (size_t be = 0; be < BATCH_PER_EPOCH; ++be)
			{
				// Print progress
				if ((be + 1) % (BATCH_PER_EPOCH / 10) == 0)
				{
					std::cout << "--|";
				}
				// Initialize batch : set weight diff/bias diff to 0
				for (size_t i = 0; i < NUM_LAYERS; ++i)
				{
					mLayers[i]->InitBatch();
				}
				// Get parameters' gradients
				concurrency::parallel_for(0, static_cast<int>(NUM_THREAD), [&](int threadIdx)
					{
						data_t* inputBuf = mInput[threadIdx];
						data_t* outputBuf = mOutput[threadIdx];
						data_t* delInBuf = mDeltaIn[threadIdx];

						for (size_t n = 0; n < BATCH_DIV_THREAD; ++n)
						{
							IM currIm = mImages[be * BATCH + threadIdx * (n + 1)];
							const data_t* input = currIm.Data;
							const int label = currIm.Class;
							// Copy input
							for (size_t y = 0; y < mInputLen; ++y)
							{
								for (size_t x = 0; x < mInputLen; ++x)
								{
									for (size_t d = 0; d < mInputDepth; ++d)
									{
										inputBuf[getIdx(x, y, d)] = input[(mInputLen * y + x) * mInputDepth + d];
									}
								}
							}
							// Forward propagation
							for (size_t i = 0; i < NUM_LAYERS; ++i)
							{
								mLayers[i]->Forward(threadIdx);
							}
							// Set output delta
							size_t truth = label;
							for (size_t i = 0; i < mOutputSize; i++)
							{
								data_t y = outputBuf[i];
								data_t yi = (label == i) ? 1.f : 0.f;
								delInBuf[i] = 0.2f * (y - yi);
							}
							// Back Propagation
							for (size_t i = 0; i < NUM_LAYERS; i++)
							{
								size_t idx = NUM_LAYERS - i - 1;
								mLayers[idx]->BackProp(threadIdx);
							}
						}
					});
				// Fit parameters
				int nl = NUM_LAYERS;
				size_t ic = 0;
				while (nl > 0)
				{
					int nt = nl < static_cast<int>(NUM_THREAD) ? nl : NUM_THREAD;
					concurrency::parallel_for(0, nt, [&](int threadIdx)
						{
							mLayers[ic * NUM_THREAD + threadIdx]->Update(BATCH, LR);
						});
					nl -= NUM_THREAD;
					ic += 1;
				}
			}
			// Print current accuracy
			constexpr size_t NUM_FOLD = 10;
			static size_t valIdx = 0;
			const size_t NUM_VIMGES = mNumImages / NUM_FOLD;
			const size_t OFFSET = valIdx * NUM_VIMGES;
			std::cout << "\nACCURACY : "
				<< GetAccuracy(mData + mInputSize * OFFSET, mLabels + OFFSET, NUM_VIMGES) << std::endl << std::endl;
			valIdx++;
			valIdx %= NUM_FOLD;
		}
	}

	data_t Network::GetAccuracy(data_t* data, char* labels, size_t n)
	{
		std::vector<std::thread> threads;
		std::vector<size_t> correctCount(NUM_THREAD);
		size_t imgIdx = 0;
		size_t numLayers = mLayers.size();
		n -= n % NUM_THREAD;
		while (imgIdx < n)
		{
			// Get outputs
			auto func = [this, &numLayers, &correctCount](size_t threadIdx, data_t* input, int label)
			{
				data_t* inputBuf = mInput[threadIdx];
				data_t* outputBuf = mOutput[threadIdx];
				data_t* delInBuf = mDeltaIn[threadIdx];
				// Copy input
				for (size_t y = 0; y < mInputLen; ++y)
				{
					for (size_t x = 0; x < mInputLen; ++x)
					{
						for (size_t d = 0; d < mInputDepth; ++d)
						{
							inputBuf[getIdx(x, y, d)] = input[(mInputLen * y + x) * mInputDepth + d];
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
			for (size_t idx = 0; idx < NUM_THREAD; ++idx)
			{
				data_t* input = data + mInputSize * imgIdx;
				int label = static_cast<int>(labels[imgIdx]);
				threads.push_back(std::thread(func, idx, input, label));
				imgIdx++;
			}
			for (size_t i = 0; i < NUM_THREAD; ++i)
			{
				threads[i].join();
			}
			threads.clear();
		}
		// Sum num true positive
		size_t sum = 0;
		for (size_t i = 0; i < NUM_THREAD; ++i)
		{
			sum += correctCount[i];
		}

		return static_cast<data_t>(sum) / n;
	}

	void Network::SetData(data_t* td, char* ld, size_t n)
	{
		mData = td;
		mLabels = ld;
		mNumImages = n;
		mImages.reserve(mNumImages);
		for (size_t i = 0; i < mNumImages; ++i)
		{
			Network::IM im;
			im.Data = td;
			im.Class = static_cast<int>(*ld);
			mImages.push_back(im);
			td += mInputSize;
			ld += 1;
		}
	}

	void Network::SetBatchSize(size_t b)
	{
		mBatchSize = b;
	}

	void Network::SetEpochSize(size_t e)
	{
		mEpochSize = e;
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
		for (size_t i = 1; i < mOutputSize; i++)
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
		size_t idx = ((mInputLen + 2 * mNumPad) * (mNumPad + y) + (mNumPad + x)) * mInputDepth + d;
		//size_t idx = (mInputLen + 2 * mNumPad) * (mInputLen + 2 * mNumPad) * d + (mInputLen + 2 * mNumPad) * (mNumPad + y) + (mNumPad + x);
		return idx;
	};
}