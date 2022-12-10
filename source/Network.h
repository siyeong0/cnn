#pragma once
#include <vector>
#include "ILayer.h"

namespace cnn
{
	enum class EAvx
	{
		FALSE = 0,
		TRUE = 1,
	};

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

		void Fit(EAvx USE_AVX = EAvx::TRUE);
		data_t GetAccuracy(data_t* data, char* labels, size_t n);

		void SetData(data_t* td, char* ld, size_t n);
		void SetBatchSize(size_t b);
		void SetEpochSize(size_t e);
		void SetLearningRate(data_t l);
	private:
		int getPredict(size_t threadIdx);
		size_t getIdx(size_t x, size_t y, size_t d) const;
	private:
		std::vector<ILayer*> mLayers;
		// vector elements are buffers allocated to threads
		std::vector<data_t*> mInput;
		std::vector<data_t*> mOutput;
		std::vector<data_t*> mDeltaIn;

		// Raw input images
		struct IM
		{
			data_t* Data;
			int Class;
		};
		data_t* mData;
		char* mLabels;
		std::vector<IM> mImages;
		size_t mNumImages;

		size_t mBatchSize;
		size_t mEpochSize;
		data_t mLearningRate;	// Default : 0.01

		size_t mInputLen;	// Not padded
		size_t mInputSize;	// Not padded
		size_t mInputDepth;
		size_t mOutputSize; // Not padded
		size_t mNumPad;		// Input's pad
	};
}