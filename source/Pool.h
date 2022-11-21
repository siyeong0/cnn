#pragma once
#include "ILayer.h"

namespace cnn
{
	class Pool : public ILayer
	{
	public:
		Pool(size_t kernelSize, size_t inLen, size_t depth, EActFn eActFn);
		~Pool();

		void Forward(size_t threadIdx) override;
		void BackProp(size_t threadIdx) override;
	private:
		inline size_t getMIBufIdx(size_t x, size_t y, size_t d) const
		{
			size_t idx = (OUTPUT_LEN * y + x) * OUTPUT_DEPTH + d;
			Assert(idx < OUTPUT_SIZE);
			return idx;
		}
	private:
		std::vector<size_t*> mMaxIdxBuf;	// Buffer to store max value's idx
	};
}