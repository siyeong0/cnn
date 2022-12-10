#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "../source/Network.h"
#include "../source/Conv.h"
#include "../source/Linear.h"
#include "../source/Pool.h"

#include "../source/DWConv.h"
#include "../source/PWConv.h"

bool ReadCIFARData(const char* filePath, data_t* datas, char* labels);

using namespace cnn;

int main()
{
	data_t* trainDatas = new float[32 * 32 * 3 * 50000];
	char* trainLabels = new char[50000];
	ReadCIFARData("resource/data_batch_1.bin", trainDatas + 32 * 32 * 3 * 0, trainLabels + 0);
	ReadCIFARData("resource/data_batch_2.bin", trainDatas + 32 * 32 * 3 * 10000, trainLabels + 10000);
	ReadCIFARData("resource/data_batch_3.bin", trainDatas + 32 * 32 * 3 * 20000, trainLabels + 20000);
	ReadCIFARData("resource/data_batch_4.bin", trainDatas + 32 * 32 * 3 * 30000, trainLabels + 30000);
	ReadCIFARData("resource/data_batch_5.bin", trainDatas + 32 * 32 * 3 * 40000, trainLabels + 40000);
	data_t* testDatas = new float[32 * 32 * 3 * 10000];
	char* testLabels = new char[10000];
	ReadCIFARData("resource/test_batch.bin", testDatas, testLabels);
	Network net;

	//DwConv dconv32x32x3(5, 32, 3, 32, EActFn::RELU);
	//PWConv pconv32x32x3(32, 3, 32, 32, EActFn::RELU);
	//Pool pool32x32x32(2, 32, 32, EActFn::RELU);
	//DwConv dconv16x16x32(5, 16, 32, 16, EActFn::RELU);
	//PWConv pconv16x16x32(16, 32, 16, 32, EActFn::RELU);
	//Pool pool16x16x32(2, 16, 32, EActFn::RELU);
	//DwConv dconv8x8x32(5, 8, 32, 8, EActFn::RELU);
	//PWConv pconv8x8x64(8, 32, 8, 64, EActFn::RELU);
	//Pool pool8x8x64(2, 8, 64, EActFn::RELU);
	//Linear full1024To64(1024, 64, EActFn::IDEN);
	//Linear full64To10(64, 10, EActFn::SIGMOID);

	//net >> dconv32x32x3 >> pconv32x32x3 >> pool32x32x32
	//	>> dconv16x16x32 >> pconv16x16x32 >> pool16x16x32
	//	>> dconv8x8x32 >> pconv8x8x64 >> pool8x8x64
	//	>> full1024To64 >> full64To10 >> ENet::END;

	Conv conv32x32x3(5, 32, 3, 32, 32, EActFn::RELU);
	Pool pool32x32x32(2, 32, 32, EActFn::RELU);
	Conv conv16x16x32(5, 16, 32, 16, 32, EActFn::RELU);
	Pool pool16x16x32(2, 16, 32, EActFn::RELU);
	Conv conv8x8x32(5, 8, 32, 8, 64, EActFn::RELU);
	Pool pool8x8x64(2, 8, 64, EActFn::RELU);
	Linear full1024To64(1024, 64, EActFn::IDEN);
	Linear full64To10(64, 10, EActFn::SIGMOID);

	net >> conv32x32x3 >> pool32x32x32
		>> conv16x16x32 >> pool16x16x32
		>> conv8x8x32 >> pool8x8x64
		>> full1024To64 >> full64To10 >> ENet::END;

	net.SetBatchSize(16);
	net.SetEpochSize(3);
	net.SetLearningRate(0.1f);
	net.SetData(trainDatas, trainLabels, 50000);


	double beg, end;

	beg = clock();
	net.Fit();
	end = clock();

	std::cout << std::endl << "TIME TAKEN : " << static_cast<int>(end - beg) / CLOCKS_PER_SEC << " sec" << std::endl;


	std::cout << std::endl << net.GetAccuracy(testDatas, testLabels, 10000);

	delete[] trainDatas;
	delete[] trainLabels;
	delete[] testDatas;
	delete[] testLabels;
	return 0;
}

bool ReadCIFARData(const char* filePath, data_t* datas, char* labels)
{
	std::ifstream file(filePath, std::ios::binary);
	if (!file.is_open())
	{
		Assert(false);
	}

	size_t totalSize = (32 * 32 * 3 + 1) * 10000;
	unsigned char* buffer = static_cast<unsigned char*>(malloc(totalSize));
	Assert(buffer != nullptr);
	file.read(reinterpret_cast<char*>(buffer), totalSize);

	for (size_t n = 0; n < 10000; n++)
	{
		labels[n] = buffer[3073 * n];
		for (size_t d = 0; d < 3; d++)
		{
			for (size_t y = 0; y < 32; y++)
			{
				for (size_t x = 0; x < 32; x++)
				{
					size_t i = d * 32 * 32 + y * 32 + x;
					size_t o = (32 * y + x) * 3 + d;
					datas[(32 * 32 * 3) * n + o] = (1.f / 255) * buffer[3073 * n + 1 + i];
				}
			}
		}
	}
	free(buffer);

	file.close();
	return true;
}