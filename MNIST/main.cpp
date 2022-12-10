#include <iostream>
#include <fstream>
#include <vector>

#include "../source/Network.h"
#include "../source/Conv.h"
#include "../source/Linear.h"
#include "../source/Pool.h"

bool ReadMNISTTrainingData(const char* filePath, data_t* dest);
bool ReadMNISTLabelData(std::string filePath, char* labels);

int main(void)
{
	data_t* trainDatas = new float[784 * 60000];
	char* trainLabels = new char[60000];
	if (!ReadMNISTTrainingData("resource/train-images.idx3-ubyte", trainDatas)) abort();
	if (!ReadMNISTLabelData("resource/train-labels.idx1-ubyte", trainLabels)) abort();
	data_t* testDatas = new float[784 * 10000];
	char* testLabels = new char[10000];
	if (!ReadMNISTTrainingData("resource/t10k-images.idx3-ubyte", testDatas)) abort();
	if (!ReadMNISTLabelData("resource/t10k-labels.idx1-ubyte", testLabels)) abort();

	using namespace cnn;

	Network net;
	Conv conv32x32x1(5, 28, 1, 28, 6, EActFn::RELU);
	Pool pool28x28x6(2, 28, 6, EActFn::RELU);
	Conv conv14x14x6(5, 14, 6, 10, 16, EActFn::RELU);
	Pool pool10x10x16(2, 10, 16, EActFn::RELU);
	Conv conv5x5x16(5, 5, 16, 1, 120, EActFn::RELU);
	Linear full120To10(120, 10, EActFn::SIGMOID);
	net >> conv32x32x1 >> pool28x28x6 >> conv14x14x6 >> pool10x10x16 >> conv5x5x16 >> full120To10 >> ENet::END;

	net.SetBatchSize(16);
	net.SetEpochSize(30);
	net.SetLearningRate(0.02f);
	net.SetData(trainDatas, trainLabels, 50000);
	net.Fit();
	std::cout << std::endl << "TEST ACCURACY : " << net.GetAccuracy(testDatas, testLabels, 10000);

	delete[] trainDatas;
	delete[] trainLabels;
	delete[] testDatas;
	delete[] testLabels;

	return 0;
}

inline uint32_t ReverseInt32(uint32_t val)
{
	uint32_t b1, b2, b3, b4;
	b1 = val & 0x0ff;
	b2 = (val >> 8) & 0x0ff;
	b3 = (val >> 16) & 0x0ff;
	b4 = (val >> 24) & 0x0ff;

	val = (b1 << 24) + (b2 << 16) + (b3 << 8) + b4;

	return val;
}

bool ReadMNISTTrainingData(const char* filePath, data_t* dest)
{
	std::ifstream file(filePath, std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}

	uint32_t magicNum = 0;
	uint32_t numElements = 0;
	uint32_t row = 0;
	uint32_t col = 0;
	file.read(reinterpret_cast<char*>(&magicNum), sizeof(magicNum));
	magicNum = ReverseInt32(magicNum);
	file.read(reinterpret_cast<char*>(&numElements), sizeof(numElements));
	numElements = ReverseInt32(numElements);
	file.read(reinterpret_cast<char*>(&row), sizeof(row));
	row = ReverseInt32(row);
	file.read(reinterpret_cast<char*>(&col), sizeof(col));
	col = ReverseInt32(col);

	size_t totalSize = row * col * numElements;
	//*dest = static_cast<data_t*>(malloc(sizeof(data_t) * totalSize));
	unsigned char* buffer = static_cast<unsigned char*>(malloc(totalSize));
	file.read(reinterpret_cast<char*>(buffer), totalSize);
	for (size_t i = 0; i < totalSize; i++)
	{
		dest[i] = (1.f / 255) * buffer[i];
	}
	free(buffer);

	file.close();
	return true;
}

bool ReadMNISTLabelData(std::string filePath, char* labels)
{
	std::ifstream file(filePath, std::ios::binary);
	if (!file.is_open())
	{
		Assert(false);
	}

	uint32_t magicNumber = 0;
	uint32_t numElements = 0;

	file.read(reinterpret_cast<char*>(&magicNumber), sizeof(uint32_t));
	magicNumber = ReverseInt32(magicNumber);
	file.read(reinterpret_cast<char*>(&numElements), sizeof(uint32_t));
	numElements = ReverseInt32(numElements);

	//*labels = static_cast<char*>(malloc(numElements));
	file.read(labels, numElements);

	file.close();
	return true;
}