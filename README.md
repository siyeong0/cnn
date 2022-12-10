# Convolution neural network in C++
## Contents
* [Features](#features)
* [Implementations](#layers)
* [Performance](#performance)
* [Add custom layer](#add-custom-layer)
* [References](#references)
* [License](#license)
## Features
- C++17을 지원하는 환경에서 모두 동작 가능
- CPU에서 동작하지만 적정한 성능을 냄 : [Performance](#performance)
    - 멀티쓰레드와 AVX256지원
- Custom layer을 쉽게 추가 가능 : [Add custom layer](#add-custom-layer)
## Implementations
### Layers
- Convolution 2d layer
- Max pooling layer
- Fully connected layer
- Depth wise convolution layer
- Point wise convolution layer
### Activation functions
- Identity
- Sigmoid
- Tanh
- Relu
- Softmax
### Weight initialization
- Glorot2010
### Loss functions
- Mean squared error
### Optimizer
- Stochastic gradient descent
- Adoptive moment

    # Example
    ```C++ 
    #include "Network.h"
    #include "Conv.h"
    #include "Linear.h"
    #include "Pool.h"
    using namespace cnn;
    int main()
    {
        // ...
        // Load your data
        // ...
        Network net;
        Conv conv32x32x1(5, 28, 1, 28, 6, EActFn::RELU);
        Pool pool28x28x6(2, 28, 6, EActFn::RELU);
        Conv conv14x14x6(5, 14, 6, 10, 16, EActFn::RELU);
        Pool pool10x10x16(2, 10, 16, EActFn::RELU);
        Conv conv5x5x16(5, 5, 16, 1, 120, EActFn::RELU);
        Linear full120To10(120, 10, EActFn::SIGMOID);
        net >> conv32x32x1 >> pool28x28x6 >> conv14x14x6 >> pool10x10x16 >> conv5x5x16 >> full120To10 >> ENet::END;
        net.SetBatchSize(16);
        net.SetEpochSize(1);
        net.SetLearningRate(0.01f);
        net.SetData(trainDatas, trainLabels, 50000);
        net.Fit();
        // ...
    }
    ```
## Performance
### MNIST
- [MNIST](https://github.com/siyeong0/cnn/tree/main/MNIST) 테스트 데이터에서 최대 98.6%
### CIFAR10
- [CIFAR10](https://github.com/siyeong0/cnn/tree/main/CIFAR) 테스트 데이터에서 최대 76.6%
### [Example](#example)모델의 학습시간 측정, epoch size 5, batch size 16 (mt = multi thread)
| |none|mt|mt/avx|pytorch(gpu)|
|--|:----:|:------------:|:------------------:|:------------:|
|time(sec)|11125|     1015  |     167          |      87    |

### Depthwise separable convolution을 적용했을 때의 학습 시간, 정확도 비교
||none|dwsep|
|:---:|:---:|:---:|
|time(sec)|167|92|
|acc(%)|76.6|64.0|
## Add custom layer
### 아래 형식에 맞게 Custom layer 구현
    ```C++ 
    #include "ILayer.h"
    namespace cnn
    {
        class CustomLayer : public ILayer
        {
            public:
            // Define constructor and destructor

            // Virtual fuctions
            // Forward propagation, propagate the values, from mIn to mOut
            virtual void Forward(size_t threadIdx) override;    
            // Backward propagation, propagate the gradients, from mDeltaIn to mDeltaOut
            virtual void BackProp(size_t threadIdx) override;
        }
    }
    ```
### 멀티쓰레드 지원
- 각 버퍼들은 NUM_THREAD(hardware쓰레드 수)길이의 std::vector컨테이너에 저장
- buffer[threadIdx]가 Forward, BackProp함수를 호출한 쓰레드에 할당된 버퍼
### AVX 지원
- 버퍼들은 depth를 마지막으로 indexing : 
$$ idx(x,y,d) == x * size^2 + y * size + d $$
- 버퍼에 접근할 때는 get__Idx함수를 사용 : 
    ```C++
    num = mOut[getOutIdx(x,y,z)];
    ```
## References
- [MACHINE LEARNING - 오일석](https://books.google.co.kr/books/about/%EA%B8%B0%EA%B3%84_%ED%95%99%EC%8A%B5.html?id=S_DwDwAAQBAJ&printsec=frontcover&source=kp_read_button&hl=ko&redir_esc=y#v=onepage&q&f=false)
- [Learning Multiple Layers of Features from Tiny Images - Alex Krizhevsky](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
## License
The BSD 3
