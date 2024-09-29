+++
title = 'Cuda runtime error: TensorRT destructor called out of main() in googletest'
date = 2024-09-29
draft = false
+++

When setting up a TensorRT inference engine in GoogleTest in a static variable, which is a pretty standard way, as following
```c++
class TRTModuleTest : public ::testing::Test {
 protected:
  // Per-test-suite set-up
  static void SetUpTestSuite() {
    EXPECT_NO_THROW(engine = std::make_unique<TRTModule>(
                        fs::path(__FILE__).parent_path() / "data/sum.trt",
                        std::vector{"input1"s, "input2"s},
                        std::vector{"output"s}));
  }

  static void TearDownTestSuite() { engine.release(); }

  inline static std::unique_ptr<TRTModule> engine;
  inline static std::array<float, 1024> input1;
  inline static std::array<float, 1024> input2;
  inline static std::array<float, 1024> output;
  inline static std::array<float, 1024> expected_output;
};  // class TRTModuleTest
```
The static variable engine has to be destroyed in TearDownTestSuite, otherwise, it will be destroyed out of main(), resulting into the following cuda error:
```
[09/29/2024-02:28:28] [E] [TRT] 1: [graphContext.h::~MyelinGraphContext::55] Error Code 1: Myelin (Error 4 setting context '0x563281141f90'.)
[09/29/2024-02:28:28] [E] [TRT] 1: [graphContext.h::~MyelinGraphContext::55] Error Code 1: Myelin (Error 4 setting context '0x563281141f90'.)
[09/29/2024-02:28:28] [E] [TRT] 1: [graphContext.h::~MyelinGraphContext::55] Error Code 1: Myelin (Error 4 setting context '0x563281141f90'.)
[09/29/2024-02:28:28] [E] [TRT] 1: [graphContext.h::~MyelinGraphContext::55] Error Code 1: Myelin (Error 4 setting context '0x563281141f90'.)
[09/29/2024-02:28:28] [E] [TRT] 1: [multiStreamContext.cpp::maybeDestroyAuxStream::264] Error Code 1: Cuda Runtime (driver shutting down)
[09/29/2024-02:28:28] [E] [TRT] 1: [defaultAllocator.cpp::deallocate::61] Error Code 1: Cuda Runtime (driver shutting down)
[09/29/2024-02:28:28] [E] [TRT] 1: [defaultAllocator.cpp::deallocate::61] Error Code 1: Cuda Runtime (driver shutting down)
[09/29/2024-02:28:28] [E] [TRT] 1: [cudaResources.cpp::~ScopedCudaStream::47] Error Code 1: Cuda Runtime (driver shutting down)
[09/29/2024-02:28:28] [E] [TRT] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (driver shutting down)
```

Ref: https://forums.developer.nvidia.com/t/tensorrt-engine-cannot-be-safely-deleted-in-deconstruction-of-a-global-or-static-instance/276391/6
