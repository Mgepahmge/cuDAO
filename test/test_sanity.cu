#include <cuda.h>
#include <gtest/gtest.h>
#include <cuDAO.h>

TEST(cuDAO, VersionDefined) {
    EXPECT_GT(CUDAO_VERSION, 0);
}

TEST(cuDAO, CudaDriverAvailable) {
    CUresult res = cuInit(0);
    EXPECT_EQ(res, CUDA_SUCCESS);
}
