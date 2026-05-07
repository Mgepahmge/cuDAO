find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

function(cudao_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cuDAO::cuDAO GTest::gtest_main)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

cudao_add_test(test_sanity test_sanity.cu)
