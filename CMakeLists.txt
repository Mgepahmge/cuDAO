cmake_minimum_required(VERSION 3.25)

project(cuDAO
    VERSION 0.1.0
    DESCRIPTION "Dependency-Aware Ordering runtime for concurrent CUDA memory access"
    HOMEPAGE_URL "https://github.com/Mgepahmge/cuDAO"
    LANGUAGES CXX CUDA
)

# ── C++ / CUDA standards ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Guard against in-source builds ─────────────────────────────────────────
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed.\n"
        "Please create a separate build directory and run cmake from there.")
endif()

# ── Default build type ──────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ── Options ─────────────────────────────────────────────────────────────────
option(CUDAO_BUILD_TESTS    "Build tests"    ${PROJECT_IS_TOP_LEVEL})
option(CUDAO_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(CUDAO_ENABLE_ASAN    "Enable AddressSanitizer (Debug only)" OFF)

# ── CUDA ────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Require Volta or newer (sm_70+) for cuStreamWaitValue64 support
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
endif()

# ── Header-only interface target ────────────────────────────────────────────
add_library(cuDAO INTERFACE)
add_library(cuDAO::cuDAO ALIAS cuDAO)

target_include_directories(cuDAO INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cuDAO INTERFACE
    CUDA::cuda_driver   # Driver API
    CUDA::cudart        # cudaLaunchHostFunc etc.
)

target_compile_features(cuDAO INTERFACE cxx_std_17)

# ── Compiler warnings ───────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(cuDAO INTERFACE
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>>
    )
elseif(MSVC)
    target_compile_options(cuDAO INTERFACE
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->>
    )
endif()

# ── AddressSanitizer ────────────────────────────────────────────────────────
if(CUDAO_ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuDAO INTERFACE -fsanitize=address)
    target_link_options(cuDAO INTERFACE -fsanitize=address)
endif()

# ── Subdirectories ──────────────────────────────────────────────────────────
if(CUDAO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(CUDAO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Installation ────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install cmake integration (allows find_package(cuDAO))
install(TARGETS cuDAO
    EXPORT cuDAOTargets
)

install(EXPORT cuDAOTargets
    FILE        cuDAOTargets.cmake
    NAMESPACE   cuDAO::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuDAOConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO
)
