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

# ── Guard against in-source builds ──────────────────────────────────────────

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
            "In-source builds are not allowed.\n"
            "Please create a separate build directory and run cmake from there."
    )
endif()

# ── Default build type ──────────────────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug"
            "Release"
            "RelWithDebInfo"
            "MinSizeRel"
    )
endif()

# ── Options ─────────────────────────────────────────────────────────────────

option(CUDAO_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(CUDAO_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(CUDAO_ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)
option(CUDAO_USE_LEAST_TASK_POLICY "Use least-task stream scheduler instead of round-robin" OFF)

# ── Dependencies ────────────────────────────────────────────────────────────

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Require Volta or newer (sm_70+) for cuStreamWaitValue64 support.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
endif()

# ── Shared interface settings ───────────────────────────────────────────────

function(cudao_apply_interface_settings target_name)
    target_link_libraries(${target_name} INTERFACE
            CUDA::cuda_driver
            CUDA::cudart
    )

target_compile_features(${target_name} INTERFACE cxx_std_17)

if(CUDAO_USE_LEAST_TASK_POLICY)
target_compile_definitions(${target_name} INTERFACE
CUDA_DAO_USE_LEAST_TASK_POLICY
)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
target_compile_options(${target_name} INTERFACE
$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>>
)
elseif(MSVC)
target_compile_options(${target_name} INTERFACE
$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->>
)
endif()

if(CUDAO_ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_options(${target_name} INTERFACE
$<BUILD_INTERFACE:-fsanitize=address>
)
target_link_options(${target_name} INTERFACE
$<BUILD_INTERFACE:-fsanitize=address>
)
endif()

endfunction()

# ── Development header-only interface target ────────────────────────────────

#

# In the source tree, cuDAO::cuDAO points to the modular development headers.

# After installation, cuDAO::cuDAO points to the installed single-header cuDAO.cuh.

add_library(cuDAO INTERFACE)
add_library(cuDAO::cuDAO ALIAS cuDAO)

target_include_directories(cuDAO INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

cudao_apply_interface_settings(cuDAO)

# ── Single-header generation ────────────────────────────────────────────────

set(CUDAO_SINGLE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/single_include")
set(CUDAO_SINGLE_HEADER "${CUDAO_SINGLE_INCLUDE_DIR}/cuDAO.cuh")

set(CUDAO_AMALGAMATE_ENTRY "${CMAKE_CURRENT_SOURCE_DIR}/include/cuDAO.cuh")

file(GLOB_RECURSE CUDAO_AMALGAMATE_SECTION_HEADERS CONFIGURE_DEPENDS
"${CMAKE_CURRENT_SOURCE_DIR}/include/cuDAO/*.cuh"
"${CMAKE_CURRENT_SOURCE_DIR}/include/cuDAO/*.h"
)

set(CUDAO_AMALGAMATE_INPUTS
"${CUDAO_AMALGAMATE_ENTRY}"
${CUDAO_AMALGAMATE_SECTION_HEADERS}
)

add_custom_command(
OUTPUT "${CUDAO_SINGLE_HEADER}"
COMMAND "${CMAKE_COMMAND}" -E make_directory "${CUDAO_SINGLE_INCLUDE_DIR}"
COMMAND "${Python3_EXECUTABLE}"
"${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
--root "${CMAKE_CURRENT_SOURCE_DIR}/include"
--input "${CUDAO_AMALGAMATE_ENTRY}"
--output "${CUDAO_SINGLE_HEADER}"
DEPENDS
"${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
${CUDAO_AMALGAMATE_INPUTS}
COMMENT "Generating single-header cuDAO.cuh"
VERBATIM
)

add_custom_target(cudao_single_header ALL
DEPENDS "${CUDAO_SINGLE_HEADER}"
)

# Build-tree-only target used by tests and examples.

# This ensures CI compiles against the generated single header, not the modular headers.

add_library(cuDAO_single INTERFACE)
add_library(cuDAO::single ALIAS cuDAO_single)

add_dependencies(cuDAO_single cudao_single_header)

target_include_directories(cuDAO_single INTERFACE
$<BUILD_INTERFACE:${CUDAO_SINGLE_INCLUDE_DIR}>
)

cudao_apply_interface_settings(cuDAO_single)

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

# Install only the generated single header as the public installed header.

install(FILES "${CUDAO_SINGLE_HEADER}"
DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Export the canonical package target.

install(TARGETS cuDAO
EXPORT cuDAOTargets
)

install(EXPORT cuDAOTargets
FILE cuDAOTargets.cmake
NAMESPACE cuDAO::
DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO"
)

write_basic_package_version_file(
"${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfigVersion.cmake"
VERSION ${PROJECT_VERSION}
COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuDAOConfig.cmake.in"
"${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfig.cmake"
INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO"
)

install(FILES
"${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfig.cmake"
"${CMAKE_CURRENT_BINARY_DIR}/cuDAOConfigVersion.cmake"
DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuDAO"
)
