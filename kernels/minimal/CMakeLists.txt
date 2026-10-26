cmake_minimum_required(VERSION 3.26.4)
project(quest_minimal LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
get_filename_component(QUEST_KERNEL_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_subdirectory("${QUEST_KERNEL_ROOT}/3rdparty/googletest"
                 "${CMAKE_BINARY_DIR}/googletest")
add_executable(test_max_possible "${QUEST_KERNEL_ROOT}/src/test/test_max_possible.cu")
target_include_directories(test_max_possible PRIVATE
    "${QUEST_KERNEL_ROOT}/include"
    "${QUEST_KERNEL_ROOT}/3rdparty/flashinfer/include"
    "${QUEST_KERNEL_ROOT}/src/include")
target_link_libraries(test_max_possible PRIVATE gtest_main)
enable_testing()
add_test(NAME max_possible COMMAND test_max_possible)

# A small C ABI avoids linking the old RAFT/PyTorch extension stack. Kernels
# remain the repository's original FP16 MHA kernels and partitioned decode.
add_library(quest_probe SHARED probe.cu)
target_include_directories(quest_probe PRIVATE
    "${QUEST_KERNEL_ROOT}/include"
    "${QUEST_KERNEL_ROOT}/3rdparty/flashinfer/include")
