function(cudao_add_example name)
    add_executable(${name} ${ARGN})

target_link_libraries(${name} PRIVATE
cuDAO::single
)

add_dependencies(${name}
cudao_single_header
)

set_target_properties(${name} PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)

endfunction()

cudao_add_example(hello_cudao hello_cudao.cu)
