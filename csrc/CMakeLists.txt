# Project name and cmake minimum requirement.
project(styleganr)
cmake_minimum_required(VERSION 3.16)

# We find a LibTorch installation trough the torch package.
# This is the best approach if we want to make sure we are
# targetting the same LibTorch version as used by torch.
execute_process (
    COMMAND Rscript -e "cat(torch::torch_install_path())"
    OUTPUT_VARIABLE TORCH_HOME
)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_HOME}")
message(STATUS "${TORCH_HOME}")

# Now that the prefix path is set we can tell cmake to go
# and find Torch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(APPLE)
    if(DEFINED ENV{CUDA})
        message(FATAL_ERROR, "CUDA not supported on macos")
    endif()
elseif(WIN32)
    if(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.1')
        message(FATAL_ERROR, "CUDA 10.1 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.2')
        add_compile_definitions(CUDA102)
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.1')
        add_compile_definitions(CUDA111)
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.3')
        add_compile_definitions(CUDA113)
    endif()
elseif(UNIX)
    if(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '9.2')
        message(FATAL_ERROR, "CUDA 9.2 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.1')
        message(FATAL_ERROR, "CUDA 10.1 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '10.2')
        add_compile_definitions(CUDA102)
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.0')
        message(FATAL_ERROR, "CUDA 11.0 is no longer supported")
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.1')
        add_compile_definitions(CUDA111)
    elseif(DEFINED ENV{CUDA} AND '$ENV{CUDA}' STREQUAL '11.3')
        add_compile_definitions(CUDA113)
    endif()
else()
    message(FATAL_ERROR, "OS not supported.")
endif()

# Here we tell CMake what are the source files of our package.
# If you want to separate your implementation in multiple files
# add their paths after `src/lltm.cpp`, the spearator is a simple
# space.
if(DEFINED ENV{CUDA})
  set(STYLEGANR_SRC src/bias_act.cpp
                    src/bias_act.cu
                    src/upfirdn2d.cpp
                    src/upfirdn2d.cu
                    src/filtered_lrelu.cpp
                    src/filtered_lrelu.cu
                    src/filtered_lrelu_ns.cu
                    src/filtered_lrelu_rd.cu
                    src/filtered_lrelu_wr.cu
                    src/types.cpp)
else()
  set(STYLEGANR_SRC src/bias_act_cpu.cpp
                    src/upfirdn2d_cpu.cpp
                    src/filtered_lrelu_cpu.cpp
                    src/types.cpp)
endif()

# On Windows we use module definition files to declare what are
# the exported functions from the library. It's similar to the
# Namespace file in the R ecossystem.
# We need to fill it manually as its used to link the Rcpp interface
# with the `csrc` library.
if(WIN32)
  set(STYLGANR_SRC ${STYLEGANR_SRC} src/styleganr.def)
endif()

# Tell cmake to build the shared library.
if(DEFINED ENV{CUDA})
  cuda_add_library(styleganr SHARED ${STYLEGANR_SRC})
else()
  add_library(styleganr SHARED ${STYLEGANR_SRC})
endif()
add_library(styleganr::library ALIAS styleganr)

# Tell cmake what are the include files.
target_include_directories(styleganr PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_HOME}/include
)

# Tell cmake the libraries we want to link to.
target_link_libraries(styleganr "${TORCH_LIBRARIES}")
set_property(TARGET styleganr PROPERTY CXX_STANDARD 17)

# Syncronize the headers and the def file with the Rcpp
# interface.
add_custom_command(TARGET styleganr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/styleganr.def ${PROJECT_SOURCE_DIR}/../inst/def/styleganr.def
    COMMENT "Copied def file to inst folder."
)

add_custom_command(TARGET styleganr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/styleganr/styleganr.h ${PROJECT_SOURCE_DIR}/../src/styleganr/styleganr.h
    COMMENT "Copied styleganr.h file to src folder."
)

# Set CPack related code to automatically generate installation bundles.
# The bundle name will have the same version as defined in the R DESCRIPTION
# file.
# The cpack configuration is used by the CI/CD workflows to create the pre-built
# binaries bundles and upload them to the GitHub Releases page.
set(CPACK_GENERATOR ZIP)
execute_process (
    COMMAND Rscript -e "cat(desc::description$new(file = '../../DESCRIPTION')$get('Version'))"
    OUTPUT_VARIABLE CPACK_PACKAGE_VERSION
)

if(DEFINED ENV{CUDA})
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cu$ENV{CUDA})
else()
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cpu)
endif()


install(TARGETS styleganr DESTINATION lib)
include(CPack)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../inst)
install(TARGETS styleganr DESTINATION lib)
