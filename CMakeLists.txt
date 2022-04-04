cmake_minimum_required(VERSION 3.10)

project(paddle-ascend910 CXX C)

option(WITH_TESTING    "compile plugin with unit testing"        OFF)
option(WITH_KERNELS    "build custom kernels"                    OFF)

set(PLUGIN_NAME        "paddle_ascend910")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_PLUGIN_DIR  "/opt/conda/lib/python3.7/site-packages/paddle-plugins/")

set(PADDLE_INC_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")

set(NPU_INC_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/")
set(NPU_LIB_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/")

set(PLUGIN_SRCS runtime/runtime.cc)
set(INCLUDE_DIR ${PADDLE_INC_DIR} ${NPU_INC_DIR})

if (WITH_KERNELS)
  set(BOOST_INC_DIR      "${PADDLE_ROOT}/build/third_party/boost/src/extern_boost")
  set(EIGEN_INC_DIR      "${PADDLE_ROOT}/build/third_party/eigen3/src/extern_eigen3/")
  set(GFLAGS_INC_DIR     "${PADDLE_ROOT}/build/third_party/install/gflags/include")
  set(GLOG_INC_DIR       "${PADDLE_ROOT}/build/third_party/install/glog/include")
  set(THREAD_INC_DIR     "${PADDLE_ROOT}/build/third_party/threadpool/src/extern_threadpool")

  set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${EIGEN_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${THREAD_INC_DIR})
  set(RUNTIME_INC_DIR ${CMAKE_SOURCE_DIR}/runtime)

  file(GLOB PLUGIN_KERNEL_FILES RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
  list(APPEND PLUGIN_SRCS ${PLUGIN_KERNEL_FILES})

  list(APPEND INCLUDE_DIR ${THIRD_PARTY_INC_DIR} ${RUNTIME_INC_DIR})

  find_file(CORE_AVX_FOUND core_avx.so ${PADDLE_LIB_DIR})
  if (CORE_AVX_FOUND)
    set(CORE_LIB ":core_avx.so")
  else()
    set(CORE_LIB ":core_noavx.so")
  endif()

  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)   # for NPU compile
  add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext
  add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid seperate
endif()

include_directories(${INCLUDE_DIR})
link_directories(${PADDLE_LIB_DIR} ${NPU_LIB_DIR})

####### build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
target_link_libraries(${PLUGIN_NAME} PRIVATE ascendcl)
target_link_libraries(${PLUGIN_NAME} PRIVATE acl_op_compiler)
if (WITH_KERNELS)
 target_link_libraries(${PLUGIN_NAME} PRIVATE ${CORE_LIB})  # special name
endif()

install(TARGETS ${PLUGIN_NAME} 
PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ 
DESTINATION "${PADDLE_PLUGIN_DIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

####### packing wheel package
add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

# if (WITH_TESTING)

# include(ExternalProject)

# SET(PADDLE_PREFIX_DIR  ${CMAKE_BINARY_DIR}/third-party/paddle)
# SET(PADDLE_BINARY_DIR  ${CMAKE_BINARY_DIR}/Paddle)
# set(PADDLE_REPOSITORY  "https://github.com/PaddlePaddle/Paddle.git")
# set(PADDLES_TAG        "develop")
# set(PADDLE_CXX_FLAGS    ${CMAKE_CXX_FLAGS} -std=c++11 -pthread)

# separate_arguments(PADDLE_BUILD_COMMAND)
# separate_arguments(PADDLE_CMAKE_ARGS)

# ExternalProject_Add(Paddle
#     GIT_REPOSITORY     ${PADDLE_REPOSITORY}
#     GIT_TAG            ${PADDLE_TAG}
#     GIT_SHALLOW        TRUE
#     GIT_PROGRESS       TRUE
#     PREFIX             ${PADDLE_PREFIX_DIR}
#     BINARY_DIR         ${PADDLE_BINARY_DIR}
#     BUILD_COMMAND      ${PADDLE_BUILD_COMMAND}
#     UPDATE_COMMAND     ""
#     PATCH_COMMAND      git reset --hard && git apply ${CMAKE_SOURCE_DIR}/plugin.patch
#     INSTALL_COMMAND    ""
#     CMAKE_ARGS         ${PADDLE_CMAKE_ARGS} -DCMAKE_CXX_FLAGS=${PADDLE_CXX_FLAGS}
# )

# endif()

# cmake .. -DWITH_TESTING=ON -DPADDLE_CMAKE_ARGS="-DPY_VERSION=3 -DPYTHON_EXECUTABLE=$(which python3) -DWITH_ARM=ON -DWITH_ASCEND=OFF -DWITH_ASCEND_CL=ON \
# -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF   \
# -DPYTHON_INCLUDE_DIR=$(python3 -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')" \
# -DPADDLE_BUILD_COMMAND="make TARGET=ARMV8 -j32"
