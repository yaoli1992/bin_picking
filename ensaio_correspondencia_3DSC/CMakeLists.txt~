cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(correspondencia_3DSC)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correspondencia_3DSC correspondencia_3DSC.cpp)
target_link_libraries (correspondencia_3DSC ${PCL_LIBRARIES})

add_executable (correspondencia_multivista_3DSC correspondencia_multivista_3DSC.cpp)
target_link_libraries (correspondencia_multivista_3DSC ${PCL_LIBRARIES})

#An optional "allclean" target

add_custom_target ( allclean
        COMMAND make clean
        COMMAND rm -rf
        "${PROJECT_SOURCE_DIR}/bin"
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/lib"
        CMakeCache.txt
        CMakeFiles
        cmake_install.cmake
        install_manifest.txt
        Makefile
        COMMAND echo "CLEARED ALL EXCEPT THE SOURCES!"
        )
