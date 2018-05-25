cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(my_virtual_scanner)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (virtual_scanner virtual_scanner.cpp)
target_link_libraries (virtual_scanner ${PCL_LIBRARIES})


#An optional "allclean" target ====================================================================

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
