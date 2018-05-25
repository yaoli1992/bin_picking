cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(plane_extraction)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (modelviewpoint modelviewpoint.cpp)
target_link_libraries (modelviewpoint ${PCL_LIBRARIES})



add_executable (correspondence_grouping  correspondence_grouping.cpp)
target_link_libraries (correspondence_grouping ${PCL_LIBRARIES})
