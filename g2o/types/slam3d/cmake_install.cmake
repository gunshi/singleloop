# Install script for directory: /home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/libg2o_types_slam3d_d.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/lib" TYPE STATIC_LIBRARY FILES "/home/gunshi/Downloads/MultiRobot/cair_online_isam/lib/libg2o_types_slam3d_d.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/g2o/types/slam3d/dquat2mat.h;/usr/local/include/g2o/types/slam3d/edge_pointxyz.h;/usr/local/include/g2o/types/slam3d/edge_se3_prior.h;/usr/local/include/g2o/types/slam3d/edge_se3_pointxyz_depth.h;/usr/local/include/g2o/types/slam3d/types_slam3d.h;/usr/local/include/g2o/types/slam3d/edge_se3_pointxyz_disparity.h;/usr/local/include/g2o/types/slam3d/isometry3d_mappings.h;/usr/local/include/g2o/types/slam3d/edge_se3_offset.h;/usr/local/include/g2o/types/slam3d/vertex_pointxyz.h;/usr/local/include/g2o/types/slam3d/se3quat.h;/usr/local/include/g2o/types/slam3d/edge_se3_lotsofxyz.h;/usr/local/include/g2o/types/slam3d/vertex_se3.h;/usr/local/include/g2o/types/slam3d/se3_ops.h;/usr/local/include/g2o/types/slam3d/edge_se3_pointxyz.h;/usr/local/include/g2o/types/slam3d/isometry3d_gradients.h;/usr/local/include/g2o/types/slam3d/parameter_camera.h;/usr/local/include/g2o/types/slam3d/parameter_stereo_camera.h;/usr/local/include/g2o/types/slam3d/g2o_types_slam3d_api.h;/usr/local/include/g2o/types/slam3d/edge_se3.h;/usr/local/include/g2o/types/slam3d/parameter_se3_offset.h;/usr/local/include/g2o/types/slam3d/se3_ops.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/include/g2o/types/slam3d" TYPE FILE FILES
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/dquat2mat.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_pointxyz.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_prior.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_pointxyz_depth.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/types_slam3d.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_pointxyz_disparity.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/isometry3d_mappings.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_offset.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/vertex_pointxyz.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/se3quat.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_lotsofxyz.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/vertex_se3.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/se3_ops.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3_pointxyz.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/isometry3d_gradients.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/parameter_camera.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/parameter_stereo_camera.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/g2o_types_slam3d_api.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/edge_se3.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/parameter_se3_offset.h"
    "/home/gunshi/Downloads/MultiRobot/cair_online_isam/g2o/types/slam3d/se3_ops.hpp"
    )
endif()

