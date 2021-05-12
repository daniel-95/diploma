cmake_minimum_required(VERSION 3.10)

project(diploma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(diploma
        track.hpp
        track.cpp
	declarations.hpp
	cuda_surf.cpp			# fast, robust enough, loses features gradually
	cpu_surf.cpp			# kinda fast, robust enough, loses features gradually
	opencl_surf.cpp			# fast, robust enough, loses features gradually
	cpu_orb.cpp			# not bad, good tracking, no lost features
	opencl_orb.cpp			# very fast, good tracking, no lost features
	cuda_orb.cpp			# very fast and tracks all 30 features, not a single one is dropped
	cpu_brisk.cpp			# really fast for cpu, loses features
	cpu_harris_brief.cpp		# unstable, may find 1 feature in one frame and 100 in subsequent
	cpu_harris_freak.cpp		# unstable, may find 1 feature in one frame and 100 in subsequent
	cpu_shi_tomasi_brief.cpp	# fast, loses features gradyally, faster than surf (check)
	cpu_shi_tomasi_freak.cpp	# fast, loses features gradyally, faster than surf (check)
	cpu_star_brief.cpp		# slow detecting, average feature loss
	cpu_star_freak.cpp		# slow detecting, average feature loss
	cpu_fast_freak.cpp		# the fastest one, relatively stable, small feature loss, potential candidate
	diploma.hpp
        diploma.cpp
)

target_link_libraries(diploma PRIVATE ${OpenCV_LIBS})
