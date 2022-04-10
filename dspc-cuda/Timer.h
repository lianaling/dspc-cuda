#pragma once

#include <chrono>
#include <thread>
#include <iostream>

struct Timer
{
	const char* operation_name;
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::duration<float> duration;

	Timer() { start = std::chrono::high_resolution_clock::now(); }
	Timer(const char* operation_name)
	{
		start = std::chrono::high_resolution_clock::now();
		this->operation_name = operation_name;
	}

	~Timer()
	{
		auto end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		float ms = duration.count() * 1000.0f;
		const char* to_print = operation_name == nullptr ? "An unknown operation" : operation_name;
		std::cout << "\033[1;31m[TIMER] " << to_print << " took " << ms << "ms\033[0m\n";
	}
};