#include <iostream>
#include <string>
#include <chrono>
#include "vehicles.h"
#include "core.h"

int main()
{
    std::cout << "Simulating drone" << std::endl;

    sim::Drone drone;

    auto startTime = std::chrono::high_resolution_clock::now();

    const int TOTAL_SIM_TIMESTEPS = Core::SIMULATION_DURATION / Core::SIMULATION_TIMESTEP;

    for (int i = 0; i < TOTAL_SIM_TIMESTEPS; i++)
    {
        // increment simulator by one timestep
        drone.next();

        std::cout << i << "] Position: " << drone.pos << ", Velocity: " << drone.vel << std::endl;
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    auto sim_duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "\nSimulation took " << sim_duration << " milliseconds to simulate " << TOTAL_SIM_TIMESTEPS << " timesteps (" << Core::SIMULATION_DURATION << " seconds)." << std::endl;
}