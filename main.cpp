#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include "vehicles.h"
#include "core.h"

// store vehicle's pose and other simulation data
struct Pose
{
    Core::Position pos;
    Core::Velocity vel;
    int timestep;
};

int main()
{
    std::cout << "Simulating drone" << std::endl;

    sim::Drone drone;

    auto startTime = std::chrono::high_resolution_clock::now();

    const int TOTAL_SIM_TIMESTEPS = Core::SIMULATION_DURATION / Core::SIMULATION_TIMESTEP;

    // store simulation data
    std::vector<Pose> sim_data;

    for (int i = 0; i < TOTAL_SIM_TIMESTEPS; i++)
    {
        std::cout << i << "] Position: " << drone.pos << ", Velocity: " << drone.vel << std::endl;
        Pose pose;
        pose.pos = drone.pos;
        pose.vel = drone.vel;
        pose.timestep = i;

        sim_data.push_back(pose);

        // increment simulator by one timestep
        drone.next();
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    auto sim_duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "\nSimulation took " << sim_duration << " milliseconds to simulate " << TOTAL_SIM_TIMESTEPS << " timesteps (" << Core::SIMULATION_DURATION << " seconds)." << std::endl;

    // write simulation data to file so that it can be viewed using another application

    std::ios_base::sync_with_stdio(false);
    auto output_path = "sim.dat";
    auto myfile = std::fstream(output_path, std::ios::out | std::ios::binary);
    myfile.write((char*)&sim_data[0], sizeof(sim_data[0]) * sim_data.size());
    myfile.close();

    std::cout << "Written simulation data to " << output_path << std::endl;
}