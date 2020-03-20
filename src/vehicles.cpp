#include <iostream>
#include <iterator>
#include <random>

#include "vehicles.h"
#include "core.h"

sim::Drone::Drone()
{
    // initialise vehicle states
    pos = Eigen::Vector3f::Zero();
    vel = Eigen::Vector3f::Zero();
}

// sim::Drone::Drone()
// {
    
// }

/*
* Increment the simulation by one timestep
*/
void sim::Drone::next()
{
    // generate random change in velocity
    auto mean = 0.;
    auto stddev = 0.2 * MAX_VEL;
    
    auto vel_gen = std::bind(std::normal_distribution<double>{mean, stddev},
            std::mt19937(std::random_device{}()));

    // apply changes to vehicle's velocity
    vel(0) += vel_gen();
    vel(1) += vel_gen();
    vel(2) += vel_gen();

    // integrate vehicle's position
    pos += Core::SIMULATION_TIMESTEP * vel;
}