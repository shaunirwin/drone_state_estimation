#ifndef CORE_H
#define CORE_H

#include <Eigen/Core>

namespace Core
{
    typedef Eigen::Vector3f Position;   // vehicle position states
    typedef Eigen::Vector3f Velocity;   // vehicle velocity states
    
    const double SIMULATION_TIMESTEP = 0.02;    // timestep of the simulation [s]
};

#endif