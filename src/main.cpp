#include <iostream>
#include <string>
#include "vehicles.h"

int main()
{
    std::cout << "Simulating drone" << std::endl;

    sim::Drone drone;

    for (int i = 0; i < 20; i++)
    {
        // increment simulator by one timestep
        drone.next();

        std::cout << i << "] Position: " << drone.pos << ", Velocity: " << drone.vel << std::endl;
    }
}