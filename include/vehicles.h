#ifndef VEHICLES_H
#define VEHICLES_H

// This file is for simulating the movement of different vehicles

#include "core.h"

namespace sim
{
    class Drone
    {
        public:
            Drone();
            Drone(Core::Position pos, Core::Velocity vel);

            // vehicle states
            
            Core::Position pos; 
            Core::Velocity vel;
            const float MAX_VEL = 0.5;  // [m/s]

            // update states
            void next();
    };
};

#endif