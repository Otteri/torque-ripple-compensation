#include "pulsations.h"

#define PI 3.1415926535897932384626433832795

Pulsator::Pulsator() {
    // Define harmonics that shall be generated
    harmonics.insert(Harmonic{ 1, 0.02 });
    harmonics.insert(Harmonic{ 2, 0.0030 });
    harmonics.insert(Harmonic{ 6, 0.006 });
    harmonics.insert(Harmonic{ 12, 0.00227});
    harmonics.insert(Harmonic{ 18, 0.00039 });
};

const double Pulsator::convertAngle(const double angle) {
    return (2.0 * PI * angle); // convert from [0, 1] to [0, 2PI]
}

const double Pulsator::getPulsations(const double angle) {
    double torque = 0.0f;
    for (const auto& harmonic : harmonics) {
        torque += harmonic.magnitude * cos(harmonic.order * angle);
    }
    return torque;
}

const double Pulsator::getSample(double rotor_angle) {
    double converted_angle = convertAngle(rotor_angle);
    return getPulsations(converted_angle);
}
