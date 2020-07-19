#ifndef TORQUEPULSATOR_H
#define TORQUEPULSATOR_H
#include <set>

struct Harmonic {
    int order;
    double magnitude;

    bool operator <(const Harmonic& harmonic) const {
        return order < harmonic.order;
    }
};

class Pulsator {

public:
    Pulsator();
    const double getSample(double rotor_angle);

private:
    const double getPulsations(const double rotor_angle);
    const double convertAngle(const double angle_mech);

    std::set<Harmonic> harmonics;
};

#endif
