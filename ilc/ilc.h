#ifndef ILC_H
#define ILC_H

// Angle-based Iterative Learning Control (ILC)
// Buffer holds samples for one electrical rotation.
// Having larger buffer size can improve accuracy and performance.
// Define in order to avoid use of dynamic memory.
#define BUFFER_SIZE 750
#define BUFFER_LAST_IDX (BUFFER_SIZE-1)

class ILC {
public:
    ILC(float fii, float gamma, float alpha);
    float getCompensationTerm(float reference, float actual, float rotor_angle);
    void toggle();

    float phi;         // ILC I-gain
    float gamma;       // ILC P-gain
    float alpha;       // Forgetting coefficient
    bool is_enabled;   // current module state

private:
    float computeCompensation(float reference, float actual);
    float clamp(float value, float lower_limit, float upper_limit);
    void clearBuffers();

    // Buffer handling:
    bool updateBufferIndex(float rotor_angle);
    void interpolate(uint16_t start_idx, uint16_t end_idx, float array[BUFFER_SIZE]);
    uint16_t getDistanceBetween(uint16_t, uint16_t);

    float iq_buffer[BUFFER_SIZE];    // Memory for correction terms
    float error_buffer[BUFFER_SIZE]; // Memory for error terms
    uint16_t idx;                    // Index for accessing the above buffers
    bool is_first_iteration;         // Due to feedback, the first iteration is not realiable.
    uint16_t ramp_steps;             // How fast the compensation term should be ramped down?
};

#endif
