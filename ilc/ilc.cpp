#include "ilc.h"
#include <stdlib.h>
#include <string.h>

//#pragma warning(disable:4351) // Do not warn about zero initialization
ILC::ILC(float fii, float gamma, float alpha) :
    phi(phi),
    gamma(gamma),
    alpha(alpha),
    is_enabled(false),
    iq_buffer(),
    error_buffer(),
    idx(0),
    is_first_iteration(true),
    ramp_steps(2000) // 2000 * 500us = 1s
{
}

// Toggle ILC on / off.
// Buffers are cleared, because the operating point may change
void ILC::toggle() {   
    if (is_enabled) {
        // Disable:
        clearBuffers();
        is_enabled = false;
        is_first_iteration = true;
    }
    else {
        is_enabled = true;
    }
}

// Fill buffers with zeroes
void ILC::clearBuffers() {
    memset(&iq_buffer, 0, sizeof(iq_buffer));
    memset(&error_buffer, 0, sizeof(error_buffer));
}

float ILC::clamp(float value, float lower_limit, float upper_limit) {
    if (value < lower_limit) {
        return lower_limit;
    }
    else if (value > upper_limit) {
        return upper_limit;
    }
    return value;
}

// The update algorithm
float ILC::computeCompensation(float reference, float actual) {
    float iq_ref = 0.0;
    float error = reference - actual;

    // Learn by using the following P-type learning law
    iq_ref = (1 - alpha) * iq_buffer[idx] + phi * error_buffer[idx] + gamma * error;

    iq_buffer[idx] = iq_ref;
    error_buffer[idx] = error;
    return iq_ref;
}

// Handle the unlinearity that occurs on full rotations with [0, 1] angle.
// Returns the corrected distance. Works on both directions (clockwise and counter clockwise).
// Uses half circle to decide how to calculate the distance, so that unlinearity point is not crossed.
uint16_t ILC::getDistanceBetween(uint16_t start_idx, uint16_t end_idx) {
    int16 e = end_idx; int16 s = start_idx;
    // Second circle half, use tricks to avoid possible crossing.
    if (end_idx > BUFFER_SIZE / 2) {
        e = BUFFER_SIZE - end_idx;
    }
    if (start_idx > BUFFER_SIZE / 2) {
        s = BUFFER_SIZE - start_idx;
        return abs(e - s);
    }
    if (e != end_idx && s != start_idx) {
        return abs(BUFFER_SIZE - e - s);
    }

    // First cicrle half, safe to calculate 'normally'.
    return abs(e - s);
}

// Interpolates data between two given array indices. Handles rotations, so it
// is possible to give: start_idx > end_idx. In such case, the function breaks
// the problem into two subproblems and recursively calls itself.
void ILC::interpolate(uint16_t start_idx, uint16_t end_idx, float array[BUFFER_SIZE]) {
    uint16_t forward_steps = getDistanceBetween(start_idx, end_idx);

    // Normal case:
    if (end_idx + forward_steps < BUFFER_SIZE || start_idx + forward_steps < BUFFER_SIZE) {
        float step = (array[end_idx] - array[start_idx]) / (end_idx - start_idx);
        for (uint16_t i = start_idx + 1; i < end_idx; i++) {
            array[i] = array[start_idx] + (i - start_idx) * step;
        }  
    }
    // Has rotated full cycle
    // Break into two linear cases by calculating first and last array values
    else {
        float step = (array[end_idx] - array[start_idx]) / forward_steps;
        array[BUFFER_LAST_IDX] = array[start_idx] + (BUFFER_LAST_IDX - start_idx) * step;
        array[0] = array[BUFFER_LAST_IDX] + step;
        
        // If more steps were skipped, now it is easy to interpolate
        interpolate(start_idx, BUFFER_LAST_IDX, array);
        interpolate(0, end_idx, array);
    }
}

// Function updates index accordingly. The memory buffer must hold samples for single period.
// This function increments the index so that the constant size memory will suffice.
// rotor_angle: [0.0, 1.0]
boolean ILC::updateBufferIndex(float rotor_angle) {
    uint16_t previous_step_angle = idx;

    // Shouldn't clamp. Just to make sure that noise doesn't cause memory read errors.
    rotor_angle = clamp(rotor_angle, 0.0, 1.0);

    uint16_t current_step_angle = idx = (uint16_t)(rotor_angle * BUFFER_LAST_IDX);
    uint16_t steps_forward = getDistanceBetween(previous_step_angle, current_step_angle);
    
    // If steps were skipped, then interpolate these. Ideally, never executed.
    // First angle difference tells nothing: avoid interpolating the whole array.
    if (steps_forward > 1 && !is_first_iteration) {
         interpolate(previous_step_angle, current_step_angle, iq_buffer);
         interpolate(previous_step_angle, current_step_angle, error_buffer);
    }
    is_first_iteration = false;

    return false;
}

// Function handles the ILC state management and returns the desired compensation term.
float ILC::getCompensationTerm(float reference, float actual, float rotor_elec_angle) {
    static float compensation = 0.0;
    static uint16_t step_idx = ramp_steps; 

    // Normal mode (ILC enabled)
    if (is_enabled) {
        compensation = computeCompensation(reference, actual);
        updateBufferIndex(rotor_elec_angle);
    }
    // Disable ILC: ramp down
    else if (abs(compensation) > 0.01) {
        compensation = (step_idx * compensation) / ramp_steps;
        step_idx--;
    }
    // Fully disabled: do nothing
    else {
        compensation = 0.0;
        step_idx = ramp_steps;
    }

    return compensation;
}
