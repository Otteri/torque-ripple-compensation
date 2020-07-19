#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "qlearning.h"
#include "qtable.h"

#define INIT_MAX = 9999;

Qtable::Qtable(float alpha, float gamma, float ek) :
    is_learning(false),
    reward(float(0.0)),
    epsilon(float(1.0)),
    alpha(alpha),
    gamma(gamma),
    ek(ek),
    lambda(32.0),
    ripple_min(float(INIT_MAX)),    // set such initial values
    rotation_min(float(INIT_MAX)),  // values, that leave room
    rotation_max(float(-INIT_MAX)), // for improvement
    train_iterations(300000),
    cumulative_reward(float(-INIT_MAX)),
    max_average_reward(float(-INIT_MAX)),
    auto_zeta_search(true),
    N(500),
    save(false)
{
}

// Reset initial state
void Qtable::resetState() {
    is_learning = false;
    iteration_number = 0;
    epsilon = 1.0;
    cumulative_reward = float(-INIT_MAX);
    max_average_reward = float(-INIT_MAX);
}

// Fill table with zeroes
void Qtable::clearTable() {
    memset(qtable_ptr, 0, sizeof(*qtable_ptr) * ACTION_NUM * ANGLE_NUM);
}

// Since the size is already known at compile time, we can just use memcpy
void Qtable::copyWeights(float* src_table, float* dest_table) {
    memcpy(dest_table, src_table, sizeof(float) * ANGLE_NUM * ACTION_NUM);
}

void Qtable::updateTargetTable(bool has_improved) {
    if (has_improved) {
        copyWeights(qtable_ptr, qtable_target_ptr);
    }
}

// Loads the table, which must have the same structure as define in header.
bool Qtable::loadTable() {
    p_qtable = &qtable_weights; // used only for debugging
    qtable_ptr = &qtable_weights[0][0];
    qtable_target_ptr = &qtable_target_weights[0][0];
    if (qtable_ptr != NULL) {
        linspaceAngles(angles, 0, 1.0);
        linspaceActions(actions, float(-T_MAX), float(T_MAX));
        return true; // load succesful

    }
    return false; // load failed
}

void Qtable::linspaceAngles(float result[], float min, float max) {
    float step = (max - min) / (ANGLE_NUM - 1);
    for (uint16_t i = 0; i < ANGLE_NUM; i++) {
        result[i] = min + i * step;
    }
}

void Qtable::linspaceActions(float result[], float min, float max) {
    float step = (max - min) / (ACTION_NUM - 1);
    for (uint16_t i = 0; i < ACTION_NUM; i++) {
        result[i] = min + i * step;
    }
}

// Iterates through the given values and then returns the maximum value and argmax
// p_weights: pointer to the first relevant value.
struct Maximum Qtable::findMax(float* values_ptr) {
    struct Maximum max = {0, values_ptr[0]};
    for (uint16_t i = 0; i < ACTION_NUM; ++i) {
        if (values_ptr[i] > max.value) {
            max.value = values_ptr[i];
            max.idx = i;
        }
    }
    return max;
}

// A helper function to get the index, which produces array value closer to the target
uint16_t Qtable::getCloserIdx(float arr[], uint16_t idx1, uint16_t idx2, float target) {
    if (fabs(arr[idx1] - target) < fabs(arr[idx2] - target)) {
        return idx1;
    }
    return idx2;
}

// Returns element closest to target in arr[]
// n: number of array items
// target: target value
uint16_t Qtable::findClosestIdx(float arr[], uint16_t n, float target)
{
    // Corner cases 
    if (target <= arr[0]) {
        return 0;
    }
    if (target >= arr[n - 1]) {
        return (n - 1);
    }

    // Binary search 
    uint16_t i = 0, j = n, mid = 0;
    while (i < j) {
        mid = (i + j) / 2;

        if (arr[mid] == target) {
            return mid;
        }

        // If target is less than array element, then search in left
        if (target < arr[mid]) {

            // If target is greater than previous 
            // to mid, return closest of two 
            if (mid > 0 && target > arr[mid - 1])
                return getCloserIdx(arr, mid - 1, mid, target);

            // Repeat for left half
            j = mid;
        }

        // If target is greater than mid 
        else {
            if (mid < n - 1 && target < arr[mid + 1])
                return getCloserIdx(arr, mid, mid + 1, target);
            // update i 
            i = mid + 1;
        }
    }

    // Only single element left after search 
    return mid;
}

// Get random number between 0.0 and 1.0
float Qtable::getRandom() {
    return (float)rand() / (float)RAND_MAX;
}

// Get random integer between the provided range
uint16_t Qtable::getRandomInteger(uint16_t min, uint16_t max) {
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

float Qtable::getReward(float actual, float reference) {
    static float actual_prev = actual; // add to class, shouldn't be static.

    // The second part is much more important, hence the multiplier.
    float cost = fabs(actual - reference) + lambda * fabs(actual - actual_prev);

    actual_prev = actual;
    return -cost; // translate cost to reward
}

// Get the best known action
float Qtable::getBestAction(float current_angle) {
    uint16_t n = sizeof(angles) / sizeof(angles[0]);
    uint16_t angle_idx = findClosestIdx(angles, n, current_angle);
    uint16_t best_action_idx = findMax(qtable_ptr + angle_idx * ACTION_NUM).idx;
    return  actions[best_action_idx];
}

// Gathers minimum and maximum actual values
// When asked to start over, computes ripple by using the min & max,
// and determines if any improvement has happened.
bool Qtable::hasImproved(float actual, bool start_again) {
    bool has_improved = false;
    if (start_again) {
        // Compute ripple
        float ripple = rotation_max - rotation_min;
        if (ripple < ripple_min) {
            ripple_min = ripple;
            has_improved = true;
        }

        // Reset
        rotation_min = INIT_MAX;
        rotation_max = -INIT_MAX;
    }

    return has_improved;
}

// Calculates running average over N-values.
// Average can be used to detemine if agent has improved.
bool Qtable::updateRewardAverage(float reward) {
    bool has_improved = false;
    cumulative_reward += reward;
    if (iteration_number % ANGLE_NUM == 0) {
        average_reward = cumulative_reward / ANGLE_NUM;
        cumulative_reward = 0;
        if (average_reward > max_average_reward) {
            max_average_reward = average_reward;
            has_improved = true;
        }
    }
    return has_improved;
}

// Updates class state
void Qtable::update(float actual, uint16_t angle_idx) {
    // Checks for massive index jumps, which indicate full electrical periods
    if (abs(angle_idx - last_angle_idx) > (ANGLE_NUM / 2.0)) {
        average_reward = cumulative_reward / ANGLE_NUM;
        if (average_reward > max_average_reward && train_iterations > 10000) {
            max_average_reward = average_reward;
            copyWeights(qtable_ptr, qtable_target_ptr);
        }
        cumulative_reward = 0;
        hasImproved(actual, true); // for monitoring
    }

    if (actual < rotation_min) {
        rotation_min = actual;
    }
    if (actual > rotation_max) {
        rotation_max = actual;
    }

    // forward step (next state)
    if (angle_idx != last_angle_idx) {
        last_angle_idx = angle_idx;
        cumulative_reward += reward;
    }
    hasFinishedTraining();
}

// Check if learning is done and act accordingly
void Qtable::hasFinishedTraining() {
    if (iteration_number >= train_iterations) {
        is_learning = false;
        copyWeights(qtable_target_ptr, qtable_ptr); // take the best weights into use
        //dumpTable();
    }
    iteration_number++;
}

void Qtable::dumpTable() {
   FILE* qfile = fopen("qtable.txt", "w");
   fprintf(qfile, "float qtable_weights[ANGLE_NUM][ACTION_NUM] = {\n");
   for (uint16_t i = 0; i < ANGLE_NUM; i++) {
       for (uint16_t j = 0; j < ACTION_NUM; j++) {
           float val = *(qtable_ptr + (i * ACTION_NUM) + j);
           fprintf(qfile, "%f, ", val);
       }
       fprintf(qfile, "\n");
   }
   fprintf(qfile, "};\n");
   fclose(qfile);
}

float Qtable::train(float current_angle, float actual, float reference) {

    // Keep exploring some times + avoid problems coming from iteration rollover.
    epsilon = epsilon <= 0.01 ? float(0.01) : epsilon = ek / (ek + iteration_number);   

    // Discretize the angle
    uint16_t angle_idx = findClosestIdx(angles, ANGLE_NUM, current_angle);

    // If the state has not changed, then we can just return the previous action
    if (angle_idx == last_angle_idx) {
        update(actual, angle_idx);
        return action;
    }

    // Define helper pointers for accessing the table
    float* Q_prev_ptr = (qtable_ptr + (last_angle_idx * ACTION_NUM) + last_action_idx); // single weight
    float* Q_row_ptr = qtable_ptr + angle_idx * ACTION_NUM; // one row of weights

    // Check if the previous action was any good
    reward = getReward(actual, reference);

    // Decide a new action: get the best known action or explore
    uint16_t action_idx = getRandom() > epsilon ? findMax(p_qtable[0][angle_idx]).idx : getRandomInteger(0, ACTION_NUM - 1);
    action = actions[action_idx];
    last_action_idx = action_idx;

    // Update the state of the instance.
    // Must be updated before touching the Q-table, because table-update is based on the last action.
    update(actual, angle_idx);
    
    // Update the Q-table 
    *Q_prev_ptr += alpha * (reward + gamma * findMax(Q_row_ptr).value - *Q_prev_ptr);

    return action;
}
